from llama_stack.apis.eval import Eval, EvaluateResponse, BenchmarkConfig
from llama_stack.apis.inference import SamplingParams
from llama_stack.distribution.datatypes import Api
from llama_stack.apis.files import Files, OpenAIFilePurpose, OpenAIFileObject
from fastapi import UploadFile, Response
from llama_stack.providers.datatypes import BenchmarksProtocolPrivate
from llama_stack.apis.benchmarks import Benchmark, ListBenchmarksResponse
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.scoring import ScoringResult
from typing import List, Dict, Optional, Any, Union, Set
import os
import logging
import json
from pathlib import Path
from .config import GarakEvalProviderConfig, GarakScanConfig, AttackType
from datetime import datetime
import asyncio
import signal
import shutil

logger = logging.getLogger(__name__)

class GarakEvalAdapter(Eval, BenchmarksProtocolPrivate):
    def __init__(self, config: GarakEvalProviderConfig, deps):
        super().__init__(config)
        self._config: GarakEvalProviderConfig = config
        self.file_api: Files = deps[Api.files]
        self.scan_config = GarakScanConfig()
        self.benchmarks: Dict[str, Benchmark] = {} # benchmark_id -> benchmark
        self.all_probes: Set[str] = set()
        self._jobs: Dict[str, Job] = {}
        self._job_metadata: Dict[str, Dict[str, str]] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {} # job_id -> running task
        self._initialized: bool = False

    async def initialize(self) -> None:
        """Initialize the Garak provider"""
        logger.info("Initializing Garak provider")

        self.scan_config.scan_dir.mkdir(exist_ok=True, parents=True)
        
        self._ensure_garak_installed()
        self.all_probes = self._get_all_probes()

        self._initialized = True
        
    def _ensure_garak_installed(self):
        """Ensure garak is installed"""
        try:
            import garak
        except ImportError:
            raise ImportError(
                "Garak is not installed. Please install it with: "
                "pip install garak"
            )
    
    def _get_all_probes(self) -> Set[str]:
        ## unfortunately, garak don't have a public API to list all probes
        ## so we need to enumerate all probes manually from private API
        from garak._plugins import enumerate_plugins
        probes_names = enumerate_plugins(category="probes", skip_base_classes=True)
        plugin_names = [p.replace(f"probes.", "") for p, _ in probes_names]
        module_names = set([m.split(".")[0] for m in plugin_names])
        plugin_names += module_names
        return set(plugin_names)

    async def list_benchmarks(self) -> ListBenchmarksResponse:
        """List all benchmarks"""
        return ListBenchmarksResponse(data=list(self.benchmarks.values()))
    
    async def get_benchmark(self, benchmark_id: str) -> Optional[Benchmark]:
        """Get a benchmark by its id"""
        return self.benchmarks.get(benchmark_id)
    
    async def register_benchmark(self, benchmark: Benchmark) -> None:
        """Register a benchmark"""
        self.benchmarks[benchmark.identifier] = benchmark
    
    def _get_job_id(self) -> str:
        """Generate a unique job ID.

        Returns:
            Unique job ID string
        """
        import uuid

        return f"garak-job-{str(uuid.uuid4())}"
    
    async def run_eval(self, benchmark_id: str, benchmark_config: BenchmarkConfig) -> Dict[str, Union[str, Dict[str, str]]]:
        """Run an evaluation for a specific benchmark and configuration.

        Args:
            benchmark_id: The benchmark id
            benchmark_config: Configuration for the evaluation task
        """
        if not self._initialized:
            await self.initialize()
        
        if not isinstance(benchmark_config, BenchmarkConfig):
            raise TypeError("Required benchmark_config to be of type BenchmarkConfig")
        
        job_id = self._get_job_id()
        job = Job(
            job_id=job_id,
            status=JobStatus.scheduled
        )
        self._jobs[job_id] = job
        self._job_metadata[job_id] = {"created_at": datetime.now().isoformat()}

        self._running_tasks[job_id] = asyncio.create_task(self._run_scan(job, benchmark_id, benchmark_config), name=f"garak-job-{job_id}")
        return {"job_id": job_id, "status": job.status, "metadata": self._job_metadata.get(job_id, {})}
        
    async def _run_scan(self, job: Job, benchmark_id: str, benchmark_config: BenchmarkConfig):
        """Run the scan with the given command.

        Args:
            job: The job object
            benchmark_id: The benchmark id
            benchmark_config: The benchmark configuration
        """
        
        stored_benchmark = await self.get_benchmark(benchmark_id)
        benchmark_metadata: dict = getattr(stored_benchmark, "metadata", {})

        job.status = JobStatus.in_progress
        self._job_metadata[job.job_id]["started_at"] = datetime.now().isoformat()

        job_scan_dir: Path = self.scan_config.scan_dir / job.job_id
        job_scan_dir.mkdir(exist_ok=True, parents=True)

        scan_log_file: Path = job_scan_dir / "scan.log"
        scan_log_file.touch(exist_ok=True)
        scan_report_prefix: Path = job_scan_dir / "scan"
        
        try:
            scan_profile:str = benchmark_id.split("::")[-1].strip()
            if not scan_profile:
                scan_profile = "custom"

            if scan_profile not in self.scan_config.SCAN_PROFILES:
                if not benchmark_metadata.get("probes", None):
                    raise ValueError("No probes found for benchmark. Please specify probes list in the benchmark metadata.")
                
                scan_profile_config:dict = {
                    "probes": benchmark_metadata["probes"],
                    "timeout": benchmark_metadata.get("timeout", self._config.timeout)
                }
            else:
                scan_profile_config:dict = self.scan_config.SCAN_PROFILES[scan_profile]


            cmd: List[str] = self._build_command(benchmark_config, str(scan_report_prefix), scan_profile_config)
            logger.info(f"Running scan with command: {' '.join(cmd)}")

            env = os.environ.copy()
            env["GARAK_LOG_FILE"] = str(scan_log_file)

            process = await asyncio.create_subprocess_exec(*cmd, 
                                                           stdout=asyncio.subprocess.PIPE, 
                                                           stderr=asyncio.subprocess.PIPE, 
                                                           env=env)
            
            self._job_metadata[job.job_id]["process_id"] = str(process.pid)
            timeout: int = scan_profile_config.get("timeout", self._config.timeout)
            
            _, stderr = await asyncio.wait_for(process.communicate(), 
                                                    timeout=timeout)

            if process.returncode == 0:
                # Upload scan files to file storage
                upload_scan_report: OpenAIFileObject = await self._upload_file(
                    file=scan_report_prefix.with_suffix(".report.jsonl"), 
                    purpose=OpenAIFilePurpose.ASSISTANTS)
                if upload_scan_report:
                    self._job_metadata[job.job_id]["scan_report_file_id"] = upload_scan_report.id

                upload_scan_log: OpenAIFileObject = await self._upload_file(
                    file=scan_log_file, 
                    purpose=OpenAIFilePurpose.ASSISTANTS)
                if upload_scan_log:
                    self._job_metadata[job.job_id]["scan_log_file_id"] = upload_scan_log.id

                upload_scan_hitlog: OpenAIFileObject = await self._upload_file(
                    file=scan_report_prefix.with_suffix(".hitlog.jsonl"), 
                    purpose=OpenAIFilePurpose.ASSISTANTS)
                if upload_scan_hitlog:
                    self._job_metadata[job.job_id]["scan_hitlog_file_id"] = upload_scan_hitlog.id

                upload_scan_report_html: OpenAIFileObject = await self._upload_file(
                    file=scan_report_prefix.with_suffix(".report.html"), 
                    purpose=OpenAIFilePurpose.ASSISTANTS)
                if upload_scan_report_html:
                    self._job_metadata[job.job_id]["scan_report_html_file_id"] = upload_scan_report_html.id

                job.status = JobStatus.completed
                # cleanup the tmp job dir
                shutil.rmtree(job_scan_dir, ignore_errors=True)

            else:
                job.status = JobStatus.failed
                self._job_metadata[job.job_id]["error"] = f"Scan failed with return code {process.returncode} - {stderr.decode('utf-8')}"
        except asyncio.TimeoutError:
            job.status = JobStatus.failed
            self._job_metadata[job.job_id]["error"] = f"Scan timed out after {timeout} seconds."
        except Exception as e:
            job.status = JobStatus.failed
            self._job_metadata[job.job_id]["error"] = str(e)
        finally:
            self._job_metadata[job.job_id]["completed_at"] = datetime.now().isoformat()
            if 'process' in locals() and process.returncode is None:
                process.kill()
                await process.wait()
            self._running_tasks.pop(job.job_id, None)

    async def _upload_file(self, file: Path, purpose: OpenAIFilePurpose) -> Optional[OpenAIFileObject]:
        """Upload a file to the file storage and return the file object.

        Args:
            file: The file to upload
            purpose: The purpose of the file
        """
        if file.exists():
            with open(file, "rb") as f:
                upload_file: OpenAIFileObject = await self.file_api.openai_upload_file(
                    # file: The File object (not file name) to be uploaded.
                    file=UploadFile(file=f, filename=file.name), 
                    purpose=purpose
                )
                return upload_file
        else:
            logger.warning(f"File {file} does not exist")
            return None
        
    def _build_command(self, benchmark_config: BenchmarkConfig, scan_report_prefix: str, scan_profile_config: dict) -> List[str]:
        """Build the command to run the scan.

        Args:
            benchmark_config: Configuration for the evaluation task
            scan_report_prefix: Prefix for the scan report
            scan_log_file: Path to the scan log file
            scan_profile_config: Configuration for the scan profile
        """
        generator_options:dict = self._get_generator_options(benchmark_config)
        cmd: List[str] = ["garak",
                          "--model_type", self._config.garak_model_type,
                          "--model_name", benchmark_config.eval_candidate.model,
                          "--generations", "1",
                          "--generator_options", json.dumps(generator_options),
                          "--report_prefix", scan_report_prefix.strip(),
                          "--parallel_attempts", str(self.scan_config.parallel_probes)
                          ]
        # Add probes
        probes = scan_profile_config["probes"]
        if probes != ["all"]:
            for probe in probes:
                if probe not in self.all_probes:
                    raise ValueError(f"Probe '{probe}' not found in garak. "
                                     "Please provide valid garak probe name. "
                                     "Or you can just use predefined scan profiles ('quick', 'standard', 'comprehensive') as benchmark_id.")
            cmd.extend(["--probes", ",".join(probes)])
        return cmd
    
    def _get_generator_options(self, benchmark_config: BenchmarkConfig) -> dict:

        base_url: str = self._config.base_url.rstrip("/")
        if not base_url.endswith("openai/v1"):
            if base_url.endswith("/v1"):
                base_url = f"{base_url}/openai/v1"
            else:
                base_url = f"{base_url}/v1/openai/v1"


        generator_options = {
                    "openai": {
                        "OpenAICompatible": {
                            "uri": base_url,
                            "model": benchmark_config.eval_candidate.model,
                            "api_key": os.getenv("OPENAICOMPATIBLE_API_KEY", "DUMMY"),
                            "suppressed_params": ["n"]
                        }
                    }
                }
        # Add extra params
        sampling_params: SamplingParams = benchmark_config.eval_candidate.sampling_params
        if sampling_params:
            generator_options["openai"]["OpenAICompatible"].update(sampling_params.model_dump())
        return generator_options
    
    async def _parse_scan_results(self, report_file_id: str, job_id: str) -> EvaluateResponse:
        """Parse the scan results from the report file.
        
        Args:
            report_file_id: The id of the report file
            job_id: The id of the job
        """
        report_file_content: Response = await self.file_api.openai_retrieve_file_content(report_file_id)
        if not report_file_content:
            logger.warning(f"Report file not found for job {job_id}")
            return EvaluateResponse(generations=[], scores={})
        
        try:
            generations: List[Dict[str, Any]] = []
            # Each key is a garak probe/attack_type and value is a ScoringResult object containing scores from respective garak detectors
            scores: Dict[str, ScoringResult] = {} 

            score_rows: Dict[str, List[Dict[str, float]]] = {}
            aggregated_results_by_probe: Dict[str, Dict[str, List[float]]] = {}
            report_file_content_str: str = report_file_content.body.decode("utf-8")

            for line in report_file_content_str.split("\n"):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    generation: Dict[str, Any] = {}

                    # Handle attempt entries
                    if (entry.get("entry_type") == "attempt" and entry.get("status") == 2):
                        probe_name: str = entry.get("probe_classname", "unknown")
                        probe_category: str = probe_name.split('.')[0]

                        detector_results: dict = entry.get("detector_results", {})
                        detector_maxscore_map: Dict[str, float] = {}
                        is_vulnerable: bool = False
                        
                        for detector, _scores in detector_results.items():
                            _scores = _scores if isinstance(_scores, list) else [_scores]
                            if any(score >= self.scan_config.VULNERABLE_SCORE for score in _scores):
                                is_vulnerable = True
                            detector_maxscore_map[detector] = max(_scores) if _scores else 0
                        outputs: list = entry.get("outputs", [])

                        generation["probe"] = probe_name
                        generation["attack_type"] = self.scan_config.PROBE_TO_ATTACK.get(probe_category, AttackType.CUSTOM)
                        generation["vulnerable"] = is_vulnerable
                        generation["prompt"] = entry.get("prompt", "")
                        generation["response"] = outputs[0] if len(outputs) > 0 and outputs[0] else ""
                        generations.append(generation)

                        if probe_name not in score_rows:
                            score_rows[probe_name] = []
                        score_rows[probe_name].append(detector_maxscore_map)

                        if probe_name not in aggregated_results_by_probe:
                            aggregated_results_by_probe[probe_name] = {}
                        for detector, score in detector_maxscore_map.items():
                            if detector not in aggregated_results_by_probe[probe_name]:
                                aggregated_results_by_probe[probe_name][detector] = []
                            aggregated_results_by_probe[probe_name][detector].append(score)
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON line in report file for job {job_id}: {line} - {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error parsing line in report file for job {job_id}: {line} - {e}")
                    continue

            # Calculate the mean of the scores for each probe
            aggregated_results_mean: Dict[str, Dict[str, float]] = {}
            for probe_name, results in aggregated_results_by_probe.items():
                aggregated_results_mean[probe_name] = {}
                for detector, _scores in results.items():
                    detector_mean_score: float = round(sum(_scores) / len(_scores), 3) if _scores else 0
                    aggregated_results_mean[probe_name][f"{detector}_mean"] = detector_mean_score

            if len(aggregated_results_mean.keys()) != len(score_rows.keys()):
                # FIXME: Change to proper error type
                raise ValueError(f"Number of probes in aggregated results ({len(aggregated_results_mean.keys())}) "
                                    f"does not match number of probes in score rows ({len(score_rows.keys())})")
            
            all_probes: List[str] = list(aggregated_results_mean.keys())
            for probe_name in all_probes:
                scores[probe_name] = ScoringResult(
                    score_rows=score_rows[probe_name],
                    aggregated_results=aggregated_results_mean[probe_name]
                    )

            return EvaluateResponse(generations=generations, scores=scores)    

        except Exception as e:
            logger.error(f"Error parsing scan results for job {job_id}: {e}")
            return EvaluateResponse(generations=[], scores={})

    async def job_status(self, benchmark_id: str, job_id: str) -> Dict[str, Union[str, Dict[str, str]]]:
        """Get the status of a job.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id
        """
        job = self._jobs.get(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return {"status": "not_found", "job_id": job_id}
        return {"job_id": job_id, "status": job.status, "metadata": self._job_metadata.get(job_id, {})}
    
    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        """Get the result of a job.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id
        """
        job = self._jobs.get(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return EvaluateResponse(generations=[], scores={})
        
        if job.status in [JobStatus.scheduled, JobStatus.in_progress]:
            logger.warning(f"Job {job_id} is not completed")
            return EvaluateResponse(generations=[], scores={})
        
        elif job.status == JobStatus.failed:
            logger.warning(f"Job {job_id} failed")
            return EvaluateResponse(generations=[], scores={})
        
        elif job.status == JobStatus.cancelled:
            logger.warning(f"Job {job_id} was cancelled")
            return EvaluateResponse(generations=[], scores={})
        
        elif job.status == JobStatus.completed:
            if self._job_metadata[job_id].get("results", None):
                return EvaluateResponse(**self._job_metadata[job_id]["results"])

            scan_report_file_id: str = self._job_metadata[job_id].get("scan_report_file_id", "")
            if not scan_report_file_id:
                logger.warning(f"No scan report file found for job {job_id}")
                return EvaluateResponse(generations=[], scores={})
            try:
                results:EvaluateResponse = await self._parse_scan_results(report_file_id = scan_report_file_id, job_id = job_id)
            except Exception as e:
                logger.error(f"Error parsing scan results for job {job_id}: {e}")
                return EvaluateResponse(generations=[], scores={})

            # storing all Job results in memory.. 
            # FIXME: Upload results to file storage? or parse the results from the scan report file every time?
            self._job_metadata[job_id]["results"] = results.model_dump()
            return results
        
        else:
            logger.warning(f"Job {job_id} has an unknown status: {job.status}")
            return EvaluateResponse(generations=[], scores={})
    
    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        """Cancel a job and kill the process.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id
        """
        job = self._jobs.get(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return
        
        if job.status in [JobStatus.completed, JobStatus.failed, JobStatus.cancelled]:
            logger.warning(f"Job {job_id} is not running")

        elif job.status in [JobStatus.in_progress, JobStatus.scheduled]:
            process_id: str = self._job_metadata[job_id].get("process_id", None)
            if process_id:
                process_id: int = int(process_id)
                logger.info(f"Killing process {process_id} for job {job_id}")
                try:
                    # TODO: Check if the process is graceful shutdown and if not, kill it with SIGKILL
                    os.kill(process_id, signal.SIGTERM)
                except ProcessLookupError:
                    logger.warning(f"Process {process_id} not found for job {job_id}")
                except Exception as e:
                    logger.error(f"Error killing process {process_id} for job {job_id}: {e}")
            job.status = JobStatus.cancelled
            self._jobs[job_id] = job
            self._job_metadata[job_id]["cancelled_at"] = datetime.now().isoformat()
            self._job_metadata[job_id]["error"] = "Job cancelled"
        else:
            logger.warning(f"Job {job_id} has an unknown status: {job.status}")
    
    async def evaluate_rows(self, benchmark_id: str, 
                            input_rows: list[dict[str, Any]], 
                            scoring_functions: list[str], 
                            benchmark_config: BenchmarkConfig) -> EvaluateResponse:
        raise NotImplementedError("Not implemented")
    
    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Shutting down Garak provider")
        # Cancel all running asyncio tasks
        for job_id, task in self._running_tasks.items():
            if not task.done():
                logger.info(f"Cancelling running task {task.get_name()} for job {job_id}")
                task.cancel()
        
        # Wait for tasks to be cancelled (with timeout)
        if self._running_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._running_tasks.values(), return_exceptions=True),
                    timeout=5
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks didn't cancel within timeout")
        
        # Kill all running jobs
        for job_id, job in self._jobs.items():
            if job.status in [JobStatus.in_progress, JobStatus.scheduled]:
                await self.job_cancel("placeholder", job_id)
        
        # Clear all running tasks, jobs and job metadata
        self._running_tasks.clear()
        self._jobs.clear()
        self._job_metadata.clear()
        
        # Cleanup the scan directory
        shutil.rmtree(self.scan_config.scan_dir, ignore_errors=True)