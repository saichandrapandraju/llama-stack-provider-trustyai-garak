from llama_stack.apis.eval import EvaluateResponse, BenchmarkConfig
from llama_stack.providers.datatypes import ProviderSpec
from llama_stack.apis.datatypes import Api
from llama_stack.apis.common.job_types import Job, JobStatus
from typing import List, Dict, Union
import os
import logging
import json
import asyncio
from ..config import GarakRemoteConfig
from ..base_eval import GarakEvalBase
from llama_stack_provider_trustyai_garak import shield_scan
from ..errors import GarakError, GarakConfigError, GarakValidationError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
JOB_ID_PREFIX = "garak-job-"


class GarakRemoteEvalAdapter(GarakEvalBase):
    """Remote Garak evaluation adapter for running scans on Kubeflow Pipelines."""

    def __init__(self, config: GarakRemoteConfig, deps: dict[Api, ProviderSpec]):
        super().__init__(config, deps)
        self._config: GarakRemoteConfig = config
        
        # S3 configuration - parse bucket and prefix from results_s3_prefix (format: bucket/prefix)
        self._s3_bucket: str = None
        self._s3_prefix: str = None
        self.s3_client = None
        self.kfp_client = None
        self._jobs_lock = asyncio.Lock()  # Will be initialized in async initialize()

    async def initialize(self) -> None:
        """Initialize the remote Garak provider."""
        self._initialize()
        
        # Parse S3 configuration from results_s3_prefix (format: bucket/prefix or s3://bucket/prefix)
        self._parse_s3_config()

        # # TODO: Do we REALLY need S3 client? Is there a better way to get the mapping from pod?
        # self._create_s3_client()
        
        # self._create_kfp_client()

        self._initialized = True
        logger.info("Initialized Garak remote provider.")
    
    def _parse_s3_config(self):
        """Parse S3 bucket and prefix from results_s3_prefix.
        
        Raises:
            GarakConfigError: If results_s3_prefix is invalid or missing
        """
        results_s3_prefix = self._config.kubeflow_config.results_s3_prefix
        
        # Validate input
        if not results_s3_prefix or not results_s3_prefix.strip():
            raise GarakConfigError(
                "results_s3_prefix must be specified in kubeflow_config. "
                "Format: 'bucket/prefix' or 's3://bucket/prefix'"
            )
        
        results_s3_prefix = results_s3_prefix.strip()
        
        # Handle s3://bucket/prefix format
        if results_s3_prefix.lower().startswith("s3://"):
            results_s3_prefix = results_s3_prefix[len("s3://"):]
        
        # validate format after stripping s3:// prefix
        if not results_s3_prefix:
            raise GarakConfigError(
                "results_s3_prefix cannot be just 's3://'. "
                "Format: 'bucket/prefix' or 's3://bucket/prefix'"
            )
        
        # Split bucket and prefix
        parts = results_s3_prefix.split("/", 1)
        self._s3_bucket = parts[0].strip()
        self._s3_prefix = parts[1].strip() if len(parts) > 1 else ""
        
        # validate bucket name is not empty
        if not self._s3_bucket:
            raise GarakConfigError(
                f"Invalid S3 bucket name in results_s3_prefix: '{self._config.kubeflow_config.results_s3_prefix}'. "
                "Bucket name cannot be empty."
            )
        
        
        logger.debug(f"Parsed S3 config - bucket: {self._s3_bucket}, prefix: {self._s3_prefix}")
    
    def _ensure_s3_client(self):
        if not self.s3_client:
            self._create_s3_client()
    
    def _ensure_kfp_client(self):
        if not self.kfp_client:
            self._create_kfp_client()
    
    def _create_s3_client(self):
        try:
            import boto3
            self.s3_client = boto3.client('s3',
                                        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                                        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                                        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
                                        endpoint_url=os.getenv('AWS_S3_ENDPOINT'),  # if using MinIO
                                        verify=self._verify_ssl,
                                    )
        except ImportError as e:
            raise GarakError(
                "Boto3 is not installed. Install with: pip install boto3"
            ) from e
        except Exception as e:
            raise GarakError(
                f"Unable to connect to S3."
            ) from e

    def _create_kfp_client(self):
        try:
            from kfp import Client
            from kfp_server_api.exceptions import ApiException

            ssl_cert = None
            if isinstance(self._verify_ssl, str):
                ssl_cert = self._verify_ssl
                verify_ssl = True
            else:
                verify_ssl = self._verify_ssl

            # Use token from config if provided, otherwise get from kubeconfig
            token = self._config.kubeflow_config.pipelines_api_token or self._get_token()
            
            self.kfp_client = Client(
                host=self._config.kubeflow_config.pipelines_endpoint,
                existing_token=token,
                verify_ssl=verify_ssl,
                ssl_ca_cert=ssl_cert
            )
        except ImportError:
            raise GarakError(
                "Kubeflow Pipelines SDK not available. Install with: pip install -e .[remote]"
            )
        except ApiException as e:
            raise GarakError(
                "Unable to connect to Kubeflow Pipelines. Please check if you are logged in to the correct cluster. "
                "If you are logged in, please check if the token & pipeline endpoint is valid. "
                "If you are not logged in, please run `oc login` command first."
            ) from e
        except Exception as e:
            raise GarakError(
                f"Unable to connect to Kubeflow Pipelines."
            ) from e
    
    def _get_token(self) -> str:
        """Get authentication token from kubernetes config."""
        try:
            from kubernetes.client.configuration import Configuration
            from kubernetes.config.kube_config import load_kube_config
            from kubernetes.config.config_exception import ConfigException
            from kubernetes.client.exceptions import ApiException

            config = Configuration()

            load_kube_config(client_configuration=config)
            token = config.api_key["authorization"].split(" ")[-1]
            return token
        except (KeyError, ConfigException) as e:
            raise ApiException(
                401, "Unauthorized, try running command like `oc login` first"
            ) from e
        except ImportError as e:
            raise GarakError(
                "Kubernetes client is not installed. Install with: pip install kubernetes"
            ) from e
    
    async def run_eval(self, benchmark_id: str, benchmark_config: BenchmarkConfig) -> Dict[str, Union[str, Dict[str, str]]]:
        """Run an evaluation for a specific benchmark and configuration.

        Args:
            benchmark_id: The benchmark id
            benchmark_config: Configuration for the evaluation task
            
        Raises:
            GarakValidationError: If benchmark_id or benchmark_config are invalid
            GarakConfigError: If configuration is invalid
            GarakError: If KFP pipeline creation fails
        """
        if not self._initialized:
            await self.initialize()
        
        if not benchmark_id or not isinstance(benchmark_id, str):
            raise GarakValidationError("benchmark_id must be a non-empty string")
        if not benchmark_config:
            raise GarakValidationError("benchmark_config cannot be None")
        
        _, benchmark_metadata = await self._validate_run_eval_request(benchmark_id, benchmark_config)
        
        job_id = self._get_job_id(prefix=JOB_ID_PREFIX)
        job = Job(
            job_id=job_id,
            status=JobStatus.scheduled
        )
        
        async with self._jobs_lock:
            self._jobs[job_id] = job
            self._job_metadata[job_id] = {}  # Initialize metadata dict

        try:
            scan_profile_config: dict = {
                "probes": benchmark_metadata["probes"],
                "timeout": benchmark_metadata.get("timeout", self._config.timeout)
            }

            cmd: List[str] = await self._build_command(benchmark_config, benchmark_id, scan_profile_config)

            from .kfp_utils.pipeline import garak_scan_pipeline
            
            self._ensure_kfp_client()
            
            # Validate config before creating pipeline
            if not self._config.kubeflow_config.namespace:
                raise GarakConfigError("kubeflow_config.namespace is not configured")
            if not self._config.kubeflow_config.s3_credentials_secret_name:
                raise GarakConfigError("kubeflow_config.s3_credentials_secret_name is not configured")
            if not self._config.llama_stack_url:
                raise GarakConfigError("llama_stack_url is not configured")
            
            experiment_name = f"trustyai-garak-{self._config.kubeflow_config.namespace}"
            os.environ['KUBEFLOW_S3_CREDENTIALS_SECRET_NAME'] = self._config.kubeflow_config.s3_credentials_secret_name
            
            llama_stack_url = self._config.llama_stack_url.strip().rstrip("/")
            if not llama_stack_url:
                raise GarakConfigError("llama_stack_url cannot be empty after normalization")
            
            run = self.kfp_client.create_run_from_pipeline_func(
                garak_scan_pipeline,
                arguments={
                    "command": cmd,
                    "llama_stack_url": llama_stack_url,
                    "job_id": job_id,
                    "eval_threshold": float(benchmark_metadata.get("eval_threshold", self.scan_config.VULNERABLE_SCORE)),
                    "timeout_seconds": int(scan_profile_config.get("timeout", self._config.timeout)),
                    "max_retries": int(benchmark_metadata.get("max_retries", 3)),
                    "use_gpu": benchmark_metadata.get("use_gpu", False),
                    "verify_ssl": str(self._verify_ssl),
                    "s3_bucket": self._s3_bucket,
                    "s3_prefix": self._s3_prefix,
                },
                run_name=f"garak-{benchmark_id.split('::')[-1]}-{job_id.removeprefix(JOB_ID_PREFIX)}",
                namespace=self._config.kubeflow_config.namespace,
                experiment_name=experiment_name
            )
            
            async with self._jobs_lock:
                self._job_metadata[job_id] = {
                    "created_at": self._convert_datetime_to_str(run.run_info.created_at), 
                    "kfp_run_id": run.run_id}
                
            return {"job_id": job_id, "status": job.status, "metadata": self._job_metadata.get(job_id, {})}
        except Exception as e:
            logger.error(f"Error running eval for {benchmark_id}: {e}")
            async with self._jobs_lock:
                job.status = JobStatus.failed
                self._job_metadata[job.job_id]["error"] = str(e)
            raise e

    def _map_kfp_run_state_to_job_status(self, run_state) -> JobStatus:
        """Map the KFP run state to the job status."""
        from kfp_server_api.models import V2beta1RuntimeState

        if run_state in [V2beta1RuntimeState.RUNTIME_STATE_UNSPECIFIED, V2beta1RuntimeState.PENDING]:
            return JobStatus.scheduled
        elif run_state in [V2beta1RuntimeState.RUNNING, V2beta1RuntimeState.CANCELING, V2beta1RuntimeState.PAUSED]:
            return JobStatus.in_progress
        elif run_state in [V2beta1RuntimeState.SUCCEEDED, V2beta1RuntimeState.SKIPPED]:
            return JobStatus.completed
        elif run_state == V2beta1RuntimeState.FAILED:
            return JobStatus.failed
        elif run_state == V2beta1RuntimeState.CANCELED:
            return JobStatus.cancelled
        else:
            logger.warning(f"KFP run has an unknown status: {run_state}, mapping to scheduled")
            return JobStatus.scheduled
    
    async def job_status(self, benchmark_id: str, job_id: str) -> Dict[str, Union[str, Dict[str, str]]]:
        """Get the status of a job.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id
        """
        from kfp_server_api.models import V2beta1Run
        
        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found")
                return {"status": "not_found", "job_id": job_id}
            
            metadata: dict = self._job_metadata.get(job_id, {})

            if "kfp_run_id" not in metadata:
                logger.warning(f"Job {job_id} has no kfp run id")
                return {"status": "not_found", "job_id": job_id}
            
            kfp_run_id = metadata["kfp_run_id"]
        
        try:
            self._ensure_kfp_client()
            run: V2beta1Run = self.kfp_client.get_run(kfp_run_id)
            new_status = self._map_kfp_run_state_to_job_status(run.state)
            
            async with self._jobs_lock:
                job.status = new_status
                if job.status in [JobStatus.completed, JobStatus.failed, JobStatus.cancelled]:
                    self._job_metadata[job_id]['finished_at'] = self._convert_datetime_to_str(run.finished_at)

                    if job.status == JobStatus.completed:
                        # Read from S3
                        try:
                            # Skip S3 reading if bucket not configured
                            if not self._s3_bucket:
                                logger.warning(f"S3 bucket not configured for job {job_id}, skipping file metadata retrieval")
                            else:
                                self._ensure_s3_client()

                                s3_key = f'{self._s3_prefix}/{job_id}.json' if self._s3_prefix else f'{job_id}.json'
                                
                                # Validate S3 object exists before reading
                                try:
                                    response = self.s3_client.get_object(
                                        Bucket=self._s3_bucket,
                                        Key=s3_key
                                    )
                                except Exception as s3_err:
                                    logger.warning(f"S3 object not found or inaccessible: {s3_key} in bucket {self._s3_bucket}: {s3_err}")
                                    raise
                                
                                # Read and parse response
                                body_content = response['Body'].read()
                                if not body_content:
                                    logger.warning(f"Empty S3 object for job {job_id}: {s3_key}")
                                    raise ValueError(f"Empty S3 object: {s3_key}")
                                
                                try:
                                    file_id_mapping: dict = json.loads(body_content)
                                except json.JSONDecodeError as e:
                                    logger.error(f"Failed to parse JSON from S3 object {s3_key}: {e}")
                                    logger.debug(f"S3 content (first 500 chars): {body_content[:500]}")
                                    raise
                                
                                if not isinstance(file_id_mapping, dict):
                                    logger.error(f"S3 object {s3_key} did not contain a dictionary: {type(file_id_mapping)}")
                                    raise ValueError(f"Invalid file mapping format in S3 object: {s3_key}")
                                
                                # Store the file IDs in job metadata
                                # logger.info(f"====== File mapping: {file_id_mapping} ======")
                                for key, value in file_id_mapping.items():
                                    self._job_metadata[job_id][key] = value
                                
                                # # Clean up - delete the file from S3
                                # self.s3_client.delete_object(
                                #     Bucket=self._s3_bucket,
                                #     Key=f'{job_id}.json'
                                # )
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON from S3 object {s3_key if 's3_key' in locals() else 'unknown'}: {e}")
                        except Exception as e:
                            logger.warning(f"Could not retrieve outputs from S3 for job {job_id}: {e}")
                
                return_metadata = self._job_metadata.get(job_id, {})
                return_status = job.status
        except Exception as e:
            logger.error(f"Error getting KFP run {kfp_run_id}: {e}")
            return {"status": "not_found", "job_id": job_id}
        
        return {"job_id": job_id, "status": return_status, "metadata": return_metadata}
    
    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        """Get the result of a job (remote-specific: updates job state from KFP first).

        Args:
            benchmark_id: The benchmark id
            job_id: The job id
        """
        # Update job status from KFP before getting results
        await self.job_status(benchmark_id, job_id)
        
        return await super().job_result(benchmark_id, job_id)
    
    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        """Cancel a job and kill the process.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id
        """
        # check/update the current status of the job
        current_status = await self.job_status(benchmark_id, job_id)
        if current_status["status"] == "not_found":
            return
        elif current_status["status"] in [JobStatus.completed, JobStatus.failed, JobStatus.cancelled]:
            logger.warning(f"Job {job_id} is not running. Can't cancel.")
            return
        else:
            async with self._jobs_lock:
                kfp_run_id = self._job_metadata[job_id].get("kfp_run_id")
            
            if kfp_run_id:
                try:
                    self._ensure_kfp_client()
                    self.kfp_client.terminate_run(kfp_run_id)
                except Exception as e:
                    logger.error(f"Error cancelling KFP run {kfp_run_id}: {e}")
    
    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Shutting down Garak provider")
        
        # Get snapshot of jobs to cancel
        async with self._jobs_lock:
            jobs_to_cancel = [(job_id, job) for job_id, job in self._jobs.items() 
                             if job.status in [JobStatus.in_progress, JobStatus.scheduled]]
        
        # Kill all running jobs
        for job_id, job in jobs_to_cancel:
            await self.job_cancel("placeholder", job_id)
        
        # # Clear all running tasks, jobs and job metadata
        async with self._jobs_lock:
            self._jobs.clear()
            self._job_metadata.clear()

        # Close the shield scanning HTTP client
        shield_scan.simple_shield_orchestrator.close()
        