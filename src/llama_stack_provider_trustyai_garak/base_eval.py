"""Base class for Garak evaluation adapters with common functionality."""

from llama_stack.apis.eval import Eval, BenchmarkConfig, EvaluateResponse
from llama_stack.providers.datatypes import ProviderSpec, BenchmarksProtocolPrivate
from llama_stack.apis.datatypes import Api
from llama_stack.apis.files import Files
from llama_stack.apis.benchmarks import Benchmark, Benchmarks
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.safety import Safety
from llama_stack.apis.shields import Shields
from typing import Dict, Optional, Set, List, Any, Union
import logging
import uuid
from .config import GarakScanConfig, GarakInlineConfig, GarakRemoteConfig
from .errors import GarakError, GarakConfigError, GarakValidationError, BenchmarkNotFoundError

logger = logging.getLogger(__name__)


class GarakEvalBase(Eval, BenchmarksProtocolPrivate):
    """Base class for Garak evaluation adapters containing common functionality."""

    def __init__(self, config: Union[GarakInlineConfig, GarakRemoteConfig], deps: dict[Api, ProviderSpec]):
        super().__init__(config)
        self._config = config
        self.file_api: Optional[Files] = deps.get(Api.files)
        self.benchmarks_api: Optional[Benchmarks] = deps.get(Api.benchmarks)
        self.safety_api: Optional[Safety] = deps.get(Api.safety)
        self.shields_api: Optional[Shields] = deps.get(Api.shields)
        self.scan_config = GarakScanConfig()
        self.benchmarks: Dict[str, Benchmark] = {}  # benchmark_id -> benchmark
        self.all_probes: Set[str] = set()
        self._verify_ssl = None
        self._initialized: bool = False

        self._jobs: Dict[str, Job] = {}
        self._job_metadata: Dict[str, Dict[str, str]] = {}
    
    def _initialize(self) -> None:
        """Initialize the Garak provider."""
        self._validate_apis()
        self._ensure_garak_installed()
        self.all_probes = self._get_all_probes()

        self._verify_ssl = self._config.tls_verify
        if isinstance(self._verify_ssl, str):
            if self._verify_ssl.lower() in ("true", "1", "yes", "on"):
                self._verify_ssl = True
            elif self._verify_ssl.lower() in ("false", "0", "no", "off"):
                self._verify_ssl = False

    def _validate_apis(self) -> None:
        """Validate that required APIs are provided and optional APIs are consistent."""
        error_msg_template = (
            "{provided_api_name} API is provided but {missing_api_name} API is not provided. "
            "Please provide both APIs."
        )
        if self.shields_api and not self.safety_api:
            raise GarakConfigError(
                error_msg_template.format(
                    provided_api_name="Shields", missing_api_name="Safety"
                )
            )
        elif not self.shields_api and self.safety_api:
            raise GarakConfigError(
                error_msg_template.format(
                    provided_api_name="Safety", missing_api_name="Shields"
                )
            )

    def _ensure_garak_installed(self) -> None:
        """Ensure garak is installed."""
        try:
            import garak  # noqa: F401
        except ImportError:
            raise GarakError(
                "Garak is not installed. Please install it with: pip install garak"
            )

    def _get_all_probes(self) -> Set[str]:
        """Get all available Garak probes.
        
        Returns:
            Set of probe names (both full names and module names)
        """
        # Unfortunately, garak doesn't have a public API to list all probes
        # so we need to enumerate all probes manually from private API
        try:
            from garak._plugins import enumerate_plugins
        except ImportError:
            raise GarakError(
                "Unable to import garak's enumerate_plugins. The internal API may have changed."
            )

        probes_names = enumerate_plugins(category="probes", skip_base_classes=True)
        plugin_names = [p.replace("probes.", "") for p, _ in probes_names]
        module_names = set([m.split(".")[0] for m in plugin_names])
        plugin_names += list(module_names)
        return set(plugin_names)

    def _resolve_framework_to_probes(self, framework_id: str) -> List[str]:
        """Resolve a framework ID to a list of probes using taxonomy filters.
        
        Args:
            framework_id: The framework identifier (e.g., 'trustyai_garak::owasp_llm_top10')
            
        Returns:
            List of probe names that match the framework's taxonomy filters
            
        Raises:
            GarakValidationError: If framework is unknown
            GarakError: If garak's parse_plugin_spec cannot be imported
        """
        framework_info = self.scan_config.FRAMEWORK_PROFILES.get(framework_id)
        if not framework_info:
            raise GarakValidationError(f"Unknown framework: {framework_id}")

        taxonomy_filters = framework_info.get("taxonomy_filters", [])
        if not taxonomy_filters:
            logger.warning(f"No taxonomy filters defined for framework {framework_id}")
            return []

        # Import garak's config parsing functionality (unfortunately not public API)
        try:
            from garak._config import parse_plugin_spec
        except ImportError:
            raise GarakError(
                "Unable to import garak's parse_plugin_spec. The internal API may have changed."
            )

        resolved_probes = []

        # For each taxonomy filter, get matching probes
        for tag_filter in taxonomy_filters:
            try:
                # Use garak's built-in tag filtering
                probes, _ = parse_plugin_spec("all", "probes", probe_tag_filter=tag_filter)
                resolved_probes.extend(probes)

            except Exception as e:
                logger.error(f"Error resolving probes for tag filter '{tag_filter}': {e}")
                continue

        # Remove duplicates and garak prefix
        unique_probes = list(set([p.replace("probes.", "") for p in resolved_probes]))
        unique_probes.sort()

        logger.info(
            f"Framework '{framework_id}' resolved to {len(unique_probes)} probes: "
            f"{unique_probes[:5]}{'...' if len(unique_probes) > 5 else ''}"
        )

        return unique_probes

    async def get_benchmark(self, benchmark_id: str) -> Optional[Benchmark]:
        """Get a benchmark by its id.
        
        Args:
            benchmark_id: The benchmark identifier
            
        Returns:
            Benchmark object if found, None otherwise
        """
        return await self.benchmarks_api.get_benchmark(benchmark_id)

    async def register_benchmark(self, benchmark: Benchmark) -> None:
        """Register a benchmark by checking if it's a pre-defined scan profile or compliance framework.
        
        Args:
            benchmark: The benchmark to register
        """
        if benchmark.identifier in (
            self.scan_config.SCAN_PROFILES | self.scan_config.FRAMEWORK_PROFILES
        ):
            logger.info(
                f"Benchmark '{benchmark.identifier}' is a pre-defined scan profile or compliance framework. "
                f"It is not recommended to register it as a custom benchmark."
            )

        if not benchmark.metadata:
            logger.info(
                f"Benchmark '{benchmark.identifier}' is pre-defined but has no metadata provided. "
                f"Using default metadata."
            )
            benchmark.metadata = self.scan_config.SCAN_PROFILES.get(
                benchmark.identifier, {}
            ) or self.scan_config.FRAMEWORK_PROFILES.get(benchmark.identifier, {})

        if not benchmark.metadata.get("probes"):
            if benchmark.identifier in self.scan_config.SCAN_PROFILES:
                logger.info(
                    f"Benchmark '{benchmark.identifier}' is a pre-defined legacy scan profile "
                    f"but has no probes provided. Using default probes."
                )
                benchmark.metadata["probes"] = self.scan_config.SCAN_PROFILES[
                    benchmark.identifier
                ]["probes"]
            elif benchmark.identifier in self.scan_config.FRAMEWORK_PROFILES:
                logger.info(
                    f"Benchmark '{benchmark.identifier}' is a pre-defined compliance framework "
                    f"but has no probes provided. Resolving probes from taxonomy."
                )
                benchmark.metadata["probes"] = self._resolve_framework_to_probes(
                    benchmark.identifier
                )

        self.benchmarks[benchmark.identifier] = benchmark

    def _get_job_id(self, prefix: str = "garak-job-") -> str:
        """Generate a unique job ID.
        
        Args:
            prefix: Prefix for the job ID
            
        Returns:
            Unique job ID string
        """
        return f"{prefix}{str(uuid.uuid4())}"

    def _normalize_list_arg(self, arg: Union[str, List[str]]) -> str:
        """Normalize a list argument to a comma-separated string.
        
        Args:
            arg: String or list of strings
            
        Returns:
            Comma-separated string
        """
        return arg if isinstance(arg, str) else ",".join(arg)

    async def _get_generator_options(self, benchmark_config: "BenchmarkConfig", benchmark_metadata: dict) -> dict:
        """Get the generator options based on the availability of shields.
        
        Args:
            benchmark_config: Configuration for the evaluation task
            benchmark_metadata: Metadata from the benchmark
            
        Returns:
            Generator options dictionary
        """
        if bool(benchmark_metadata.get("shield_ids", []) or benchmark_metadata.get("shield_config", {})):
            return await self._get_function_based_generator_options(benchmark_config, benchmark_metadata)
        else:
            return await self._get_openai_compatible_generator_options(benchmark_config, benchmark_metadata)

    async def _get_openai_compatible_generator_options(
        self, benchmark_config: "BenchmarkConfig", benchmark_metadata: dict
    ) -> dict:
        """Get the generator options for the OpenAI compatible generator.
        
        Args:
            benchmark_config: Configuration for the evaluation task
            benchmark_metadata: Metadata from the benchmark
            
        Returns:
            Generator options for OpenAI compatible model
        """
        import os
        
        if 'generator_options' in benchmark_metadata:
            return benchmark_metadata['generator_options']
        
        llama_stack_url: str = self._get_llama_stack_url()

        generator_options = {
            "openai": {
                "OpenAICompatible": {
                    "uri": llama_stack_url,
                    "model": benchmark_config.eval_candidate.model,
                    "api_key": os.getenv("OPENAICOMPATIBLE_API_KEY", "DUMMY"),
                    "suppressed_params": ["n"]
                }
            }
        }
        
        # Add extra params
        valid_params: dict = self._get_sampling_params(benchmark_config)
        
        if valid_params:
            generator_options["openai"]["OpenAICompatible"].update(valid_params)
        return generator_options
    
    def _get_sampling_params(self, benchmark_config: "BenchmarkConfig") -> dict:
        """Extract and validate sampling parameters from benchmark config.
        
        Args:
            benchmark_config: Configuration containing sampling parameters
            
        Returns:
            Dictionary of valid sampling parameters
            
        Raises:
            GarakValidationError: If benchmark_config or sampling_params are invalid
        """
        from llama_stack.apis.inference import SamplingParams, SamplingStrategy, TopPSamplingStrategy, TopKSamplingStrategy

        if not benchmark_config:
            raise GarakValidationError("benchmark_config cannot be None")
        if not hasattr(benchmark_config, 'eval_candidate'):
            raise GarakValidationError("benchmark_config must have eval_candidate attribute")
        if not hasattr(benchmark_config.eval_candidate, 'sampling_params'):
            logger.warning("benchmark_config.eval_candidate has no sampling_params, using defaults")
            return {}

        sampling_params: SamplingParams = benchmark_config.eval_candidate.sampling_params
        if not sampling_params:
            logger.warning("sampling_params is None, using defaults")
            return {}
        
        valid_params: dict = {}

        if hasattr(sampling_params, 'strategy') and sampling_params.strategy:
            strategy: SamplingStrategy = sampling_params.strategy
            # valid_params["strategy"] = strategy.type
            if isinstance(strategy, TopPSamplingStrategy):
                if strategy.top_p is not None:
                    valid_params['top_p'] = strategy.top_p
                if strategy.temperature is not None:
                    valid_params['temperature'] = strategy.temperature
            elif isinstance(strategy, TopKSamplingStrategy):
                if strategy.top_k is not None:
                    valid_params['top_k'] = strategy.top_k

        if sampling_params.max_tokens is not None:
            valid_params['max_tokens'] = sampling_params.max_tokens
        # if sampling_params.repetition_penalty is not None:
        #     valid_params['repetition_penalty'] = sampling_params.repetition_penalty
        if sampling_params.stop is not None:
            valid_params['stop'] = sampling_params.stop
        
        return valid_params
    
    def _get_llama_stack_url(self) -> str:
        """Get the normalized Llama Stack URL with /v1 suffix.
        
        Returns:
            Llama Stack URL with /v1 suffix
            
        Raises:
            GarakConfigError: If llama_stack_url is invalid
        """
        import re

        if not hasattr(self._config, 'llama_stack_url') or not self._config.llama_stack_url:
            raise GarakConfigError("llama_stack_url is not configured")
        
        llama_stack_url: str = self._config.llama_stack_url.strip()
        
        if not llama_stack_url:
            raise GarakConfigError("llama_stack_url cannot be empty")
        
        if not llama_stack_url.startswith(("http://", "https://")):
            raise GarakConfigError(
                f"llama_stack_url must start with http:// or https://, got: {llama_stack_url}"
            )
        
        llama_stack_url = llama_stack_url.rstrip("/")
        
        # check if URL ends with /v{number} and if not, add v1
        if not re.match(r"^.*\/v\d+$", llama_stack_url):
            llama_stack_url = f"{llama_stack_url}/v1"
        return llama_stack_url

    async def _get_function_based_generator_options(
        self, benchmark_config: "BenchmarkConfig", benchmark_metadata: dict
    ) -> dict:
        """Get the generator options for the custom function-based generator.
        
        Args:
            benchmark_config: Configuration for the evaluation task
            benchmark_metadata: Metadata from the benchmark
            
        Returns:
            Generator options for function-based model with shield support
            
        Raises:
            GarakValidationError: If benchmark_config is invalid
            GarakConfigError: If llama_stack_url is not configured
        """
        from llama_stack_provider_trustyai_garak import shield_scan
        
        if not benchmark_config:
            raise GarakValidationError("benchmark_config cannot be None")
        if not hasattr(benchmark_config, 'eval_candidate') or not benchmark_config.eval_candidate:
            raise GarakValidationError("benchmark_config must have eval_candidate")
        if not hasattr(benchmark_config.eval_candidate, 'model') or not benchmark_config.eval_candidate.model:
            raise GarakValidationError("benchmark_config.eval_candidate must have model")
        
        if not hasattr(self._config, 'llama_stack_url') or not self._config.llama_stack_url:
            raise GarakConfigError("llama_stack_url is not configured")
        
        llama_stack_url: str = self._config.llama_stack_url.strip().rstrip("/")
        
        if not llama_stack_url:
            raise GarakConfigError("llama_stack_url cannot be empty")
        
        # Map the shields to the input and output of LLM
        llm_io_shield_mapping: dict = {
            "input": [],
            "output": []
        }

        if not benchmark_metadata:
            logger.warning("No benchmark metadata found.")
        else:
            # get the shield_ids and/or shield_config from the benchmark metadata
            shield_ids: List[str] = benchmark_metadata.get("shield_ids", [])
            shield_config: dict = benchmark_metadata.get("shield_config", {})
            if not shield_ids and not shield_config:
                logger.warning("No shield_ids or shield_config found in the benchmark metadata")
            elif shield_ids:
                if not isinstance(shield_ids, list):
                    raise GarakValidationError("shield_ids must be a list")
                if shield_config:
                    logger.warning(
                        "Both shield_ids and shield_config found in the benchmark metadata. "
                        "Using shield_ids only."
                    )
                llm_io_shield_mapping["input"] = shield_ids
            elif shield_config:
                if not isinstance(shield_config, dict):
                    raise GarakValidationError("shield_config must be a dictionary")
                if not shield_config.get("input") and not shield_config.get("output"):
                    logger.warning("No input or output found in the shield_config.")
                llm_io_shield_mapping["input"] = shield_config.get("input", [])
                llm_io_shield_mapping["output"] = shield_config.get("output", [])
            
            await self._check_shield_availability(llm_io_shield_mapping)

        generator_options = {
            "function": {
                "Single": {
                    "name": f"{shield_scan.__name__}#simple_shield_orchestrator",
                    "kwargs": {
                        "model": benchmark_config.eval_candidate.model,
                        "base_url": llama_stack_url,
                        "llm_io_shield_mapping": llm_io_shield_mapping,
                        "max_workers": self._config.max_workers,
                        # "skip_llm": benchmark_metadata.get("skip_llm", False)
                    }
                }
            }
        }
        valid_params: dict = self._get_sampling_params(benchmark_config)
        if valid_params:
            generator_options["function"]["Single"]["kwargs"]["sampling_params"] = valid_params
        return generator_options
    
    async def _check_shield_availability(self, llm_io_shield_mapping: dict) -> None:
        """Check the availability of shields and raise an error if any shield is not available.
        
        Args:
            llm_io_shield_mapping: Mapping of input/output shields
            
        Raises:
            GarakConfigError: If shields API is not available
            GarakValidationError: If any specified shield is not available
        """
        if not self.shields_api:
            raise GarakConfigError("Shields API is not available. Please enable shields API.")
        
        error_msg: str = "{type} shield '{shield_id}' is not available. Please provide a valid shield_id in the benchmark metadata."
        
        for shield_id in llm_io_shield_mapping["input"]:
            if not await self.shields_api.get_shield(shield_id):
                raise GarakValidationError(error_msg.format(type="Input", shield_id=shield_id))
            
        for shield_id in llm_io_shield_mapping["output"]:
            if not await self.shields_api.get_shield(shield_id):
                raise GarakValidationError(error_msg.format(type="Output", shield_id=shield_id))

    async def _build_command(
        self,
        benchmark_config: "BenchmarkConfig",
        benchmark_id: str,
        scan_profile_config: dict,
        scan_report_prefix: Optional[str] = None
    ) -> List[str]:
        """Build the garak command to run the scan.
        
        Args:
            benchmark_config: Configuration for the evaluation task
            benchmark_id: The benchmark identifier
            scan_profile_config: Configuration for the scan profile (probes, timeout, etc.)
            scan_report_prefix: Optional prefix for the scan report (used in inline mode)
            
        Returns:
            List of command arguments for garak
            
        Raises:
            BenchmarkNotFoundError: If benchmark is not found
            GarakValidationError: If configuration is invalid
        """
        import json
        from llama_stack_provider_trustyai_garak import shield_scan
        
        stored_benchmark = await self.get_benchmark(benchmark_id)
        if not stored_benchmark:
            raise BenchmarkNotFoundError(f"Benchmark {benchmark_id} not found")

        benchmark_metadata: dict = getattr(stored_benchmark, "metadata", {})

        generator_options: dict = await self._get_generator_options(benchmark_config, benchmark_metadata)
        if not benchmark_config.eval_candidate.type == "model":
            raise GarakValidationError("Eval candidate type must be 'model'")

        if bool(benchmark_metadata.get("shield_ids", []) or benchmark_metadata.get("shield_config", {})):
            model_type: str = self._config.garak_model_type_function
            model_name: str = f"{shield_scan.__name__}#simple_shield_orchestrator" 
        else:
            model_type: str = self._config.garak_model_type_openai
            model_name: str = benchmark_config.eval_candidate.model
            
        cmd: List[str] = [
            "garak",
            "--model_type", model_type,
            "--model_name", model_name,
            "--generator_options", json.dumps(generator_options),
        ]
        
        # Add report prefix if provided (inline mode)
        if scan_report_prefix:
            cmd.extend(["--report_prefix", scan_report_prefix.strip()])
        
        cmd.extend(["--parallel_attempts", str(benchmark_metadata.get("parallel_attempts", self.scan_config.parallel_probes))])
        cmd.extend(["--generations", str(benchmark_metadata.get("generations", 1))])

        if "seed" in benchmark_metadata:
            cmd.extend(["--seed", str(benchmark_metadata["seed"])])

        if "deprefix" in benchmark_metadata:
            cmd.extend(["--deprefix", benchmark_metadata["deprefix"]])

        if "eval_threshold" in benchmark_metadata:
            cmd.extend(["--eval_threshold", str(benchmark_metadata["eval_threshold"])])
        
        if "probe_tags" in benchmark_metadata:
            cmd.extend(["--probe_tags", self._normalize_list_arg(benchmark_metadata["probe_tags"])])
        
        if "probe_options" in benchmark_metadata:
            cmd.extend(["--probe_options", json.dumps(benchmark_metadata["probe_options"])])
        
        if "detectors" in benchmark_metadata:
            cmd.extend(["--detectors", self._normalize_list_arg(benchmark_metadata["detectors"])])
        
        if "extended_detectors" in benchmark_metadata:
            cmd.extend(["--extended_detectors", self._normalize_list_arg(benchmark_metadata["extended_detectors"])])
        
        if "detector_options" in benchmark_metadata:
            cmd.extend(["--detector_options", json.dumps(benchmark_metadata["detector_options"])])
        
        if "buffs" in benchmark_metadata:
            cmd.extend(["--buffs", self._normalize_list_arg(benchmark_metadata["buffs"])])
        
        if "buff_options" in benchmark_metadata:
            cmd.extend(["--buff_options", json.dumps(benchmark_metadata["buff_options"])])
        
        if "harness_options" in benchmark_metadata:
            cmd.extend(["--harness_options", json.dumps(benchmark_metadata["harness_options"])])
        
        if "taxonomy" in benchmark_metadata:
            cmd.extend(["--taxonomy", benchmark_metadata["taxonomy"]])
        
        if "generate_autodan" in benchmark_metadata:
            cmd.extend(["--generate_autodan", benchmark_metadata["generate_autodan"]])
    
        # Add probes
        probes = scan_profile_config["probes"]
        if isinstance(probes, str):
            if "," in probes:
                probes = probes.split(",")
            else:
                probes = [probes]
        
        if probes != ["all"]:
            for probe in probes:
                if probe not in self.all_probes:
                    raise GarakValidationError(
                        f"Probe '{probe}' not found in garak. "
                        "Please provide valid garak probe name. "
                        "Or you can just use predefined scan profiles ('quick', 'standard') as benchmark_id."
                    )
            cmd.extend(["--probes", ",".join(probes)])
        return cmd

    async def _validate_run_eval_request(
        self, benchmark_id: str, benchmark_config: "BenchmarkConfig"
    ) -> tuple["Benchmark", dict]:
        """Validate run_eval request and return benchmark and metadata.
        
        Args:
            benchmark_id: The benchmark identifier
            benchmark_config: Configuration for the evaluation task
            
        Returns:
            Tuple of (benchmark, benchmark_metadata dict)
            
        Raises:
            GarakValidationError: If validation fails
            BenchmarkNotFoundError: If benchmark not found
        """
        
        if not isinstance(benchmark_config, BenchmarkConfig):
            raise GarakValidationError("Required benchmark_config to be of type BenchmarkConfig")
        
        # Validate that the benchmark exists
        benchmark = await self.get_benchmark(benchmark_id)
        if not benchmark:
            available_benchmarks = list(self.benchmarks.keys())
            raise BenchmarkNotFoundError(
                f"Benchmark '{benchmark_id}' not found. "
                f"Available benchmarks: {', '.join(available_benchmarks[:10])}"
                f"{'...' if len(available_benchmarks) > 10 else ''}"
            )
        
        benchmark_metadata: dict = getattr(benchmark, "metadata", {})
        
        if not benchmark_metadata.get("probes"):
            raise GarakValidationError(
                "No probes found for benchmark. Please specify probes list in the benchmark metadata."
            )
        
        return benchmark, benchmark_metadata

    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        """Get the result of a job (common implementation).
        
        Args:
            benchmark_id: The benchmark id
            job_id: The job id
            
        Returns:
            EvaluateResponse with results or empty response
            
        Raises:
            BenchmarkNotFoundError: If benchmark not found
        """
        import json
        
        stored_benchmark = await self.get_benchmark(benchmark_id)
        if not stored_benchmark:
            raise BenchmarkNotFoundError(f"Benchmark {benchmark_id} not found")

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
            if self._job_metadata[job_id].get("scan_result.json"):
                scan_result_file_id: str = self._job_metadata[job_id].get("scan_result.json", "")
                scan_result = await self.file_api.openai_retrieve_file_content(scan_result_file_id)
                return EvaluateResponse(**json.loads(scan_result.body.decode("utf-8")))
            else:
                logger.error(f"No scan_result.json file found for job {job_id}")
                return EvaluateResponse(generations=[], scores={})
        
        else:
            logger.warning(f"Job {job_id} has an unknown status: {job.status}")
            return EvaluateResponse(generations=[], scores={})

    async def evaluate_rows(
        self,
        benchmark_id: str,
        input_rows: list[dict[str, Any]],
        scoring_functions: list[str],
        benchmark_config: BenchmarkConfig,
    ) -> EvaluateResponse:
        """Evaluate rows (not implemented for Garak).
        
        Args:
            benchmark_id: The benchmark id
            input_rows: Input rows to evaluate
            scoring_functions: Scoring functions to use
            benchmark_config: Configuration for evaluation
            
        Raises:
            NotImplementedError: This method is not implemented for Garak
        """
        raise NotImplementedError("evaluate_rows is not implemented for Garak provider")

    def _convert_datetime_to_str(self, datetime_obj) -> str:
        """Convert a datetime object to ISO format string.
        
        Args:
            datetime_obj: Datetime object to convert
            
        Returns:
            ISO format string representation
        """
        from datetime import datetime
        return datetime_obj.isoformat() if isinstance(datetime_obj, datetime) else str(datetime_obj)

