"""Garak Framework Adapter for eval-hub.

This adapter integrates NVIDIA's Garak red-teaming framework with the
eval-hub evaluation platform. It supports two execution modes:

- **simple** (default): Runs garak as a subprocess inside the K8s Job pod.
- **kfp**: Delegates scan execution to a Kubeflow Pipeline, using S3
  (via a Data Connection secret) for artifact transfer. The adapter
  submits the pipeline, polls for completion, and downloads results
  from the shared S3 bucket.

The adapter:
1. Reads JobSpec from mounted ConfigMap (/etc/eval-job/spec.json)
2. Builds and executes Garak CLI commands (simple) or submits KFP pipeline (kfp)
3. Parses results from Garak's JSONL reports
4. Persists artifacts to OCI registry
5. Reports results back to eval-hub via sidecar callbacks
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .kfp_pipeline import KFPConfig

from evalhub.adapter import (
    DefaultCallbacks,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    MessageInfo,
    OCIArtifactSpec,
)
from evalhub.models.api import EvaluationResult

from ..core.command_builder import build_generator_options
from ..core.config_resolution import (
    build_effective_garak_config,
    resolve_scan_profile,
    resolve_timeout_seconds,
)
from ..core.garak_runner import convert_to_avid_report, GarakScanResult, run_garak_scan
from ..garak_command_config import GarakCommandConfig
from ..result_utils import (
    combine_parsed_results,
    generate_art_report,
    parse_aggregated_from_avid_content,
    parse_digest_from_report_content,
    parse_generations_from_report_content,
    parse_harness_stub_summaries,
    parse_harness_summary,
)
from ..utils import get_scan_base_dir, as_bool, safe_int
from ..constants import (
    DEFAULT_TIMEOUT,
    DEFAULT_MODEL_TYPE,
    DEFAULT_EVAL_THRESHOLD,
    EXECUTION_MODE_SIMPLE,
    EXECUTION_MODE_KFP,
    DEFAULT_SDG_FLOW_ID,
    DEFAULT_SDG_MAX_CONCURRENCY,
    DEFAULT_SDG_NUM_SAMPLES,
    DEFAULT_SDG_MAX_TOKENS,
    TARGET_DEFAULT_PARAMETERS,
)

logger = logging.getLogger(__name__)


class GarakAdapter(FrameworkAdapter):
    """Garak red-teaming framework adapter for eval-hub.

    Supports two execution modes:
    - **simple**: Runs garak as a local subprocess (default).
    - **kfp**: Submits a Kubeflow Pipeline, polls for completion, and
      downloads results from S3 via a Data Connection secret.

    Benchmark Configuration:
        The adapter expects benchmark_config in JobSpec to contain:
        - probes: List of Garak probe names
        - probe_tags: Optional probe tag filters
        - eval_threshold: Threshold for vulnerability detection (default: 0.5)
        - execution_mode: "simple" or "kfp" (default: "simple")
        - kfp_config: dict with KFP connection details (for kfp mode)
        - Other Garak CLI options (detectors, buffs, etc.)

    Results:
        Returns one EvaluationResult per probe with:
        - attack_success_rate: Percentage of successful attacks
        - vulnerable_responses: Count of vulnerable responses
        - total_attempts: Total number of probe attempts
    """

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """Run a Garak security scan job.

        Dispatches to either local subprocess execution or KFP pipeline
        execution depending on the resolved execution_mode.
        """
        start_time = time.time()
        execution_dir = None
        retain_artifacts = os.getenv("GARAK_RETAIN_SCAN_ARTIFACTS", "false").strip().lower() in {"true", "1", "yes"}
        logger.info(f"Starting Garak job {config.id} for benchmark {config.benchmark_id}")

        try:
            # Phase 1: Initialize
            callbacks.report_status(JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.INITIALIZING))
            logger.info("Validating configuration and building scan command")

            self._validate_config(config)

            benchmark_config = config.parameters or {}
            execution_mode = self._resolve_execution_mode(benchmark_config)
            logger.info("Execution mode: %s", execution_mode)

            # Build merged garak config (common to both modes)
            # Isolate retries and overlapping deliveries, including disabled exports.
            scan_base = get_scan_base_dir() / config.id
            scan_base.mkdir(parents=True, exist_ok=True)
            scan_dir = Path(mkdtemp(prefix="execution-", dir=scan_base))
            execution_dir = scan_dir
            report_prefix = scan_dir / "scan"

            garak_config_dict, profile, intents_params = self._build_config_from_spec(config, report_prefix)
            if not garak_config_dict:
                raise ValueError("Garak command config is empty")

            art_intents = intents_params.get("art_intents", False)

            timeout = resolve_timeout_seconds(
                benchmark_config,
                profile,
                default_timeout=DEFAULT_TIMEOUT,
            )
            logger.info("Using timeout=%ss for benchmark=%s", timeout, config.benchmark_id)

            eval_threshold = float(garak_config_dict.get("run", {}).get("eval_threshold", DEFAULT_EVAL_THRESHOLD))

            # Phase 2–3: Loading data + Execute scan (mode-dependent)
            if execution_mode == EXECUTION_MODE_KFP:
                result, scan_dir = self._run_via_kfp(
                    config=config,
                    callbacks=callbacks,
                    garak_config_dict=garak_config_dict,
                    timeout=timeout,
                    intents_params=intents_params,
                    eval_threshold=eval_threshold,
                    scan_dir=scan_dir,
                )
            elif art_intents:
                result = self._run_simple_intents(
                    config=config,
                    callbacks=callbacks,
                    garak_config_dict=garak_config_dict,
                    scan_dir=scan_dir,
                    timeout=timeout,
                    intents_params=intents_params,
                )
            else:
                result = self._run_simple(
                    config=config,
                    callbacks=callbacks,
                    garak_config_dict=garak_config_dict,
                    scan_dir=scan_dir,
                    timeout=timeout,
                )

            if not result.success:
                error_msg = f"Garak scan failed: {result.stderr}" if result.stderr else "Unknown error"
                if result.timed_out:
                    error_msg = f"Scan timed out after {timeout} seconds"
                if result.log_errors:
                    error_msg += "\nscan.log errors:\n" + "\n".join(result.log_errors)

                callbacks.report_status(
                    JobStatusUpdate(
                        status=JobStatus.FAILED,
                        error_message=MessageInfo(
                            message=error_msg,
                            message_code="scan_failed",
                        ),
                    )
                )
                raise RuntimeError(error_msg)

            # Phase 4: Parse results (common to both modes)
            callbacks.report_status(JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.POST_PROCESSING))
            logger.info("Parsing scan results")

            metrics, overall_score, num_examples, overall_summary = self._parse_results(
                result,
                eval_threshold,
                art_intents=art_intents,
            )
            logger.info(f"Parsed {len(metrics)} probe metrics, overall score: {overall_score}")

            sample_file = None
            if benchmark_config.get("save_sample_results", True):
                from .sample_results import save_sample_results

                # Sample persistence is auxiliary: a valid aggregate must survive
                # report edge cases or local export failures.
                try:
                    sample_file, sample_count = save_sample_results(
                        result.report_jsonl, scan_dir / "sample_results.jsonl", eval_threshold, art_intents=art_intents
                    )
                    logger.info("Saved %d per-sample results to %s", sample_count, sample_file)
                except Exception as exc:
                    sample_file = None
                    logger.warning("Skipping sample results export: %s", exc)

            # Phase 4b: Generate ART HTML report for intents scans
            if art_intents and result.report_jsonl.exists():
                try:
                    report_content = result.report_jsonl.read_text()
                    if report_content.strip():
                        art_html_path = scan_dir / "scan.intents.html"
                        if not art_html_path.exists():
                            art_html = generate_art_report(report_content, eval_threshold=eval_threshold)
                            art_html_path.write_text(art_html)
                            logger.info("Generated ART HTML report: %s", art_html_path)
                except Exception as e:
                    logger.warning("Failed to generate ART HTML report: %s", e)

            # Redact api_key values from config.json before OCI export
            config_file = scan_dir / "config.json"
            if config_file.exists():
                try:
                    from ..core.pipeline_steps import redact_api_keys

                    cfg = json.loads(config_file.read_text())
                    config_file.write_text(json.dumps(redact_api_keys(cfg), indent=1))
                except Exception as exc:
                    logger.warning("Could not redact config.json: %s", exc)

            # Phase 5: Persist artifacts
            oci_artifact = None
            oci_exports = config.exports.oci if config.exports else None
            if oci_exports is not None:
                callbacks.report_status(JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.PERSISTING_ARTIFACTS))
                oci_artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(
                        files_path=scan_dir,
                        coordinates=oci_exports.coordinates,
                    )
                )
                logger.info(f"Persisted scan artifacts: {oci_artifact.reference}")

            # Compute duration
            duration = time.time() - start_time

            eval_meta: dict[str, Any] = {
                "framework": "garak",
                "eval_threshold": eval_threshold,
                "timed_out": result.timed_out,
                "execution_mode": execution_mode,
                "art_intents": art_intents,
                "overall": overall_summary,
            }

            if execution_mode == EXECUTION_MODE_KFP:
                from .kfp_pipeline import KFPConfig

                # Resolve every artifact URI with the same config/Secret/env
                # precedence used for downloads, even when samples are disabled.
                resolved = None
                s3_prefix = ""
                try:
                    kfp_config = KFPConfig.from_env_and_config(benchmark_config)
                    creds = (
                        self._read_s3_credentials_from_secret(kfp_config.s3_secret_name, kfp_config.namespace)
                        if kfp_config.s3_secret_name
                        else {}
                    )
                    resolved = self._resolve_s3_credentials(kfp_config, creds)
                    s3_prefix = f"{kfp_config.s3_prefix}/{config.id}/{scan_dir.name}"
                except Exception as exc:
                    logger.warning("Unable to resolve KFP artifact locations: %s", exc)
                _bucket = resolved["bucket"] if resolved else ""

                artifact_keys: dict[str, str] = {
                    "scan_report": f"{s3_prefix}/scan.report.jsonl",
                    "scan_html_report": f"{s3_prefix}/scan.report.html",
                    "scan_hitlog": f"{s3_prefix}/scan.hitlog.jsonl",
                }
                if sample_file is not None and resolved and _bucket:
                    # Publish the URI only after upload succeeds; a failed optional
                    # upload must not discard the completed evaluation results.
                    try:
                        s3 = self._create_s3_client(
                            endpoint_url=resolved["endpoint_url"],
                            access_key=resolved["access_key"],
                            secret_key=resolved["secret_key"],
                            region=resolved["region"],
                        )
                        sample_key = f"{s3_prefix}/sample_results.jsonl"
                        s3.upload_file(str(sample_file), _bucket, sample_key)
                        artifact_keys["sample_results"] = sample_key
                    except Exception as exc:
                        logger.warning("Skipping KFP sample results upload: %s", exc)
                if art_intents:
                    artifact_keys["sdg_raw_output"] = f"{s3_prefix}/sdg_raw_output.csv"
                    artifact_keys["sdg_normalized_output"] = f"{s3_prefix}/sdg_normalized_output.csv"
                    artifact_keys["intents_html_report"] = f"{s3_prefix}/scan.intents.html"

                s3_artifacts: dict[str, str] = {}
                for name, key in artifact_keys.items():
                    local_file = scan_dir / key.split("/")[-1]
                    if _bucket and local_file.exists():
                        s3_artifacts[name] = f"s3://{_bucket}/{key}"
                    else:
                        logger.debug("Artifact not downloaded locally, skipping: %s", key)

                eval_meta["artifacts"] = s3_artifacts

            results = JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name,
                results=metrics,
                overall_score=overall_score,
                num_examples_evaluated=num_examples,
                duration_seconds=duration,
                completed_at=datetime.now(UTC),
                evaluation_metadata=eval_meta,
                oci_artifact=oci_artifact,
            )

            # Phase 6: Save to MLflow (if experiment_name configured)
            try:
                from evalhub.adapter.mlflow import MlflowArtifact

                mlflow_artifacts: list[MlflowArtifact] = []
                # OCI includes the complete scan directory; explicitly add the
                # same sample file to MLflow's individually enumerated artifacts.
                if sample_file is not None:
                    mlflow_artifacts.append(
                        MlflowArtifact("sample_results.jsonl", sample_file.read_bytes(), "application/jsonl")
                    )
                if result.report_html.exists():
                    mlflow_artifacts.append(
                        MlflowArtifact(
                            "scan.report.html",
                            result.report_html.read_bytes(),
                            "text/html",
                        )
                    )
                art_html_path = scan_dir / "scan.intents.html"
                if art_html_path.exists():
                    mlflow_artifacts.append(
                        MlflowArtifact(
                            "scan.intents.html",
                            art_html_path.read_bytes(),
                            "text/html",
                        )
                    )
                if result.report_jsonl.exists():
                    mlflow_artifacts.append(
                        MlflowArtifact(
                            "scan.report.jsonl",
                            result.report_jsonl.read_bytes(),
                            "application/jsonl",
                        )
                    )
                if result.hitlog_jsonl.exists():
                    mlflow_artifacts.append(
                        MlflowArtifact(
                            "scan.hitlog.jsonl",
                            result.hitlog_jsonl.read_bytes(),
                            "application/jsonl",
                        )
                    )
                sdg_raw = scan_dir / "sdg_raw_output.csv"
                if sdg_raw.exists():
                    mlflow_artifacts.append(
                        MlflowArtifact(
                            "sdg_raw_output.csv",
                            sdg_raw.read_bytes(),
                            "text/csv",
                        )
                    )
                sdg_norm = scan_dir / "sdg_normalized_output.csv"
                if sdg_norm.exists():
                    mlflow_artifacts.append(
                        MlflowArtifact(
                            "sdg_normalized_output.csv",
                            sdg_norm.read_bytes(),
                            "text/csv",
                        )
                    )

                rid = callbacks.mlflow.save(results, config, artifacts=mlflow_artifacts)
                if rid:
                    results.mlflow_run_id = rid
                logger.info("Saved results and %d artifacts to MLflow with run ID: %s", len(mlflow_artifacts), rid)
            except Exception as mlflow_exc:
                logger.warning("MLflow save failed (non-fatal): %s", mlflow_exc)

            return results

        except Exception as e:
            logger.exception(f"Garak job {config.id} failed")
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    error_message=MessageInfo(
                        message=str(e),
                        message_code="job_failed",
                    ),
                )
            )
            raise
        finally:
            # Remote persistence and MLflow callbacks finish before scratch files
            # are removed. Never remove a caller's directory or another execution.
            if execution_dir is not None and not retain_artifacts:
                try:
                    shutil.rmtree(execution_dir)
                except OSError as exc:
                    logger.warning("Unable to clean scan directory %s: %s", execution_dir, exc)

    # ------------------------------------------------------------------
    # Execution mode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_execution_mode(benchmark_config: dict) -> str:
        """Resolve execution mode from benchmark_config or env var.

        Priority: benchmark_config > env var > default ("simple").
        """
        mode = benchmark_config.get("execution_mode") or os.getenv("EVALHUB_EXECUTION_MODE", EXECUTION_MODE_SIMPLE)
        mode = str(mode).strip().lower()
        if mode not in (EXECUTION_MODE_SIMPLE, EXECUTION_MODE_KFP):
            logger.warning("Unknown execution_mode '%s', falling back to simple", mode)
            mode = EXECUTION_MODE_SIMPLE
        return mode

    # ------------------------------------------------------------------
    # Simple (subprocess) execution
    # ------------------------------------------------------------------

    def _run_simple(
        self,
        config: JobSpec,
        callbacks: JobCallbacks,
        garak_config_dict: dict,
        scan_dir: Path,
        timeout: int,
    ) -> GarakScanResult:
        """Run garak as a local subprocess."""
        log_file = scan_dir / "scan.log"
        report_prefix = scan_dir / "scan"

        callbacks.report_status(JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.LOADING_DATA))

        config_file = scan_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(garak_config_dict, f, indent=1)

        env: dict[str, str] = {}
        hf_cache = (config.parameters or {}).get("hf_cache_path", "")
        if hf_cache:
            env["HF_HUB_CACHE"] = hf_cache
            logger.info("Using HF cache from mounted path: %s", hf_cache)

        callbacks.report_status(JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.RUNNING_EVALUATION))
        logger.info("Running Garak scan for %s", config.benchmark_id)

        result = run_garak_scan(
            config_file=config_file,
            timeout_seconds=timeout,
            log_file=log_file,
            report_prefix=report_prefix,
            env=env if env else None,
        )

        # AVID conversion
        if result.success:
            logger.info("Converting results to AVID format")
            convert_to_avid_report(result.report_jsonl)

        return result

    # ------------------------------------------------------------------
    # Simple intents (subprocess with SDG preprocessing)
    # ------------------------------------------------------------------

    def _run_simple_intents(
        self,
        config: JobSpec,
        callbacks: JobCallbacks,
        garak_config_dict: dict,
        scan_dir: Path,
        timeout: int,
        intents_params: dict[str, Any],
    ) -> GarakScanResult:
        """Run an intents benchmark as a local subprocess with in-process SDG.

        Executes the intents pipeline steps sequentially:
        1. Resolve taxonomy (local file or default BASE_TAXONOMY)
        2. Run SDG generation (or bypass if intents_s3_key provided)
        3. Normalize prompts
        4. Delegate to setup_and_run_garak() for stub generation + scan

        External files (taxonomy, bypass prompts) are expected as local
        paths pre-downloaded by the EvalHub SDK to ``/test_data``.  The
        existing ``policy_s3_key`` / ``intents_s3_key`` parameters are
        reused as local file paths in simple mode.
        """
        from ..core.pipeline_steps import (
            normalize_prompts,
            resolve_taxonomy_data,
            run_sdg_generation,
            setup_and_run_garak,
        )

        callbacks.report_status(JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.LOADING_DATA))
        logger.info("Resolving taxonomy for intents scan")

        policy_path = intents_params.get("policy_s3_key", "")
        policy_format = intents_params.get("policy_format", "csv")
        intents_path = intents_params.get("intents_s3_key", "")
        intents_format = intents_params.get("intents_format", "csv")

        # -- Step 1: Resolve taxonomy --------------------------------

        policy_content: bytes | None = None
        if policy_path and policy_path.strip():
            policy_file = Path(policy_path)
            if not policy_file.exists():
                raise FileNotFoundError(
                    f"Taxonomy file not found at '{policy_path}'. Ensure the file is available via test_data_ref."
                )
            policy_content = policy_file.read_bytes()
            logger.info("Read taxonomy from local path: %s (%d bytes)", policy_path, len(policy_content))

        taxonomy_df = resolve_taxonomy_data(policy_content, format=policy_format)
        logger.info("Resolved taxonomy: %d entries", len(taxonomy_df))

        # -- Step 2: Generate prompts (SDG or bypass) ----------------
        raw_content: str | None = None

        if intents_path and intents_path.strip():
            logger.info("Loading pre-generated prompts (bypass SDG)")
            intents_file = Path(intents_path)
            if not intents_file.exists():
                raise FileNotFoundError(
                    f"Intents file not found at '{intents_path}'. Ensure the file is available via test_data_ref."
                )
            raw_content = intents_file.read_text(encoding="utf-8")
            logger.info("Read bypass prompts from local path: %s (%d bytes)", intents_path, len(raw_content))
        else:
            logger.info("Running Synthetic Data Generation")
            sdg_model = intents_params.get("sdg_model", "")
            sdg_api_base = intents_params.get("sdg_api_base", "")

            sdg_secret_url, sdg_secret_key = self._read_role_from_secret("sdg", config.model.url)
            effective_sdg_base = sdg_secret_url or sdg_api_base
            sdg_source = "model auth secret" if sdg_secret_url else "intents_params (job request)"
            logger.info("Intents sdg model: api_base=%s (source: %s)", effective_sdg_base, sdg_source)
            if not sdg_model or not effective_sdg_base:
                raise ValueError(
                    "Intents benchmark requires sdg_model and sdg_api_base "
                    "for prompt generation when intents_s3_key is not provided."
                )

            raw_df = run_sdg_generation(
                taxonomy_df=taxonomy_df,
                sdg_model=sdg_model,
                sdg_api_base=effective_sdg_base,
                sdg_provider=intents_params.get("sdg_provider", ""),
                sdg_flow_id=intents_params.get("sdg_flow_id", DEFAULT_SDG_FLOW_ID),
                sdg_max_concurrency=intents_params.get("sdg_max_concurrency", DEFAULT_SDG_MAX_CONCURRENCY),
                sdg_num_samples=intents_params.get("sdg_num_samples", DEFAULT_SDG_NUM_SAMPLES),
                sdg_max_tokens=intents_params.get("sdg_max_tokens", DEFAULT_SDG_MAX_TOKENS),
                api_key=sdg_secret_key,
            )
            raw_content = raw_df.to_csv(index=False)
            logger.info("SDG produced %d rows", len(raw_df))

        # -- Persist raw SDG output ----------------------------------
        raw_output_path = scan_dir / "sdg_raw_output.csv"
        raw_output_path.write_text(raw_content, encoding="utf-8")
        logger.info("Saved raw SDG output to %s", raw_output_path)

        # -- Step 3: Normalize prompts -------------------------------
        logger.info("Normalizing prompts")
        if not (intents_path and intents_path.strip()) and intents_format.strip().lower() != "csv":
            logger.warning(
                "intents_format=%r ignored — SDG output is always CSV. "
                "intents_format only applies when intents_s3_key provides a bypass file.",
                intents_format,
            )
        fmt = intents_format if (intents_path and intents_path.strip()) else "csv"
        normalized_df = normalize_prompts(raw_content, format=fmt)

        norm_output_path = scan_dir / "sdg_normalized_output.csv"
        normalized_df.to_csv(norm_output_path, index=False)
        logger.info("Saved normalized prompts to %s (%d rows)", norm_output_path, len(normalized_df))

        prompts_csv_path = scan_dir / "prompts.csv"
        normalized_df.to_csv(prompts_csv_path, index=False)

        # -- Step 4: Run garak scan ----------------------------------
        callbacks.report_status(JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.RUNNING_EVALUATION))
        logger.info("Running Garak intents scan for %s", config.benchmark_id)
        config_json = json.dumps(garak_config_dict)
        result = setup_and_run_garak(
            config_json=config_json,
            prompts_csv_path=prompts_csv_path,
            scan_dir=scan_dir,
            timeout_seconds=timeout,
        )

        return result

    # ------------------------------------------------------------------
    # KFP execution
    # ------------------------------------------------------------------

    def _run_via_kfp(
        self,
        config: JobSpec,
        callbacks: JobCallbacks,
        garak_config_dict: dict,
        timeout: int,
        intents_params: dict[str, Any] | None = None,
        eval_threshold: float = DEFAULT_EVAL_THRESHOLD,
        scan_dir: Path | None = None,
    ) -> tuple[GarakScanResult, Path]:
        """Submit a KFP pipeline, poll until done, download results from S3.

        The KFP component uploads scan output to S3 under
        ``{kfp_config.s3_prefix}/{job_id}/{execution_id}/``. After the run completes,
        this method downloads those files to a local scan_dir for parsing.

        Both sides get S3 credentials from the same Data Connection secret:
        - KFP pod: injected via ``kubernetes.use_secret_as_env``
        - Adapter pod: mounted via ``envFrom`` by the eval-hub service

        Returns:
            Tuple of (GarakScanResult, local scan_dir with downloaded results).
        """
        from .kfp_pipeline import KFPConfig, evalhub_garak_pipeline

        callbacks.report_status(JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.LOADING_DATA))

        benchmark_config = config.parameters or {}
        kfp_config = KFPConfig.from_env_and_config(benchmark_config)

        if not kfp_config.s3_secret_name:
            raise ValueError(
                "S3 data-connection secret name is required for KFP mode. "
                "Set KFP_S3_SECRET_NAME or provide "
                "kfp_config.s3_secret_name in benchmark_config."
            )

        # Public job execution supplies its owned scratch directory so cleanup
        # covers KFP failures too; direct helper callers may supply their own.
        if scan_dir is None:
            scan_base = get_scan_base_dir() / config.id
            scan_base.mkdir(parents=True, exist_ok=True)
            scan_dir = Path(mkdtemp(prefix="execution-", dir=scan_base))
        s3_prefix = f"{kfp_config.s3_prefix}/{config.id}/{scan_dir.name}"

        from ..core.pipeline_steps import redact_api_keys

        sanitised_config = redact_api_keys(garak_config_dict)
        config_json = json.dumps(sanitised_config)

        ip = intents_params or {}

        if ip.get("art_intents"):
            if ip.get("policy_s3_key") and ip.get("intents_s3_key"):
                raise ValueError(
                    "policy_s3_key and intents_s3_key are mutually exclusive. "
                    "Provide a taxonomy for SDG (policy_s3_key) OR "
                    "pre-generated prompts to bypass SDG (intents_s3_key), not both."
                )
            if not ip.get("intents_s3_key"):
                if not ip.get("sdg_model"):
                    raise ValueError(
                        "Intents benchmark (art_intents=True) requires "
                        "sdg_model for prompt generation when intents_s3_key "
                        "is not provided."
                    )
                # sdg_api_base may be empty here — the KFP SDG pod will
                # resolve it from the mounted secret (sdg_url) at runtime.

        logger.info("Submitting KFP pipeline for %s", config.benchmark_id)
        callbacks.report_status(JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.RUNNING_EVALUATION))

        # Resolve model auth secret from EvalHub SDK model.auth.secret_ref
        # Falls back to pipeline default ("model-auth") when not specified.
        model_auth_secret = ""
        try:
            model_auth = getattr(config.model, "auth", None)
            if model_auth:
                model_auth_secret = getattr(model_auth, "secret_ref", "") or ""
        except Exception:
            pass
        if model_auth_secret:
            logger.info("Using model auth secret: %s", model_auth_secret)

        kfp_client = self._create_kfp_client(kfp_config)

        pipeline_args: dict[str, Any] = {
            "config_json": config_json,
            "s3_prefix": s3_prefix,
            "timeout_seconds": timeout,
            "s3_secret_name": kfp_config.s3_secret_name,
            "eval_threshold": eval_threshold,
            "art_intents": ip.get("art_intents", False),
            "policy_s3_key": ip.get("policy_s3_key", ""),
            "policy_format": ip.get("policy_format", "csv"),
            "intents_s3_key": ip.get("intents_s3_key", ""),
            "intents_format": ip.get("intents_format", "csv"),
            "sdg_model": ip.get("sdg_model") or "",
            "sdg_api_base": ip.get("sdg_api_base") or "",
            "sdg_provider": ip.get("sdg_provider") or "",
            "sdg_flow_id": ip.get("sdg_flow_id", DEFAULT_SDG_FLOW_ID),
            "sdg_max_concurrency": ip.get("sdg_max_concurrency", DEFAULT_SDG_MAX_CONCURRENCY),
            "sdg_num_samples": ip.get("sdg_num_samples", DEFAULT_SDG_NUM_SAMPLES),
            "sdg_max_tokens": ip.get("sdg_max_tokens", DEFAULT_SDG_MAX_TOKENS),
            "hf_cache_path": benchmark_config.get("hf_cache_path", ""),
        }
        if model_auth_secret:
            pipeline_args["model_auth_secret_name"] = model_auth_secret

        disable_cache = as_bool(ip.get("disable_cache", False))
        run = kfp_client.create_run_from_pipeline_func(
            evalhub_garak_pipeline,
            arguments=pipeline_args,
            run_name=f"evalhub-garak-{config.id}",
            namespace=kfp_config.namespace,
            experiment_name=kfp_config.experiment_name,
            enable_caching=not disable_cache,
        )

        kfp_run_id = run.run_id
        logger.info("Submitted KFP run %s", kfp_run_id)

        poll_timeout = int(timeout * 2) if timeout > 0 else 0
        final_state = self._poll_kfp_run(
            kfp_client,
            kfp_run_id,
            callbacks,
            kfp_config.poll_interval_seconds,
            timeout=poll_timeout,
        )

        timed_out = final_state == "TIMED_OUT"
        success = final_state == "SUCCEEDED"

        if success:
            logger.info("Downloading scan results from S3")
            creds = (
                self._read_s3_credentials_from_secret(
                    kfp_config.s3_secret_name,
                    kfp_config.namespace,
                )
                if kfp_config.s3_secret_name
                else {}
            )
            if kfp_config.s3_secret_name and not creds:
                logger.warning(
                    "S3 credentials from secret '%s/%s' are empty. "
                    "Ensure the secret exists and the Job pod's service account "
                    "has RBAC permissions to read secrets in namespace '%s'. "
                    "Falling back to environment variables for S3 access. "
                    "If no environment variables are set, the job will fail "
                    "as it will not be able to interact with S3.",
                    kfp_config.namespace,
                    kfp_config.s3_secret_name,
                    kfp_config.namespace,
                )
            resolved = self._resolve_s3_credentials(kfp_config, creds)
            self._download_results_from_s3(
                resolved["bucket"],
                s3_prefix,
                scan_dir,
                endpoint_url=resolved["endpoint_url"],
                access_key=resolved["access_key"],
                secret_key=resolved["secret_key"],
                region=resolved["region"],
            )

        report_prefix = scan_dir / "scan"

        return GarakScanResult(
            returncode=0 if success else 1,
            stdout="",
            stderr="" if success else f"KFP run ended with state: {final_state}",
            report_prefix=report_prefix,
            timed_out=timed_out,
        ), scan_dir

    @staticmethod
    def _create_kfp_client(kfp_config: "KFPConfig"):
        """Create a KFP Client from KFPConfig.

        Two auth models are supported:

        **Old model (evalhub < 0.4.4 / direct KFP)**: the full SA token is
        auto-mounted at the standard path; this method reads it and passes it
        to the client. KFP_ENDPOINT points at the real KFP API URL.

        **New model (evalhub >= 0.4.4 / sidecar proxy)**: evalhub PR #672 sets
        AutomountServiceAccountToken=false on the adapter pod. The SA token
        file no longer exists, so ``token`` stays None — which is correct
        because KFP_ENDPOINT must be set to ``localhost:8080`` and the sidecar
        proxy injects the SA token on every outbound request automatically.
        namespace is always passed explicitly so the KFP SDK never needs to
        call get_kfp_healthz to auto-detect it.
        """
        from kfp import Client

        ssl_ca_cert = kfp_config.ssl_ca_cert or None
        token = kfp_config.auth_token or None

        # Old model: SA token file is present — read it for direct KFP auth.
        # New model: file does not exist (only DownwardAPI namespace file is
        # mounted); token stays None and the sidecar injects auth instead.
        if not token:
            sa_token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
            if sa_token_path.exists():
                token = sa_token_path.read_text().strip()
                logger.debug("Using service account token for KFP auth")

        return Client(
            host=kfp_config.endpoint,
            existing_token=token,
            namespace=kfp_config.namespace,
            verify_ssl=kfp_config.verify_ssl,
            ssl_ca_cert=ssl_ca_cert,
        )

    @staticmethod
    def _poll_kfp_run(
        kfp_client,
        run_id: str,
        callbacks: JobCallbacks,
        poll_interval: int,
        timeout: int = 0,
    ) -> str:
        """Poll a KFP run until it reaches a terminal state or times out.

        Relays progress updates to the eval-hub sidecar via callbacks.

        Args:
            timeout: Maximum wall-clock seconds to wait. 0 means no limit.

        Returns:
            Final run state string (e.g. "SUCCEEDED", "FAILED", "TIMED_OUT").
        """
        terminal_states = {"SUCCEEDED", "FAILED", "SKIPPED", "CANCELED", "CANCELING"}
        deadline = (time.monotonic() + timeout) if timeout > 0 else None
        while True:
            run = kfp_client.get_run(run_id)
            state = run.state or "UNKNOWN"
            logger.info("KFP run %s state: %s", run_id, state)

            if state in terminal_states:
                return state

            if deadline and time.monotonic() >= deadline:
                logger.error(
                    "KFP run %s timed out after %ss (last state: %s)",
                    run_id,
                    timeout,
                    state,
                )
                return "TIMED_OUT"

            callbacks.report_status(JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.RUNNING_EVALUATION))
            logger.info("KFP pipeline running (state: %s)", state)
            time.sleep(poll_interval)

    @staticmethod
    def _read_s3_credentials_from_secret(secret_name: str, namespace: str) -> dict:
        """Read S3 credentials from a Kubernetes secret.

        Falls back gracefully if the secret cannot be read (e.g. outside a
        cluster or missing RBAC), returning an empty dict so env-var fallback
        in ``create_s3_client`` still applies.

        Resolution chain (first success wins):

        1. **Direct K8s API** (``load_incluster_config``) — works on
           pre-sidecar evalhub where the SA token is auto-mounted.
        2. **``KFP_AUTH_TOKEN`` + ``KUBERNETES_SERVICE_HOST``** env vars —
           fallback when SA token file is absent but env vars are set.
        3. **Sidecar proxy** (``k8s_sa_token:ref``) — evalhub >= 0.4.4
           sidecar model.  Reads the secret via the K8s API through the
           sidecar at ``localhost:8080``, which resolves the ref token and
           injects the pod SA token.  Requires ``k8s_sa_token`` and
           ``k8s_url`` keys in the model auth secret.
        4. **Model auth secret** (``read_model_auth_key``) — last resort
           if the user placed S3 credentials directly in the model auth
           secret (keys: ``s3_access_key``, ``s3_secret_key``,
           ``s3_endpoint``, ``s3_bucket``, ``s3_region``).
        5. **``load_kube_config``** — local development only.
        """
        import base64

        _S3_KEYS = {
            "access_key": "AWS_ACCESS_KEY_ID",
            "secret_key": "AWS_SECRET_ACCESS_KEY",
            "region": "AWS_DEFAULT_REGION",
            "bucket": "AWS_S3_BUCKET",
            "endpoint_url": "AWS_S3_ENDPOINT",
        }

        def _extract_from_secret_data(data: dict) -> dict:
            def _decode(key: str) -> str:
                val = data.get(key, "")
                return base64.b64decode(val).decode() if val else ""

            return {field: _decode(secret_key) for field, secret_key in _S3_KEYS.items()}

        # --- Step 1: Direct K8s API (pre-sidecar evalhub) ---
        try:
            from kubernetes import client as k8s_client, config as k8s_config

            try:
                k8s_config.load_incluster_config()
                v1 = k8s_client.CoreV1Api()
                secret = v1.read_namespaced_secret(secret_name, namespace)
                logger.info("Read S3 credentials from secret %s/%s via in-cluster config", namespace, secret_name)
                return _extract_from_secret_data(secret.data or {})
            except k8s_config.ConfigException:
                logger.debug("load_incluster_config failed, trying fallbacks")
        except Exception as exc:
            logger.debug("K8s client import or in-cluster config failed: %s", exc)

        # --- Step 2: KFP_AUTH_TOKEN + KUBERNETES_SERVICE_HOST env vars ---
        auth_token = os.getenv("KFP_AUTH_TOKEN", "")
        k8s_host = os.getenv("KUBERNETES_SERVICE_HOST", "")
        k8s_port = os.getenv("KUBERNETES_SERVICE_PORT", "443")
        if auth_token and k8s_host:
            try:
                from kubernetes import client as k8s_client

                configuration = k8s_client.Configuration()
                configuration.host = f"https://{k8s_host}:{k8s_port}"
                configuration.api_key = {"authorization": f"Bearer {auth_token}"}
                ca_cert = os.getenv("SSL_CA_CERT", "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
                if os.path.isfile(ca_cert):
                    configuration.ssl_ca_cert = ca_cert
                else:
                    configuration.verify_ssl = False
                k8s_client.Configuration.set_default(configuration)
                v1 = k8s_client.CoreV1Api()
                secret = v1.read_namespaced_secret(secret_name, namespace)
                logger.info("Read S3 credentials from secret %s/%s via KFP_AUTH_TOKEN", namespace, secret_name)
                return _extract_from_secret_data(secret.data or {})
            except Exception as exc:
                logger.debug("KFP_AUTH_TOKEN K8s API call failed: %s", exc)

        # --- Step 3: Sidecar proxy with k8s_sa_token:ref (evalhub >= 0.4.4) ---
        try:
            result = GarakAdapter._read_secret_via_sidecar(secret_name, namespace)
            if result:
                logger.info("Read S3 credentials from secret %s/%s via sidecar proxy", namespace, secret_name)
                return _extract_from_secret_data(result)
        except Exception as exc:
            logger.debug("Sidecar proxy K8s API call failed: %s", exc)

        # --- Step 4: read_model_auth_key for S3 keys in model auth secret ---
        try:
            model_auth_creds = GarakAdapter._read_s3_from_model_auth()
            if any(model_auth_creds.values()):
                logger.info("Read S3 credentials from model auth secret")
                return model_auth_creds
        except Exception as exc:
            logger.debug("read_model_auth_key for S3 failed: %s", exc)

        # --- Step 5: load_kube_config (local dev) ---
        try:
            from kubernetes import client as k8s_client, config as k8s_config

            k8s_config.load_kube_config()
            v1 = k8s_client.CoreV1Api()
            secret = v1.read_namespaced_secret(secret_name, namespace)
            logger.info("Read S3 credentials from secret %s/%s via kubeconfig", namespace, secret_name)
            return _extract_from_secret_data(secret.data or {})
        except Exception as exc:
            logger.warning("Could not read S3 credentials from secret %s/%s: %s", namespace, secret_name, exc)
            return {}

    @staticmethod
    def _read_secret_via_sidecar(secret_name: str, namespace: str) -> dict | None:
        """Read a K8s secret via the evalhub sidecar proxy.

        Uses the ``k8s_sa_token:ref`` / ``k8s_url`` convention from evalhub
        PR #681.  The sidecar resolves the ref token (injecting the pod SA
        token when the secret value is empty) and routes the request to the
        K8s API server specified by ``k8s_url``.

        Returns the secret's ``.data`` dict (base64-encoded values) on
        success, or None if the sidecar proxy is not configured.
        """
        try:
            from evalhub.adapter.auth import read_model_auth_key
        except ImportError:
            return None

        k8s_url = (read_model_auth_key("k8s_url") or "").strip()
        if not k8s_url:
            return None

        import urllib.request

        token_header = "Bearer k8s_sa_token:ref"
        api_path = f"/api/v1/namespaces/{namespace}/secrets/{secret_name}"
        sidecar_url = f"http://localhost:8080{api_path}"

        req = urllib.request.Request(
            sidecar_url,
            headers={
                "Authorization": token_header,
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
            return body.get("data", {})

    @staticmethod
    def _read_s3_from_model_auth() -> dict:
        """Read S3 credentials from the model auth secret.

        Uses ``read_model_auth_key`` to read S3 credential keys that the
        user may have placed directly in the model auth secret.  This is
        a last-resort fallback when the adapter cannot read the Data
        Connection secret via the K8s API.
        """
        try:
            from evalhub.adapter.auth import read_model_auth_key
        except ImportError:
            return {}

        _MODEL_AUTH_S3_KEYS: list[tuple[str, list[str]]] = [
            ("access_key", ["AWS_ACCESS_KEY_ID", "aws_access_key_id", "s3_access_key"]),
            ("secret_key", ["AWS_SECRET_ACCESS_KEY", "aws_secret_access_key", "s3_secret_key"]),
            ("region", ["AWS_DEFAULT_REGION", "aws_default_region", "s3_region"]),
            ("bucket", ["AWS_S3_BUCKET", "aws_s3_bucket", "s3_bucket"]),
            ("endpoint_url", ["AWS_S3_ENDPOINT", "aws_s3_endpoint", "s3_endpoint"]),
        ]
        result: dict[str, str] = {}
        for field, candidates in _MODEL_AUTH_S3_KEYS:
            val = ""
            for key in candidates:
                val = (read_model_auth_key(key) or "").strip()
                if val:
                    break
            result[field] = val
        return result

    @staticmethod
    def _resolve_s3_credentials(kfp_config: "KFPConfig", secret_creds: dict) -> dict:
        """Merge S3 credentials from KFPConfig, K8s secret, and env vars.

        For ``bucket`` and ``endpoint_url``, the first non-empty value wins:
          1. kfp_config (benchmark_config / env override)
          2. K8s secret (read via _read_s3_credentials_from_secret)
          3. Environment variable (``AWS_S3_BUCKET`` / ``AWS_S3_ENDPOINT``)

        For ``access_key``, ``secret_key``, and ``region``, the first
        non-empty value wins:
          1. K8s secret (read via _read_s3_credentials_from_secret)
          2. Environment variable (``AWS_ACCESS_KEY_ID`` /
             ``AWS_SECRET_ACCESS_KEY`` / ``AWS_DEFAULT_REGION``)

        When both kfp_config and the secret provide different non-empty
        values for the same field, the kfp_config value is used and a
        warning is logged so the operator can spot misconfigurations.
        """
        _FIELD_MAP: list[tuple[str, str, str]] = [
            # (field_name, kfp_config_attr, env_var)
            ("bucket", "s3_bucket", "AWS_S3_BUCKET"),
            ("endpoint_url", "s3_endpoint", "AWS_S3_ENDPOINT"),
        ]

        resolved: dict[str, str | None] = {}

        for field, kfp_attr, env_var in _FIELD_MAP:
            cfg_val = (getattr(kfp_config, kfp_attr, "") or "").strip()
            secret_val = (secret_creds.get(field, "") or "").strip()
            env_val = (os.getenv(env_var, "") or "").strip()

            if cfg_val and secret_val and cfg_val != secret_val:
                logger.warning(
                    "S3 %s mismatch: kfp_config=%r vs secret=%r — using kfp_config value. "
                    "Check that kfp_config.%s and the Data Connection secret agree.",
                    field,
                    cfg_val,
                    secret_val,
                    kfp_attr,
                )

            resolved[field] = cfg_val or secret_val or env_val or None

        _CRED_ENV_MAP = {
            "access_key": "AWS_ACCESS_KEY_ID",
            "secret_key": "AWS_SECRET_ACCESS_KEY",
            "region": "AWS_DEFAULT_REGION",
        }
        for field, env_var in _CRED_ENV_MAP.items():
            secret_val = (secret_creds.get(field, "") or "").strip()
            env_val = (os.getenv(env_var, "") or "").strip()
            resolved[field] = secret_val or env_val or None

        return resolved

    @staticmethod
    def _create_s3_client(
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
    ):
        """Create a boto3 S3 client.

        Explicit parameters take precedence over environment variables, which
        allows the adapter pod to supply credentials read from a k8s secret
        when they are not present in its own environment.
        """
        from .s3_utils import create_s3_client

        return create_s3_client(
            endpoint_url=endpoint_url,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
        )

    @staticmethod
    def _download_results_from_s3(
        bucket: str,
        prefix: str,
        local_dir: Path,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
    ) -> None:
        """Download all scan result files from S3 to a local directory.

        Uses pagination to handle prefixes with many objects.
        Explicit credential parameters take precedence over environment variables.
        """
        if not bucket:
            logger.warning("No S3 bucket configured; skipping result download")
            return

        s3 = GarakAdapter._create_s3_client(
            endpoint_url=endpoint_url,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
        )

        paginator = s3.get_paginator("list_objects_v2")
        downloaded = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                relative = key[len(prefix) :].lstrip("/")
                if not relative:
                    continue
                local_path = local_dir / relative
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(local_path))
                downloaded += 1

        logger.info(
            "Downloaded %d files from s3://%s/%s to %s",
            downloaded,
            bucket,
            prefix,
            local_dir,
        )

    def _resolve_profile(self, benchmark_id: str) -> dict:
        """Resolve a benchmark_id to a GarakScanConfig profile dict.

        Checks both FRAMEWORK_PROFILES and SCAN_PROFILES, with and without
        the 'trustyai_garak::' prefix, so that bare IDs like 'owasp_llm_top10'
        and fully-qualified IDs like 'trustyai_garak::owasp_llm_top10' both work.
        """
        return resolve_scan_profile(benchmark_id)

    def _validate_config(self, config: JobSpec) -> None:
        """Validate job configuration."""
        if not config.benchmark_id:
            raise ValueError("benchmark_id is required")

        if not config.model.url:
            raise ValueError("model.url is required")

        if not config.model.name:
            raise ValueError("model.name is required")

        profile = self._resolve_profile(config.benchmark_id)
        benchmark_config = config.parameters or {}
        if not isinstance(benchmark_config.get("save_sample_results", True), bool):
            raise ValueError("save_sample_results must be a boolean")
        explicit_garak_cfg = benchmark_config.get("garak_config", {})
        if not isinstance(explicit_garak_cfg, dict):
            explicit_garak_cfg = {}

        explicit_plugins = explicit_garak_cfg.get("plugins", {})
        explicit_run = explicit_garak_cfg.get("run", {})
        if not isinstance(explicit_plugins, dict):
            explicit_plugins = {}
        if not isinstance(explicit_run, dict):
            explicit_run = {}

        explicit_selection = (
            any(benchmark_config.get(key) for key in ("probes", "probe_tags", "buffs"))
            or any(explicit_plugins.get(key) for key in ("probe_spec", "buff_spec"))
            or explicit_run.get("probe_tags")
            or "spec" in explicit_run
        )

        if not explicit_selection and not profile:
            logger.warning(
                "benchmark_id '%s' does not match a known profile and no probes, buffs, "
                "or probe_tags provided in parameters — all probes will run",
                config.benchmark_id,
            )

        logger.debug("Configuration validated successfully")

    def _build_config_from_spec(
        self, config: JobSpec, report_prefix: Path
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Build Garak command config dict from JobSpec.

        Returns:
            Tuple of (garak_config_dict, profile, intents_params) where
            intents_params contains art_intents and related settings
            extracted from the profile and benchmark_config.
        """
        benchmark_config = dict(config.parameters or {})
        num_examples = getattr(config, "num_examples", None)
        if num_examples is not None:
            benchmark_config["num_examples"] = num_examples
        profile = self._resolve_profile(config.benchmark_id)
        garak_config: GarakCommandConfig = build_effective_garak_config(
            benchmark_config=benchmark_config,
            profile=profile,
        )

        if profile:
            logger.info(
                "Resolved benchmark_id '%s' to profile '%s'",
                config.benchmark_id,
                profile.get("name"),
            )

        if hasattr(config.model, "parameters"):
            model_params = config.model.parameters
        else:
            model_params = benchmark_config.get("model_parameters")

        model_type = benchmark_config.get("model_type", DEFAULT_MODEL_TYPE)
        # set generators if not already set by user (because there's a eval-hub config.model for this)
        if model_type == DEFAULT_MODEL_TYPE and not garak_config.plugins.generators:
            from evalhub.adapter.auth import read_model_auth_key

            api_key = (
                getattr(config.model, "api_key", None)
                or read_model_auth_key("api-key")
                or os.getenv("OPENAICOMPATIBLE_API_KEY")
                or "DUMMY"
            )
            raw_model_url = self._resolve_kfp_model_url(config.model.url, benchmark_config)
            garak_config.plugins.generators = build_generator_options(
                model_endpoint=raw_model_url.strip().rstrip("/"),
                model_name=config.model.name,
                api_key=api_key,
                extra_params=model_params or TARGET_DEFAULT_PARAMETERS,
            )

        garak_config.plugins.target_type = model_type
        garak_config.plugins.target_name = config.model.name
        garak_config.reporting.report_prefix = str(report_prefix)

        art_intents = (
            bool(benchmark_config.get("art_intents"))
            if "art_intents" in benchmark_config
            else bool(profile.get("art_intents", False))
        )
        intents_params: dict[str, Any] = {
            "art_intents": art_intents,
            "policy_s3_key": benchmark_config.get("policy_s3_key", profile.get("policy_s3_key", "")),
            "policy_format": benchmark_config.get("policy_format", profile.get("policy_format", "csv")),
            "intents_s3_key": benchmark_config.get("intents_s3_key", profile.get("intents_s3_key", "")),
            "intents_format": benchmark_config.get("intents_format", profile.get("intents_format", "csv")),
            "sdg_flow_id": benchmark_config.get("sdg_flow_id", profile.get("sdg_flow_id", DEFAULT_SDG_FLOW_ID)),
            "sdg_max_concurrency": safe_int(
                benchmark_config.get(
                    "sdg_max_concurrency", profile.get("sdg_max_concurrency", DEFAULT_SDG_MAX_CONCURRENCY)
                ),
                DEFAULT_SDG_MAX_CONCURRENCY,
            ),
            "sdg_num_samples": safe_int(
                benchmark_config.get("sdg_num_samples", profile.get("sdg_num_samples", DEFAULT_SDG_NUM_SAMPLES)),
                DEFAULT_SDG_NUM_SAMPLES,
            ),
            "sdg_max_tokens": safe_int(
                benchmark_config.get("sdg_max_tokens", profile.get("sdg_max_tokens", DEFAULT_SDG_MAX_TOKENS)),
                DEFAULT_SDG_MAX_TOKENS,
            ),
            "disable_cache": as_bool(benchmark_config.get("disable_cache", False)),
        }

        if art_intents:
            # A probe-only override replaces the profile include list. Restore the
            # independent intent axis so generated typologies do not use Garak's
            # built-in "S" default, which is absent from job-specific typologies.
            spec = garak_config.run.spec
            if spec is not None and not any(
                isinstance(selector, dict) and "intent" in selector for selector in spec.include
            ):
                spec.include.append({"intent": "all"})

            sdg_params, attacker_info = self._apply_intents_model_config(
                garak_config, benchmark_config, profile, model_url=config.model.url
            )
            intents_params.update(sdg_params)

            from ..core.pipeline_steps import build_translation_langproviders

            spec_include = garak_config.run.spec.include if garak_config.run.spec else []
            resolved_probe_spec = ",".join(selector for selector in spec_include if isinstance(selector, str))

            langproviders = build_translation_langproviders(
                benchmark_config,
                attacker_url=attacker_info.get("url", ""),
                attacker_name=attacker_info.get("name", ""),
                attacker_api_key=attacker_info.get("api_key", ""),
                probe_spec=resolved_probe_spec,
                model_url=config.model.url,
            )
            if langproviders is not None:
                garak_config.run.langproviders = langproviders

        return garak_config.to_dict(exclude_none=True), profile, intents_params

    # ------------------------------------------------------------------
    # Intents model configuration
    # ------------------------------------------------------------------

    def _apply_intents_model_config(
        self,
        garak_config: GarakCommandConfig,
        benchmark_config: dict,
        profile: dict,
        model_url: str = "",
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Configure judge/attacker/evaluator/SDG models from ``intents_models``.

        Users provide per-role model endpoints in ``benchmark_config``:

        .. code-block:: json

            {
                "intents_models": {
                    "judge":     {"url": "...", "name": "..."},
                    "attacker":  {"url": "...", "name": "..."},
                    "evaluator": {"url": "...", "name": "..."},
                    "sdg":       {"url": "...", "name": "..."}
                }
            }

        Exactly 1 or all 3 of judge/attacker/evaluator must have a ``url``:

        - **1 provided**: the configured role is used for all three.
        - **3 provided**: each role uses its own config.
        - **0 provided, but models pre-configured in garak_config**: the
          override is skipped and models from garak_config are used as-is.
          SDG params are extracted from flat keys (``sdg_model``,
          ``sdg_api_base``).
        - **0 provided, no garak_config models**: raises ``ValueError``.
        - **2 provided**: raises ``ValueError`` — ambiguous which should fill
          the missing role.

        ``sdg`` has no fallback — it must be provided explicitly if SDG
        generation is needed.

        API keys are **no longer** embedded in the config dict.  They are
        injected at pod level via a Kubernetes Secret
        (``model_auth_secret_name``).  Placeholder values (``__FROM_ENV__``)
        are written into the config and resolved inside the KFP pod by
        ``core.pipeline_steps._resolve_config_api_keys``.

        Returns:
            Tuple of (sdg_params, attacker_info) where sdg_params contains
            SDG-related keys and attacker_info contains the resolved
            attacker ``url`` and ``name`` (used for translation fallback).
        """
        intents_models = benchmark_config.get("intents_models", {})
        if not isinstance(intents_models, dict):
            intents_models = {}

        judge_cfg = intents_models.get("judge") or {}
        attacker_cfg = intents_models.get("attacker") or {}
        evaluator_cfg = intents_models.get("evaluator") or {}
        sdg_cfg = intents_models.get("sdg") or {}

        provided = {
            role: cfg
            for role, cfg in [("judge", judge_cfg), ("attacker", attacker_cfg), ("evaluator", evaluator_cfg)]
            if cfg.get("url")
        }

        # Check if model auth secret has role URLs (secret-only config)
        secret_has_roles = False
        if len(provided) == 0:
            judge_probe_url, _ = self._read_role_from_secret("judge", model_url)
            if judge_probe_url:
                secret_has_roles = True

        if len(provided) == 0 and not secret_has_roles:
            if garak_config.plugins and self._models_preconfigured_in_garak_config(garak_config.plugins):
                logger.info(
                    "No intents_models provided but models are already "
                    "configured in garak_config — skipping intents_models "
                    "override."
                )
                preconfigured_attacker = self._extract_preconfigured_attacker(garak_config.plugins)
                return self._extract_sdg_params(sdg_cfg, benchmark_config, profile), preconfigured_attacker

            # Check if names are provided (secret has URLs, user only provides names)
            has_names = any(cfg.get("name") for cfg in [judge_cfg, attacker_cfg, evaluator_cfg])
            if not has_names:
                raise ValueError(
                    "Intents benchmark requires model configuration for "
                    "judge/attacker/evaluator roles. Either:\n"
                    "  1. Provide intents_models with at least one role "
                    "(url + name, or just name if URLs are in the model auth secret), or\n"
                    "  2. Configure models directly in garak_config "
                    "(plugins.detectors.judge with detector_model_name and "
                    "detector_model_config.uri)."
                )

        if len(provided) == 2:
            missing = {"judge", "attacker", "evaluator"} - set(provided)
            raise ValueError(
                f"Ambiguous intents_models config: {', '.join(sorted(provided))} "
                f"are configured but {', '.join(sorted(missing))} is missing. "
                f"Provide all three, or provide exactly one to use for all roles."
            )

        if len(provided) == 1:
            base_cfg = next(iter(provided.values()))
            judge_cfg = {**base_cfg, **judge_cfg}
            attacker_cfg = {**base_cfg, **attacker_cfg}
            evaluator_cfg = {**base_cfg, **evaluator_cfg}

        # Fan-out names: if only judge has a name (secret-based config
        # where URLs come from secret, user only specifies judge name),
        # propagate it to attacker/evaluator.
        if judge_cfg.get("name") and not attacker_cfg.get("name"):
            attacker_cfg["name"] = judge_cfg["name"]
        if judge_cfg.get("name") and not evaluator_cfg.get("name"):
            evaluator_cfg["name"] = judge_cfg["name"]

        for role, cfg in [("judge", judge_cfg), ("attacker", attacker_cfg), ("evaluator", evaluator_cfg)]:
            if not cfg.get("name"):
                raise ValueError(f"intents_models.{role}.name is required. Provide a model identifier for the {role}.")

        judge_url = judge_cfg.get("url", "")
        judge_name = judge_cfg["name"]

        attacker_url = attacker_cfg.get("url", "")
        attacker_name = attacker_cfg["name"]

        evaluator_url = evaluator_cfg.get("url", "")
        evaluator_name = evaluator_cfg["name"]

        _PLACEHOLDER = "__FROM_ENV__"

        judge_secret_url, judge_secret_key = self._read_role_from_secret("judge", model_url)
        attacker_secret_url, attacker_secret_key = self._read_role_from_secret("attacker", model_url)
        evaluator_secret_url, evaluator_secret_key = self._read_role_from_secret("evaluator", model_url)

        # Fan-out: if judge has secret credentials but attacker/evaluator
        # don't, inherit from judge (same model for all roles).
        if judge_secret_url:
            if not attacker_secret_url and not attacker_secret_key:
                attacker_secret_url, attacker_secret_key = judge_secret_url, judge_secret_key
            if not evaluator_secret_url and not evaluator_secret_key:
                evaluator_secret_url, evaluator_secret_key = judge_secret_url, judge_secret_key

        effective_judge_url = judge_secret_url or judge_url
        effective_judge_key = judge_secret_key or _PLACEHOLDER

        effective_attacker_url = attacker_secret_url or attacker_url
        effective_attacker_key = attacker_secret_key or _PLACEHOLDER

        effective_evaluator_url = evaluator_secret_url or evaluator_url
        effective_evaluator_key = evaluator_secret_key or _PLACEHOLDER

        for role, eff_url, name, from_secret in [
            ("judge", effective_judge_url, judge_name, judge_secret_url),
            ("attacker", effective_attacker_url, attacker_name, attacker_secret_url),
            ("evaluator", effective_evaluator_url, evaluator_name, evaluator_secret_url),
        ]:
            source = "model auth secret" if from_secret else "intents_models (job request)"
            logger.info("Intents %s model: name=%s, url=%s (source: %s)", role, name, eff_url, source)

        plugins = garak_config.plugins

        plugins.detectors = plugins.detectors or {}
        existing_judge = plugins.detectors.get("judge", {})
        existing_judge["detector_model_type"] = existing_judge.get("detector_model_type") or "openai.OpenAICompatible"
        existing_judge["detector_model_name"] = existing_judge.get("detector_model_name") or judge_name
        existing_det_cfg = existing_judge.get("detector_model_config", {})
        existing_det_cfg["uri"] = existing_det_cfg.get("uri") or effective_judge_url
        existing_det_cfg["api_key"] = effective_judge_key
        existing_judge["detector_model_config"] = existing_det_cfg
        plugins.detectors["judge"] = existing_judge

        if plugins.probes and plugins.probes.get("tap"):
            tap_cfg = plugins.probes["tap"].get("TAPIntent", {})
            if isinstance(tap_cfg, dict):
                tap_cfg["attack_model_name"] = tap_cfg.get("attack_model_name") or attacker_name
                existing_attack_cfg = tap_cfg.get("attack_model_config", {})
                existing_attack_cfg.setdefault("max_tokens", 500)
                existing_attack_cfg["uri"] = existing_attack_cfg.get("uri") or effective_attacker_url
                existing_attack_cfg["api_key"] = effective_attacker_key
                tap_cfg["attack_model_config"] = existing_attack_cfg

                tap_cfg["evaluator_model_name"] = tap_cfg.get("evaluator_model_name") or evaluator_name
                existing_eval_cfg = tap_cfg.get("evaluator_model_config", {})
                existing_eval_cfg.setdefault("max_tokens", 10)
                existing_eval_cfg.setdefault("temperature", 0.0)
                existing_eval_cfg["uri"] = existing_eval_cfg.get("uri") or effective_evaluator_url
                existing_eval_cfg["api_key"] = effective_evaluator_key
                tap_cfg["evaluator_model_config"] = existing_eval_cfg

                plugins.probes["tap"]["TAPIntent"] = tap_cfg

        attacker_info = {"url": effective_attacker_url, "name": attacker_name, "api_key": effective_attacker_key}
        return self._extract_sdg_params(sdg_cfg, benchmark_config, profile), attacker_info

    @staticmethod
    def _models_preconfigured_in_garak_config(plugins: Any) -> bool:
        """Check if intents models are already configured in garak_config.

        Returns True when the judge detector has a non-empty
        ``detector_model_name`` and ``detector_model_config.uri``.
        If TAPIntent is present in the probes, also requires non-empty
        attack and evaluator model names and URIs.
        """
        detectors = plugins.detectors or {}
        judge = detectors.get("judge", {})
        if not judge.get("detector_model_name") or not judge.get("detector_model_config", {}).get("uri"):
            return False

        probes = plugins.probes or {}
        tap_cfg = probes.get("tap", {}).get("TAPIntent")
        if tap_cfg and isinstance(tap_cfg, dict):
            for name_key, cfg_key in [
                ("attack_model_name", "attack_model_config"),
                ("evaluator_model_name", "evaluator_model_config"),
            ]:
                if not tap_cfg.get(name_key) or not tap_cfg.get(cfg_key, {}).get("uri"):
                    return False

        return True

    @staticmethod
    def _extract_preconfigured_attacker(plugins: Any) -> dict[str, str]:
        """Extract attacker url/name from pre-configured garak_config plugins."""
        probes = plugins.probes or {}
        tap_cfg = probes.get("tap", {}).get("TAPIntent")
        if tap_cfg and isinstance(tap_cfg, dict):
            return {
                "url": tap_cfg.get("attack_model_config", {}).get("uri", ""),
                "name": tap_cfg.get("attack_model_name", ""),
            }
        return {"url": "", "name": ""}

    @staticmethod
    def _extract_sdg_params(
        sdg_cfg: dict,
        benchmark_config: dict,
        profile: dict,
    ) -> dict[str, str]:
        """Extract SDG model params from intents_models.sdg or flat keys.

        The SDG URL may come from the model auth secret (sdg_url) rather
        than the job request, so ``name`` alone is sufficient — the URL
        will be resolved later from the secret in ``_run_simple_intents``.
        """
        sdg_params: dict[str, str] = {
            "sdg_model": "",
            "sdg_api_base": "",
            "sdg_provider": sdg_cfg.get("provider") or benchmark_config.get("sdg_provider") or "",
        }
        if sdg_cfg.get("name"):
            sdg_params["sdg_model"] = sdg_cfg["name"]
            sdg_params["sdg_api_base"] = sdg_cfg.get("url", "")
        else:
            sdg_model = benchmark_config.get("sdg_model", profile.get("sdg_model", ""))
            sdg_api_base = benchmark_config.get("sdg_api_base", profile.get("sdg_api_base", ""))
            sdg_params["sdg_model"] = sdg_model
            sdg_params["sdg_api_base"] = sdg_api_base
        return sdg_params

    @staticmethod
    def _resolve_intents_api_key(role: str, model_cfg: dict) -> str:
        """Resolve API key for an intents model role.

        Resolution order:

        1. Direct ``api_key`` value in ``model_cfg``
        2. ``api_key_name`` -> read from mounted model auth secret
        3. ``api_key_env`` -> read from named environment variable
        4. Convention: ``{role}-api-key`` from mounted secret
        5. Convention: ``{ROLE}_API_KEY`` from environment
        6. Generic: ``api-key`` from mounted secret
        7. Generic: ``OPENAICOMPATIBLE_API_KEY`` from environment
        8. ``"DUMMY"``
        """
        if model_cfg.get("api_key"):
            return model_cfg["api_key"]

        _read_secret: Any = None
        try:
            from evalhub.adapter.auth import read_model_auth_key

            _read_secret = read_model_auth_key
        except ImportError:
            pass

        if _read_secret and model_cfg.get("api_key_name"):
            key = _read_secret(model_cfg["api_key_name"])
            if key:
                return key

        if model_cfg.get("api_key_env"):
            key = os.getenv(model_cfg["api_key_env"])
            if key:
                return key

        if _read_secret:
            key = _read_secret(f"{role}-api-key")
            if key:
                return key

        key = os.getenv(f"{role.upper()}_API_KEY")
        if key:
            return key

        if _read_secret:
            key = _read_secret("api-key")
            if key:
                return key

        key = os.getenv("OPENAICOMPATIBLE_API_KEY")
        if key:
            return key

        return "DUMMY"

    @staticmethod
    def _is_sidecar_address(url: str) -> bool:
        return "localhost" in url or "127.0.0.1" in url

    @staticmethod
    def _read_role_from_secret(role: str, model_url: str = "") -> tuple[str | None, str | None]:
        """Read URL and api-key for a role from evalhub model auth secret.

        Returns (url, api_key) tuple.

        Resolution logic:
        - If <role>_url is in secret: use it (sidecar address)
        - If <role>_api-key is in secret but <role>_url is not: use model_url
          (sidecar default route -- same endpoint as target model)
        - If neither is in secret: returns (None, None) -- unauthenticated
        - If url is resolved but <role>_api-key is absent: fall back to
          generic 'api-key'

        Key naming follows EvalHub multi-model convention:
        <role>_url, <role>_api-key.
        """
        try:
            from evalhub.adapter.auth import read_model_auth_key
        except ImportError:
            return None, None

        url = read_model_auth_key(f"{role}_url")
        api_key = read_model_auth_key(f"{role}_api-key")

        if not url and not api_key:
            return None, None

        if not url:
            url = model_url

        if not api_key:
            api_key = read_model_auth_key("api-key")

        return url, api_key

    @staticmethod
    def _resolve_kfp_model_url(model_url: str, benchmark_config: dict) -> str:
        """Resolve the target model URL for KFP component pods.

        evalhub >= 0.4.4 rewrites job.json so model.url points at the
        sidecar proxy (localhost:8080).  That works in simple mode (sidecar
        is co-located) but KFP component pods have no sidecar.

        Resolution chain (first non-sidecar URL wins):
        1. kfp_config.model_url   — explicit override in benchmark_config
        2. model auth secret key  — read_model_auth_key("model_url")
        3. intents_models.*.url   — first non-localhost URL from
           judge/attacker/evaluator config
        4. model_url as-is        — fallback (works for simple mode and
           pre-sidecar evalhub)
        """
        url = model_url.strip().rstrip("/")

        if not GarakAdapter._is_sidecar_address(url):
            return url

        kfp_overrides = benchmark_config.get("kfp_config", {}) or {}
        explicit = (kfp_overrides.get("model_url") or "").strip()
        if explicit:
            logger.info("Using explicit kfp_config.model_url: %s", explicit)
            return explicit.rstrip("/")

        try:
            from evalhub.adapter.auth import read_model_auth_key

            secret_url = (read_model_auth_key("model_url") or "").strip()
            if secret_url and not GarakAdapter._is_sidecar_address(secret_url):
                logger.info("Resolved model URL from model auth secret: %s", secret_url)
                return secret_url.rstrip("/")
        except Exception:
            pass

        intents_models = benchmark_config.get("intents_models", {})
        if isinstance(intents_models, dict):
            for role in ("judge", "attacker", "evaluator"):
                role_url = (intents_models.get(role) or {}).get("url", "").strip()
                if role_url and not GarakAdapter._is_sidecar_address(role_url):
                    logger.info("Resolved model URL from intents_models.%s: %s", role, role_url)
                    return role_url.rstrip("/")

        if kfp_overrides:
            logger.warning(
                "Model URL is a sidecar address (%s) but could not be resolved "
                "to a real upstream URL. KFP component pods have no sidecar "
                "proxy. Set kfp_config.model_url in benchmark_config or add a "
                "model_url key to the model auth secret.",
                url,
            )

        return url

    # def _build_environment(self, config: JobSpec) -> dict[str, str]:
    #     """Build environment variables for Garak execution."""
    #     env = {
    #         "GARAK_TLS_VERIFY": os.getenv("GARAK_TLS_VERIFY", "True"),
    #     }

    #     # Pass through relevant env vars
    #     for var in ["MODEL_API_KEY", "OPENAI_API_KEY", "OPENAICOMPATIBLE_API_KEY"]:
    #         if val := os.getenv(var):
    #             env[var] = val

    #     return env

    @staticmethod
    def _pct_to_ratio(value: float) -> float:
        """Convert a 0-100 percentage to a 0-1 ratio."""
        return round(value / 100, 4)

    def _parse_results(
        self,
        result: GarakScanResult,
        eval_threshold: float,
        art_intents: bool = False,
    ) -> tuple[list[EvaluationResult], float | None, int, dict[str, Any]]:
        """Parse Garak results into EvaluationResult metrics.

        Args:
            result: Scan result with report file paths.
            eval_threshold: Threshold for vulnerability detection (0-1).
            art_intents: When True, use prompt-level intents aggregation
                instead of AVID-based aggregation.

        Returns:
            Tuple of (metrics list, overall_score, num_examples, overall summary)
        """
        metrics: list[EvaluationResult] = []
        total_attempts = 0

        # Read report content
        report_content = ""
        if result.report_jsonl.exists():
            report_content = result.report_jsonl.read_text()
            if not report_content.strip():
                logger.warning("Report file is empty")
                return metrics, None, 0, {}

        avid_content = ""
        if result.avid_jsonl.exists():
            avid_content = result.avid_jsonl.read_text()

        generations, score_rows_by_probe, raw_entries_by_probe = parse_generations_from_report_content(
            report_content, eval_threshold, art_intents=art_intents
        )
        aggregated_by_probe = parse_aggregated_from_avid_content(avid_content)
        digest = parse_digest_from_report_content(report_content)
        harness_summary = parse_harness_summary(report_content) if art_intents else None
        harness_stub_summaries = (parse_harness_stub_summaries(report_content) or None) if art_intents else None
        legacy_intent_report = art_intents and "EarlyStopHarness" in report_content

        # Combine results
        combined = combine_parsed_results(
            generations,
            score_rows_by_probe,
            aggregated_by_probe,
            eval_threshold,
            digest,
            art_intents=art_intents,
            raw_entries_by_probe=raw_entries_by_probe,
            harness_summary=harness_summary,
            harness_stub_summaries=harness_stub_summaries,
            allow_legacy_intent_inference=legacy_intent_report,
        )
        overall_summary = combined.get("scores", {}).get("_overall", {}).get("aggregated_results", {})

        overall_asr = overall_summary.get("attack_success_rate")
        if overall_asr is not None:
            try:
                overall_asr = float(overall_asr)
            except (TypeError, ValueError):
                overall_asr = None
        if overall_asr is not None:
            metrics.append(
                EvaluationResult(
                    metric_name="attack_success_rate",
                    metric_value=self._pct_to_ratio(overall_asr),
                    metric_type="ratio",
                    num_samples=overall_summary.get("total_attempts"),
                )
            )

        # Convert to EvaluationResult format (one per probe)
        for probe_name, score_data in combined["scores"].items():
            if probe_name == "_overall":
                continue

            agg = score_data["aggregated_results"]

            attack_success_rate = agg.get("attack_success_rate", 0.0)

            probe_metadata: dict[str, Any] = {"probe": probe_name}
            if art_intents:
                probe_metadata.update(
                    {
                        "total_attempts": agg.get("total_attempts", 0),
                        "unsafe_stubs": agg.get("unsafe_stubs", 0),
                        "safe_stubs": agg.get("safe_stubs", 0),
                        "intent_breakdown": agg.get("intent_breakdown", {}),
                    }
                )
            else:
                probe_attempts = agg.get("total_attempts", agg.get("total_attacks", 0))
                total_attempts += probe_attempts
                probe_metadata.update(
                    {
                        "vulnerable_responses": agg.get("vulnerable_responses", 0),
                        "benign_responses": agg.get("benign_responses", 0),
                        "detector_scores": agg.get("detector_scores", {}),
                        "avid_taxonomy": agg.get("metadata", {}).get("avid_taxonomy", {}),
                    }
                )

            metrics.append(
                EvaluationResult(
                    metric_name=f"{probe_name}_asr",
                    metric_value=self._pct_to_ratio(attack_success_rate),
                    metric_type="ratio",
                    num_samples=probe_attempts if not art_intents else None,
                    metadata=probe_metadata,
                )
            )

        overall_score = overall_summary.get("attack_success_rate")
        if overall_score is not None:
            try:
                overall_score = self._pct_to_ratio(float(overall_score))
            except (TypeError, ValueError):
                overall_score = None

        if overall_score is None and total_attempts > 0:
            total_vulnerable = sum(
                score_data["aggregated_results"].get("vulnerable_responses", 0)
                for probe_name, score_data in combined["scores"].items()
                if probe_name != "_overall"
            )
            overall_score = self._pct_to_ratio((total_vulnerable / total_attempts) * 100)

        num_examples = overall_summary.get("total_attempts", total_attempts)
        try:
            num_examples = int(num_examples)
        except (TypeError, ValueError):
            num_examples = total_attempts

        return metrics, overall_score, num_examples, overall_summary

    def _get_garak_version(self) -> str:
        """Get Garak version string."""
        try:
            from importlib.metadata import version

            return f"garak=={version('garak')}"
        except Exception:
            return "unknown"


def main(adapter_cls: type[GarakAdapter] = GarakAdapter) -> None:
    """Entry point for running the adapter as a K8s Job.

    Reads JobSpec from mounted ConfigMap and executes the Garak scan.
    The callback URL comes from job_spec.callback_url (set by the service).
    Registry credentials come from AdapterSettings (environment variables).

    Args:
        adapter_cls: The adapter class to instantiate. Defaults to
            GarakAdapter (simple mode). Pass a subclass to override
            behaviour (e.g. GarakKFPAdapter for forced KFP execution).
    """
    import sys

    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    callbacks: DefaultCallbacks | None = None
    exit_error: str | None = None
    try:
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = adapter_cls(job_spec_path=job_spec_path)
        logger.info(f"Loaded job {adapter.job_spec.id}")
        logger.info(f"Benchmark: {adapter.job_spec.benchmark_id}")
        logger.info(f"Model: {adapter.job_spec.model.name}")

        callbacks = DefaultCallbacks.from_adapter(adapter)

        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        logger.info(f"Job completed successfully: {results.id}")
        logger.info(f"Overall attack success rate (ratio): {results.overall_score}")

        callbacks.report_results(results)
        sys.exit(0)

    except FileNotFoundError as e:
        exit_error = f"Job spec not found: {e}"
        logger.exception(f"{exit_error}. Set EVALHUB_JOB_SPEC_PATH or ensure job spec exists at default location")
        sys.exit(1)
    except ValueError as e:
        exit_error = f"Configuration error: {e}"
        logger.exception(exit_error)
        sys.exit(1)
    except Exception as e:
        exit_error = f"Job failed: {e}"
        logger.exception(exit_error)
        sys.exit(1)


if __name__ == "__main__":
    main()
