"""Eval-hub-specific KFP pipeline for running Garak scans.

This module provides a Kubeflow Pipeline that executes a Garak scan inside
a KFP pod, uploading all output to S3 via a Data Connection secret. The
eval-hub adapter submits this pipeline, polls for completion, and downloads
results from the same S3 bucket.

Artifact transfer uses an OpenShift AI "Data Connection" -- a Kubernetes
Secret with standard S3 credential keys. The secret is injected into the
KFP pods via ``kubernetes.use_secret_as_env``, and into the eval-hub Job
pod via ``envFrom`` (configured by the eval-hub service).

The pipeline has six steps:
1. **validate** -- pre-flight checks (garak install, config, S3, model).
2. **resolve_taxonomy** -- resolve the taxonomy input (custom from S3 or
   default BASE_TAXONOMY).
3. **sdg_generate** -- run SDG on the taxonomy to produce raw prompts,
   or emit an empty marker for non-intents / bypass-sdg scans.
4. **prepare_prompts** -- fetch bypass data from S3 as raw dataset (if applicable),
   upload raw output (either from SDG or bypass data) to S3, normalise, upload normalised output to S3,
   and emit the normalised prompts for garak.
5. **garak_scan** -- run the scan and upload results to S3.
6. **write_kfp_outputs** -- parse results from S3 and write KFP Metrics /
   HTML artifacts so results are visible in the KFP dashboard.
"""

import logging
import os
from dataclasses import dataclass
from typing import NamedTuple

from kfp import dsl, kubernetes

from ..constants import (
    DEFAULT_MAX_RETRY_SECONDS,
    DEFAULT_SDG_FLOW_ID,
    DEFAULT_SDG_MAX_CONCURRENCY,
    DEFAULT_SDG_NUM_SAMPLES,
    DEFAULT_SDG_MAX_TOKENS,
)
from ..core.pipeline_steps import MODEL_AUTH_MOUNT_PATH

logger = logging.getLogger(__name__)

DEFAULT_KFP_EXPERIMENT = "evalhub-garak"
DEFAULT_POLL_INTERVAL = 30
DEFAULT_S3_PREFIX = "evalhub-garak-kfp"

S3_DATA_CONNECTION_KEYS = {
    "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
    "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
    "AWS_S3_BUCKET": "AWS_S3_BUCKET",
    "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
}


@dataclass
class KFPConfig:
    """Configuration for connecting to and running pipelines on KFP.

    Resolved from environment variables with benchmark_config overrides.
    """

    endpoint: str
    namespace: str
    auth_token: str = ""
    s3_secret_name: str = ""
    s3_bucket: str = ""
    s3_endpoint: str = ""
    experiment_name: str = DEFAULT_KFP_EXPERIMENT
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL
    s3_prefix: str = DEFAULT_S3_PREFIX
    base_image: str = ""
    verify_ssl: bool = True
    ssl_ca_cert: str = ""

    @classmethod
    def from_env_and_config(cls, benchmark_config: dict | None = None) -> "KFPConfig":
        """Build KFPConfig from env vars, with benchmark_config overrides.

        Resolution precedence (lowest to highest):
        1. Dataclass defaults
        2. Model auth secret keys read via evalhub read_model_auth_key
           (new sidecar-proxy model — evalhub >= 0.4.4):
           - ``kfp_url``      → read_model_auth_key returns ``http://localhost:8080``
           - ``kfp_sa_token`` → read_model_auth_key returns ``kfp_sa_token:ref``
             (a placeholder the sidecar swaps with the real SA token on every request)
        3. Environment variables (KFP_*)
        4. benchmark_config dict keys
        """
        bc = benchmark_config or {}
        kfp_overrides = bc.get("kfp_config", {})
        if not isinstance(kfp_overrides, dict):
            kfp_overrides = {}

        # New sidecar-proxy model (evalhub >= 0.4.4): KFP connection details
        # come from the model auth secret rather than env vars. The secret key
        # names use the operator-chosen prefix (default: "kfp") followed by
        # "_url" and "_sa_token". read_model_auth_key resolves the mounted
        # secret file and returns the placeholder ref for the sidecar to swap.
        _secret_kfp_url = ""
        _secret_kfp_token = ""
        try:
            from evalhub.adapter.auth import read_model_auth_key

            _secret_kfp_url = read_model_auth_key("kfp_url") or ""
            _secret_kfp_token = read_model_auth_key("kfp_sa_token") or ""
        except Exception:
            pass

        def _resolve(key: str, env_var: str, default: str = "") -> str:
            return str(kfp_overrides.get(key, os.getenv(env_var, default)))

        endpoint = _resolve("endpoint", "KFP_ENDPOINT", _secret_kfp_url)
        if not endpoint:
            raise ValueError(
                "KFP endpoint is required. Set KFP_ENDPOINT, provide kfp_config.endpoint "
                "in benchmark_config, or add a kfp_url key to the model auth secret."
            )

        namespace = _resolve("namespace", "KFP_NAMESPACE")
        if not namespace:
            raise ValueError(
                "KFP namespace is required. Set KFP_NAMESPACE or provide kfp_config.namespace in benchmark_config."
            )

        verify_ssl_raw = _resolve("verify_ssl", "KFP_VERIFY_SSL", "true")
        verify_ssl = verify_ssl_raw.lower() not in ("false", "0", "no", "off")

        return cls(
            endpoint=endpoint,
            namespace=namespace,
            auth_token=_resolve("auth_token", "KFP_AUTH_TOKEN", _secret_kfp_token),
            s3_secret_name=_resolve("s3_secret_name", "KFP_S3_SECRET_NAME"),
            s3_bucket=_resolve("s3_bucket", "AWS_S3_BUCKET"),
            s3_endpoint=_resolve("s3_endpoint", "AWS_S3_ENDPOINT"),
            experiment_name=_resolve(
                "experiment_name",
                "KFP_EXPERIMENT",
                DEFAULT_KFP_EXPERIMENT,
            ),
            poll_interval_seconds=int(
                _resolve("poll_interval_seconds", "KFP_POLL_INTERVAL", str(DEFAULT_POLL_INTERVAL))
            ),
            s3_prefix=_resolve(
                "s3_prefix",
                "KFP_S3_PREFIX",
                DEFAULT_S3_PREFIX,
            ),
            base_image=_resolve("base_image", "KUBEFLOW_GARAK_BASE_IMAGE"),
            verify_ssl=verify_ssl,
            ssl_ca_cert=_resolve("ssl_ca_cert", "KFP_SSL_CA_CERT"),
        )


def _resolve_base_image(kfp_config: KFPConfig | None = None) -> str:
    """Resolve the container image for KFP components.

    Priority: KFPConfig.base_image > env var > k8s ConfigMap > default.
    """
    import os
    import logging

    from ..constants import (
        GARAK_PROVIDER_IMAGE_CONFIGMAP_NAME,
        GARAK_PROVIDER_IMAGE_CONFIGMAP_KEY,
        KUBEFLOW_CANDIDATE_NAMESPACES,
        DEFAULT_GARAK_PROVIDER_IMAGE,
    )

    _logger = logging.getLogger(__name__)

    if kfp_config and kfp_config.base_image:
        return kfp_config.base_image

    if (base_image := os.environ.get("KUBEFLOW_GARAK_BASE_IMAGE")) is not None:
        return base_image

    try:
        from kubernetes import client, config as k8s_config
        from kubernetes.client.exceptions import ApiException

        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()

        api = client.CoreV1Api()
        for candidate_namespace in KUBEFLOW_CANDIDATE_NAMESPACES:
            try:
                configmap = api.read_namespaced_config_map(
                    name=GARAK_PROVIDER_IMAGE_CONFIGMAP_NAME,
                    namespace=candidate_namespace,
                )
                data: dict[str, str] | None = configmap.data
                if data and GARAK_PROVIDER_IMAGE_CONFIGMAP_KEY in data:
                    return data[GARAK_PROVIDER_IMAGE_CONFIGMAP_KEY]
            except ApiException as api_exc:
                if api_exc.status == 404:
                    continue
                else:
                    _logger.warning(f"Could not read from ConfigMap: {api_exc}")
            except Exception as e:
                _logger.warning(f"Could not read from ConfigMap: {e}")

        return DEFAULT_GARAK_PROVIDER_IMAGE
    except Exception as e:
        _logger.debug(f"Kubernetes config not available: {e}. Using default image.")
        return DEFAULT_GARAK_PROVIDER_IMAGE


# ---------------------------------------------------------------------------
# KFP Components
# ---------------------------------------------------------------------------


@dsl.component(
    install_kfp_package=False,
    packages_to_install=[],
    base_image=_resolve_base_image(),
)
def validate(
    config_json: str,
) -> NamedTuple("Outputs", [("valid", bool)]):
    """Pre-flight validation: garak, config JSON, flags, S3 connectivity."""
    import logging
    import os
    from collections import namedtuple

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("validate")

    from llama_stack_provider_trustyai_garak.core.pipeline_steps import validate_scan_config
    from llama_stack_provider_trustyai_garak.errors import GarakValidationError

    validate_scan_config(config_json)
    log.info("Core config validation passed")

    # S3 connectivity check (EvalHub-specific)
    s3_bucket = os.environ.get("AWS_S3_BUCKET", "")
    if not s3_bucket:
        raise GarakValidationError("AWS_S3_BUCKET env var is not set")
    try:
        from llama_stack_provider_trustyai_garak.evalhub.s3_utils import create_s3_client

        s3 = create_s3_client()
        s3.head_bucket(Bucket=s3_bucket)
        log.info("S3 bucket '%s' is reachable", s3_bucket)
    except Exception as exc:
        raise GarakValidationError(f"S3 bucket '{s3_bucket}' is not reachable: {exc}") from exc

    log.info("All pre-flight checks passed")
    Outputs = namedtuple("Outputs", ["valid"])
    return Outputs(valid=True)


@dsl.component(
    install_kfp_package=False,
    packages_to_install=[],
    base_image=_resolve_base_image(),
)
def resolve_taxonomy(taxonomy_dataset: dsl.Output[dsl.Dataset], policy_s3_key: str = "", policy_format: str = "csv"):
    """Resolve taxonomy: custom from S3 or built-in BASE_TAXONOMY."""
    import logging
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("resolve_taxonomy")

    from llama_stack_provider_trustyai_garak.core.pipeline_steps import resolve_taxonomy_data
    from llama_stack_provider_trustyai_garak.errors import GarakValidationError

    policy_content = None
    if policy_s3_key and policy_s3_key.strip():
        from llama_stack_provider_trustyai_garak.evalhub.s3_utils import create_s3_client

        if policy_s3_key.startswith("s3://"):
            parts = policy_s3_key[len("s3://") :].split("/", 1)
            s3_bucket = parts[0]
            s3_key = parts[1] if len(parts) > 1 else ""
            if not s3_key:
                raise GarakValidationError(f"Invalid policy_s3_key '{policy_s3_key}': expected s3://bucket/key format")
        else:
            s3_bucket = os.environ.get("AWS_S3_BUCKET", "")
            s3_key = policy_s3_key

        if not s3_bucket:
            raise GarakValidationError(
                "Cannot determine S3 bucket for policy taxonomy. "
                "Provide a full s3://bucket/key URI in policy_s3_key, "
                "or set AWS_S3_BUCKET."
            )

        log.info(
            "Fetching policy taxonomy from S3 (bucket=%s, key=%s, format=%s)",
            s3_bucket,
            s3_key,
            policy_format,
        )
        s3 = create_s3_client()
        response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        policy_content = response["Body"].read()

    taxonomy_df = resolve_taxonomy_data(policy_content, format=policy_format)
    taxonomy_df.to_csv(taxonomy_dataset.path, index=False)
    log.info("Wrote taxonomy (%d entries) to artifact", len(taxonomy_df))


@dsl.component(
    install_kfp_package=False,
    packages_to_install=[],
    base_image=_resolve_base_image(),
)
def sdg_generate(
    art_intents: bool,
    intents_s3_key: str,
    sdg_model: str,
    sdg_api_base: str,
    sdg_flow_id: str,
    sdg_max_concurrency: int,
    sdg_num_samples: int,
    sdg_max_tokens: int,
    taxonomy_dataset: dsl.Input[dsl.Dataset],
    sdg_dataset: dsl.Output[dsl.Dataset],
    sdg_provider: str = "",
):
    """Run Synthetic Data Generation on a taxonomy to produce raw prompts.

    The SDG API key is injected into the pod as an environment variable via a Kubernetes
    Secret (``model_auth_secret_name``).  The core function
    ``run_sdg_generation`` resolves the key via ``resolve_api_key("sdg")``
    which checks env vars and volume-mounted secret files automatically.

    Two modes:

    * ``art_intents=False`` or ``intents_s3_key`` set -- writes an empty
      marker (native probes or bypass; downstream ``prepare_prompts``
      handles the bypass fetch).
    * Otherwise -- run SDG on the taxonomy artifact, emit the raw
      (un-normalised) output including all pool columns.

    Only depends on taxonomy + SDG model params so KFP caching works
    across multiple garak runs with identical inputs.
    """
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("sdg_generate")

    if not art_intents:
        log.info("Non-intents scan — writing empty dataset marker")
        with open(sdg_dataset.path, "w") as f:
            f.write("")
        return

    if intents_s3_key and intents_s3_key.strip():
        log.info("Bypass mode (intents_s3_key set) — SDG not needed")
        with open(sdg_dataset.path, "w") as f:
            f.write("")
        return

    import pandas as pd
    from llama_stack_provider_trustyai_garak.core.pipeline_steps import run_sdg_generation

    taxonomy = pd.read_csv(taxonomy_dataset.path)
    log.info("Read taxonomy: %d entries", len(taxonomy))

    raw_df = run_sdg_generation(
        taxonomy_df=taxonomy,
        sdg_model=sdg_model,
        sdg_api_base=sdg_api_base,
        sdg_provider=sdg_provider,
        sdg_flow_id=sdg_flow_id,
        sdg_max_concurrency=sdg_max_concurrency,
        sdg_num_samples=sdg_num_samples,
        sdg_max_tokens=sdg_max_tokens,
    )
    raw_df.to_csv(sdg_dataset.path, index=False)
    log.info("Wrote %d raw SDG rows to artifact", len(raw_df))


@dsl.component(
    install_kfp_package=False,
    packages_to_install=[],
    base_image=_resolve_base_image(),
)
def prepare_prompts(
    art_intents: bool,
    s3_prefix: str,
    intents_s3_key: str,
    intents_format: str,
    sdg_dataset: dsl.Input[dsl.Dataset],
    prompts_dataset: dsl.Output[dsl.Dataset],
):
    """Prepare normalised prompts for the Garak scan.

    Handles three scenarios:

    * **Non-intents** (``art_intents=False``): passes through an empty
      marker — garak uses its native probes.
    * **Bypass SDG** (``intents_s3_key`` set): fetches the user's
      pre-generated dataset from S3, uploads the original file as the
      raw artifact, normalises it via :func:`load_intents_dataset`,
      uploads the normalised version, and outputs it for garak.
    * **SDG ran** (``sdg_dataset`` non-empty): uploads the raw SDG
      output, normalises it, uploads the normalised version, and
      outputs it for garak.

    Raw and normalised CSVs are persisted to the S3 job folder
    *before* the scan starts, so they are available for review while
    the scan is still running.
    """
    import logging
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("prepare_prompts")

    if not art_intents:
        log.info("Non-intents scan — passing through empty marker")
        with open(prompts_dataset.path, "w") as f:
            f.write("")
        return

    from llama_stack_provider_trustyai_garak.evalhub.s3_utils import create_s3_client
    from llama_stack_provider_trustyai_garak.errors import GarakValidationError
    from llama_stack_provider_trustyai_garak.core.pipeline_steps import normalize_prompts

    s3_bucket = os.environ.get("AWS_S3_BUCKET", "")
    if not s3_bucket:
        raise GarakValidationError("AWS_S3_BUCKET is required. Ensure the Data Connection secret is configured.")
    if not s3_prefix:
        raise GarakValidationError("s3_prefix is required. Ensure the s3_prefix is configured in the benchmark_config.")

    s3 = create_s3_client()

    raw_content: str | None = None

    # Bypass SDG: fetch user's pre-generated dataset from S3
    if intents_s3_key and intents_s3_key.strip():
        if intents_s3_key.startswith("s3://"):
            parts = intents_s3_key[len("s3://") :].split("/", 1)
            fetch_bucket = parts[0]
            fetch_key = parts[1] if len(parts) > 1 else ""
            if not fetch_key:
                raise GarakValidationError(
                    f"Invalid intents_s3_key '{intents_s3_key}': expected s3://bucket/key format"
                )
        else:
            fetch_bucket = s3_bucket
            fetch_key = intents_s3_key

        log.info(
            "Bypass mode — fetching from S3 (bucket=%s, key=%s, format=%s)",
            fetch_bucket,
            fetch_key,
            intents_format,
        )
        response = s3.get_object(Bucket=fetch_bucket, Key=fetch_key)
        raw_content = response["Body"].read().decode("utf-8")

    # SDG ran: read the raw artifact
    elif os.path.getsize(sdg_dataset.path) > 0:
        with open(sdg_dataset.path) as f:
            raw_content = f.read()
        log.info("Read raw SDG output from artifact (%d bytes)", len(raw_content))
    else:
        log.warning("No SDG output and no intents_s3_key — writing empty marker")
        with open(prompts_dataset.path, "w") as f:
            f.write("")
        return

    # Upload raw to S3
    raw_key = f"{s3_prefix}/sdg_raw_output.csv"
    s3.put_object(Bucket=s3_bucket, Key=raw_key, Body=raw_content.encode("utf-8"))
    log.info("Uploaded raw output to s3://%s/%s", s3_bucket, raw_key)

    # Normalise via core function
    fmt = intents_format if (intents_s3_key and intents_s3_key.strip()) else "csv"
    normalized = normalize_prompts(raw_content, format=fmt)
    normalized.to_csv(prompts_dataset.path, index=False)

    # Upload normalised to S3
    norm_key = f"{s3_prefix}/sdg_normalized_output.csv"
    s3.upload_file(prompts_dataset.path, s3_bucket, norm_key)
    log.info("Uploaded normalised output to s3://%s/%s", s3_bucket, norm_key)


@dsl.component(
    install_kfp_package=False,
    packages_to_install=[],
    base_image=_resolve_base_image(),
)
def garak_scan(
    config_json: str,
    s3_prefix: str,
    timeout_seconds: int,
    max_retry_seconds: int,
    hf_cache_path: str,
    prompts_dataset: dsl.Input[dsl.Dataset],
) -> NamedTuple("Outputs", [("success", bool), ("return_code", int)]):
    """Run a Garak scan and upload output to S3 via Data Connection credentials.

    If the ``prompts_dataset`` artifact contains data, intent stubs are
    generated from it before launching garak.  S3 credentials and model
    API keys are injected as environment variables by the pipeline using
    ``kubernetes.use_secret_as_env``.
    """
    import json
    import logging
    import os
    import tempfile
    from collections import namedtuple
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("evalhub_garak_scan")

    from llama_stack_provider_trustyai_garak.core.pipeline_steps import (
        setup_and_run_garak,
        redact_api_keys,
    )
    from llama_stack_provider_trustyai_garak.errors import GarakError

    if hf_cache_path and hf_cache_path.strip():
        from llama_stack_provider_trustyai_garak.evalhub.s3_utils import create_s3_client

        if hf_cache_path.startswith("s3://"):
            parts = hf_cache_path[len("s3://") :].split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
        else:
            bucket = os.environ.get("AWS_S3_BUCKET", "")
            prefix = hf_cache_path.lstrip("/")

        if not bucket:
            raise GarakError(
                "Cannot determine S3 bucket for HF cache. "
                "Provide a full s3://bucket/prefix URI in hf_cache_path, "
                "or set AWS_S3_BUCKET."
            )

        if not prefix:
            log.warning(
                "hf_cache_path has no sub-prefix; downloading all objects from bucket '%s'.",
                bucket,
            )

        hf_cache_dir = Path(tempfile.mkdtemp(prefix="hf-cache-"))
        s3 = create_s3_client()
        downloaded = 0

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(prefix) :].lstrip("/")
                if not rel:
                    continue
                dest = hf_cache_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, obj["Key"], str(dest))
                downloaded += 1

        os.environ["HF_HUB_CACHE"] = str(hf_cache_dir)
        log.info(
            "Populated HF cache from s3://%s/%s -> %s (%d files)",
            bucket,
            prefix,
            hf_cache_dir,
            downloaded,
        )

    scan_dir = Path(tempfile.mkdtemp(prefix="garak-scan-"))

    prompts_path = Path(prompts_dataset.path)
    has_prompts = prompts_path.exists() and prompts_path.stat().st_size > 0

    # This body is compiled from the adapter's package but runs against the package installed in
    # the base image, which can be older (e.g. a pinned or operator-configured image). Only pass
    # arguments the installed setup_and_run_garak accepts, so version skew can't fail every scan.
    import inspect

    scan_kwargs = {}
    if "max_retry_seconds" in inspect.signature(setup_and_run_garak).parameters:
        scan_kwargs["max_retry_seconds"] = max_retry_seconds
    else:
        log.warning("The base image's adapter predates max_retry_seconds; scanning without a retry budget")

    result = setup_and_run_garak(
        config_json=config_json,
        prompts_csv_path=prompts_path if has_prompts else None,
        scan_dir=scan_dir,
        timeout_seconds=timeout_seconds,
        **scan_kwargs,
    )

    # Redact api_key values from config.json before uploading to S3
    config_file = scan_dir / "config.json"
    if config_file.exists():
        try:
            cfg = json.loads(config_file.read_text())
            config_file.write_text(json.dumps(redact_api_keys(cfg), indent=1))
            log.info("Redacted api_key values from config.json before S3 upload")
        except Exception as exc:
            log.warning("Could not redact config.json: %s", exc)

    # Upload results to S3
    s3_bucket = os.environ.get("AWS_S3_BUCKET", "")
    if s3_bucket and s3_prefix:
        try:
            from llama_stack_provider_trustyai_garak.evalhub.s3_utils import create_s3_client

            s3 = create_s3_client()
            uploaded = 0
            for file_path in scan_dir.rglob("*"):
                if file_path.is_file():
                    key = f"{s3_prefix}/{file_path.relative_to(scan_dir)}"
                    s3.upload_file(str(file_path), s3_bucket, key)
                    uploaded += 1
            log.info("Uploaded %d files to s3://%s/%s", uploaded, s3_bucket, s3_prefix)
        except Exception as e:
            raise GarakError(f"Failed to upload results to S3: {e}") from e
    else:
        raise GarakError(f"S3 not configured (bucket={s3_bucket}, prefix={s3_prefix}); results not uploaded")

    Outputs = namedtuple("Outputs", ["success", "return_code"])
    return Outputs(success=result.success, return_code=result.returncode)


@dsl.component(
    install_kfp_package=False,
    packages_to_install=[],
    base_image=_resolve_base_image(),
)
def write_kfp_outputs(
    s3_prefix: str,
    eval_threshold: float,
    art_intents: bool,
    summary_metrics: dsl.Output[dsl.Metrics],
    html_report: dsl.Output[dsl.HTML],
):
    """Parse scan results from S3, log KFP metrics, and write an HTML report.

    This component validates the report before KFP marks the pipeline as successful.
    The adapter pod performs the same parse after it downloads the artifacts.
    """
    import logging
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("write_kfp_outputs")

    try:
        from llama_stack_provider_trustyai_garak.evalhub.s3_utils import create_s3_client
        from llama_stack_provider_trustyai_garak.core.pipeline_steps import (
            parse_and_build_results,
            log_kfp_metrics,
        )
        from llama_stack_provider_trustyai_garak.result_utils import generate_art_report

        s3_bucket = os.environ.get("AWS_S3_BUCKET", "")
        if not s3_bucket:
            raise RuntimeError("AWS_S3_BUCKET is not set for KFP output generation")

        s3 = create_s3_client()

        def _download_text(key: str) -> str:
            try:
                resp = s3.get_object(Bucket=s3_bucket, Key=key)
                return resp["Body"].read().decode("utf-8")
            except Exception:
                return ""

        report_content = _download_text(f"{s3_prefix}/scan.report.jsonl")
        if not report_content.strip():
            raise RuntimeError("The Garak report file is empty or absent")

        avid_content = _download_text(f"{s3_prefix}/scan.avid.jsonl")

        # Parse via core function
        combined = parse_and_build_results(
            report_content=report_content,
            avid_content=avid_content,
            art_intents=art_intents,
            eval_threshold=eval_threshold,
        )

        # Log metrics via core function
        log_kfp_metrics(summary_metrics, combined, art_intents)

        # Generate HTML report
        html_content = None
        if art_intents:
            try:
                html_content = generate_art_report(report_content, eval_threshold=eval_threshold)
                log.info("Generated ART intents HTML report")
            except Exception as exc:
                log.warning("Failed to generate ART HTML report: %s", exc)
        else:
            html_content = _download_text(f"{s3_prefix}/scan.report.html")

        if html_content:
            with open(html_report.path, "w") as f:
                f.write(html_content)

            if art_intents and s3_prefix:
                try:
                    s3.upload_file(
                        html_report.path,
                        s3_bucket,
                        f"{s3_prefix}/scan.intents.html",
                    )
                    log.info("Uploaded intents HTML to S3")
                except Exception as exc:
                    log.warning("Failed to upload intents HTML to S3: %s", exc)
        else:
            with open(html_report.path, "w") as f:
                f.write("<html><body><p>No HTML report available.</p></body></html>")

    except Exception as exc:
        log.error("KFP output generation failed: %s", exc)
        with open(html_report.path, "w") as f:
            f.write(f"<html><body><p>Report generation failed: {exc}</p></body></html>")
        raise


# ---------------------------------------------------------------------------
# KFP Pipeline
# ---------------------------------------------------------------------------


@dsl.pipeline(name="evalhub-garak-scan")
def evalhub_garak_pipeline(
    config_json: str,
    s3_prefix: str,
    timeout_seconds: int,
    s3_secret_name: str,
    model_auth_secret_name: str = "model-auth",
    eval_threshold: float = 0.5,
    art_intents: bool = False,
    policy_s3_key: str = "",
    policy_format: str = "csv",
    intents_s3_key: str = "",
    intents_format: str = "csv",
    sdg_model: str = "",
    sdg_api_base: str = "",
    sdg_provider: str = "",
    sdg_flow_id: str = DEFAULT_SDG_FLOW_ID,
    sdg_max_concurrency: int = DEFAULT_SDG_MAX_CONCURRENCY,
    sdg_num_samples: int = DEFAULT_SDG_NUM_SAMPLES,
    sdg_max_tokens: int = DEFAULT_SDG_MAX_TOKENS,
    hf_cache_path: str = "",
    max_retry_seconds: int = DEFAULT_MAX_RETRY_SECONDS,
):
    """Six-step pipeline: validate, resolve taxonomy, SDG, prepare prompts, scan, write outputs.

    ``s3_secret_name`` is required -- the Data Connection secret is injected
    as environment variables into all pipeline pods.

    ``model_auth_secret_name`` (optional) -- a Kubernetes Secret mounted
    as a volume at ``/mnt/model-auth`` (with ``optional=True`` so pods
    start even if the secret is missing).  Each key in the secret becomes
    a file that ``resolve_api_key`` reads at runtime.  Resolution order
    per role: ``{ROLE}_API_KEY`` env -> ``{ROLE}_API_KEY`` file ->
    ``API_KEY`` env -> ``API_KEY`` file -> ``"DUMMY"``.

    * ``API_KEY`` -- generic fallback, sufficient for most setups
    * ``TARGET_API_KEY``, ``JUDGE_API_KEY``, ``EVALUATOR_API_KEY``,
      ``ATTACKER_API_KEY``, ``SDG_API_KEY`` -- optional per-role overrides

    Three intents modes:
    - Default taxonomy + SDG: no file params, requires ``sdg_model``/``sdg_api_base``.
    - Custom taxonomy + SDG: ``policy_s3_key`` set, requires ``sdg_model``/``sdg_api_base``.
    - Bypass SDG: ``intents_s3_key`` set, SDG params NOT required.

    ``policy_s3_key`` and ``intents_s3_key`` are mutually exclusive.
    """
    # Step 1: Pre-flight validation
    validate_task = validate(
        config_json=config_json,
    )
    validate_task.set_caching_options(False)

    kubernetes.use_secret_as_env(
        validate_task,
        secret_name=s3_secret_name,
        secret_key_to_env=S3_DATA_CONNECTION_KEYS,
    )

    # Step 2: Resolve taxonomy (custom from S3 or BASE_TAXONOMY)
    taxonomy_task = resolve_taxonomy(
        policy_s3_key=policy_s3_key,
        policy_format=policy_format,
    )
    taxonomy_task.set_caching_options(True)
    taxonomy_task.after(validate_task)

    kubernetes.use_secret_as_env(
        taxonomy_task,
        secret_name=s3_secret_name,
        secret_key_to_env=S3_DATA_CONNECTION_KEYS,
    )

    # Step 3: Run SDG (or no-op for non-intents / bypass)
    sdg_task = sdg_generate(
        art_intents=art_intents,
        intents_s3_key=intents_s3_key,
        sdg_model=sdg_model,
        sdg_api_base=sdg_api_base,
        sdg_provider=sdg_provider,
        sdg_flow_id=sdg_flow_id,
        sdg_max_concurrency=sdg_max_concurrency,
        sdg_num_samples=sdg_num_samples,
        sdg_max_tokens=sdg_max_tokens,
        taxonomy_dataset=taxonomy_task.outputs["taxonomy_dataset"],
    )
    sdg_task.set_caching_options(True)

    kubernetes.use_secret_as_env(
        sdg_task,
        secret_name=s3_secret_name,
        secret_key_to_env=S3_DATA_CONNECTION_KEYS,
    )
    kubernetes.use_secret_as_volume(
        sdg_task,
        secret_name=model_auth_secret_name,
        mount_path=MODEL_AUTH_MOUNT_PATH,
        optional=True,
    )

    # Step 4: Prepare prompts (normalise, persist raw+normalised to S3)
    prep_task = prepare_prompts(
        art_intents=art_intents,
        s3_prefix=s3_prefix,
        intents_s3_key=intents_s3_key,
        intents_format=intents_format,
        sdg_dataset=sdg_task.outputs["sdg_dataset"],
    )
    prep_task.set_caching_options(False)

    kubernetes.use_secret_as_env(
        prep_task,
        secret_name=s3_secret_name,
        secret_key_to_env=S3_DATA_CONNECTION_KEYS,
    )

    # Step 5: Run garak scan
    scan_task = garak_scan(
        config_json=config_json,
        s3_prefix=s3_prefix,
        timeout_seconds=timeout_seconds,
        max_retry_seconds=max_retry_seconds,
        hf_cache_path=hf_cache_path,
        prompts_dataset=prep_task.outputs["prompts_dataset"],
    )
    scan_task.set_caching_options(False)
    scan_task.after(prep_task)

    kubernetes.use_secret_as_env(
        scan_task,
        secret_name=s3_secret_name,
        secret_key_to_env=S3_DATA_CONNECTION_KEYS,
    )
    kubernetes.use_secret_as_volume(
        scan_task,
        secret_name=model_auth_secret_name,
        mount_path=MODEL_AUTH_MOUNT_PATH,
        optional=True,
    )

    # Step 6: Write KFP Metrics + HTML to dashboard, upload intents HTML to S3
    outputs_task = write_kfp_outputs(
        s3_prefix=s3_prefix,
        eval_threshold=eval_threshold,
        art_intents=art_intents,
    )
    outputs_task.set_caching_options(False)
    outputs_task.after(scan_task)

    kubernetes.use_secret_as_env(
        outputs_task,
        secret_name=s3_secret_name,
        secret_key_to_env=S3_DATA_CONNECTION_KEYS,
    )
