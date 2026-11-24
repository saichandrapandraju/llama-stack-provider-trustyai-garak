"""Shared pipeline step logic for both EvalHub KFP and Llama Stack KFP pipelines.

These functions contain framework-agnostic business logic without I/O adapter
dependencies (no S3, no Files API).  They operate on local file paths, content
strings, and DataFrames.

Each KFP adapter (EvalHub/S3, Llama Stack/Files API) calls these functions
and handles its own I/O.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from ..constants import DEFAULT_MAX_RETRY_SECONDS, DEFAULT_SDG_FLOW_ID
from ..errors import GarakError, GarakValidationError

from .garak_runner import GarakScanResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API key resolution & redaction
# ---------------------------------------------------------------------------
#
# Model Authentication
# ====================
#
# API keys are provided via a Kubernetes Secret referenced by
# ``model.auth.secret_ref`` in the EvalHub job spec.
#
# No auth needed (local/in-cluster models)
# -----------------------------------------
# Do nothing. No secret, no config. Keys default to "DUMMY".
#
# Same key for all models
# -----------------------
# Create a secret with a single ``api-key`` entry::
#
#     oc create secret generic model-auth \
#       --from-literal=api-key=sk-your-key
#
# Then reference it in the job spec::
#
#     "model": { "auth": { "secret_ref": "model-auth" } }
#
# The ``api-key`` entry is used for all models in both simple and KFP modes.
# If ``model.auth.secret_ref`` is omitted, the pipeline defaults to looking
# for a secret named ``model-auth`` (with ``optional=True`` so pods run
# even if it doesn't exist).
#
# Different keys per model role (KFP intents only)
# -------------------------------------------------
# Add role-specific keys to the same secret::
#
#     oc create secret generic model-auth \
#       --from-literal=api-key=sk-target-key \
#       --from-literal=SDG_API_KEY=sk-sdg-key \
#       --from-literal=JUDGE_API_KEY=sk-judge-key
#
# Secret key reference:
#
#   ``api-key``           Target model (all modes). Required if model needs auth.
#   ``SDG_API_KEY``       SDG prompt generation.    Only if different from api-key.
#   ``JUDGE_API_KEY``     Judge model.              Only if different from api-key.
#   ``ATTACKER_API_KEY``  Attacker model.           Only if different from api-key.
#   ``EVALUATOR_API_KEY`` Evaluator model.          Only if different from api-key.
#
# Role-specific keys take precedence over ``api-key``. If a role key is
# missing, ``api-key`` is used as the fallback.
#
# How it works
# ------------
# - **Simple mode**: The EvalHub service mounts the secret at
#   ``/var/run/secrets/model/``. ``resolve_api_key(role)`` checks both
#   the KFP path and the evalhub SDK path via ``read_model_auth_key()``,
#   so role-specific keys (e.g. ``JUDGE_API_KEY``) are resolved from
#   whichever mount point exists.
# - **KFP mode**: The pipeline mounts the secret at ``/mnt/model-auth``
#   via ``use_secret_as_volume(optional=True)``. ``resolve_api_key(role)``
#   reads files from that path.
# - ``config.json`` is always **redacted** (``api_key: "***"``) before
#   upload to S3, OCI, or Files API.
# ---------------------------------------------------------------------------

MODEL_AUTH_MOUNT_PATH = "/mnt/model-auth"


def _read_secret_file(name: str) -> str:
    """Read a single key from the volume-mounted model auth secret.

    Returns the file content stripped of whitespace, or empty string if
    the file does not exist.
    """
    path = Path(MODEL_AUTH_MOUNT_PATH) / name
    try:
        return path.read_text().strip() if path.is_file() else ""
    except OSError:
        return ""


def _read_evalhub_secret(name: str) -> str:
    """Read a key from the evalhub model auth secret (simple mode).

    In simple (non-KFP) mode, the eval-hub service mounts the model auth
    secret at ``/var/run/secrets/model/``.  This function delegates to the
    evalhub SDK's ``read_model_auth_key`` which reads from that path.

    Returns the value or empty string if unavailable.
    """
    try:
        from evalhub.adapter.auth import read_model_auth_key

        return read_model_auth_key(name) or ""
    except ImportError:
        return ""


def resolve_api_key(role: str) -> str:
    """Resolve an API key for a model role.

    Keys may come from environment variables, a Kubernetes Secret mounted
    at ``/mnt/model-auth`` (KFP mode), or the evalhub model auth secret
    at ``/var/run/secrets/model/`` (simple mode).

    Both uppercase (``JUDGE_API_KEY``), lowercase hyphenated
    (``judge-api-key``), and EvalHub multi-model convention
    (``judge_api-key``) key names are checked.

    Resolution order (first non-empty wins):

    1.  ``{ROLE}_API_KEY`` env var  (e.g. ``SDG_API_KEY``)
    2.  ``{ROLE}_API_KEY`` KFP secret file (``/mnt/model-auth/``)
    3.  ``{ROLE}_API_KEY`` evalhub secret (``/var/run/secrets/model/``)
    4.  ``{role}-api-key`` KFP secret file (lowercase, K8s convention)
    5.  ``{role}-api-key`` evalhub secret
    6.  ``{role}_api-key`` KFP secret file (EvalHub multi-model convention)
    7.  ``{role}_api-key`` evalhub secret
    8.  ``API_KEY`` env var          (generic fallback)
    9.  ``API_KEY`` KFP secret file
    10. ``API_KEY`` evalhub secret
    11. ``api-key`` KFP secret file  (EvalHub convention)
    12. ``api-key`` evalhub secret   (EvalHub convention)
    13. ``"DUMMY"``                  (unauthenticated / local endpoints)
    """
    role_upper = role.upper()
    role_var = f"{role_upper}_API_KEY"
    role_lower = f"{role.lower()}-api-key"
    role_evalhub = f"{role.lower()}_api-key"
    return (
        os.environ.get(role_var)
        or _read_secret_file(role_var)
        or _read_evalhub_secret(role_var)
        or _read_secret_file(role_lower)
        or _read_evalhub_secret(role_lower)
        or _read_secret_file(role_evalhub)
        or _read_evalhub_secret(role_evalhub)
        or os.environ.get("API_KEY")
        or _read_secret_file("API_KEY")
        or _read_evalhub_secret("API_KEY")
        or _read_secret_file("api-key")
        or _read_evalhub_secret("api-key")
        or "DUMMY"
    )


def redact_api_keys(config: dict) -> dict:
    """Return a deep copy of *config* with all ``api_key`` values replaced by ``***``.

    Suitable for sanitising the config before uploading to S3 / Files API
    so that secrets are never persisted in object storage.
    """
    import copy

    redacted = copy.deepcopy(config)
    _redact_recursive(redacted)
    return redacted


def _redact_recursive(obj: Any) -> None:
    """Walk a nested dict/list and replace any ``api_key`` values."""
    if isinstance(obj, dict):
        for key in obj:
            if key.lower() == "api_key" and isinstance(obj[key], str):
                obj[key] = "***"
            else:
                _redact_recursive(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            _redact_recursive(item)


def _resolve_config_api_keys(config: dict) -> None:
    """Replace ``api_key`` placeholders in the garak config with real values.

    Walks the config and replaces any ``api_key`` that is a known
    placeholder (empty, ``"DUMMY"``, ``"__FROM_ENV__"``, ``"***"``) with the
    value from :func:`resolve_api_key` for the appropriate role.
    Comparison is case-insensitive so ``"dummy"``, ``"Dummy"``, ``"DUMMY"``
    are all treated as placeholders.
    Non-placeholder keys (real values already set) are left untouched.
    """
    _PLACEHOLDERS = {"", "DUMMY", "__FROM_ENV__", "***"}

    def _is_placeholder(value: str) -> bool:
        stripped = value.strip()
        if stripped.endswith(":ref"):
            return False
        return stripped.upper() in _PLACEHOLDERS

    _ROLE_URL_KEYS = {
        "JUDGE": ["judge_url"],
        "ATTACKER": ["attacker_url", "judge_url"],
        "EVALUATOR": ["evaluator_url", "judge_url"],
        "TRANSLATION": ["translation_url", "attacker_url", "judge_url"],
    }

    def _is_sidecar_uri(value: str) -> bool:
        return not value or "localhost" in value or "127.0.0.1" in value

    def _resolve_uri(role: str, current_uri: str) -> str:
        """Resolve URI from mounted secret when current value is a sidecar address."""
        if not _is_sidecar_uri(current_uri):
            return current_uri
        secret_keys = _ROLE_URL_KEYS.get(role, [])
        for secret_key in secret_keys:
            real_url = _read_secret_file(secret_key) or _read_evalhub_secret(secret_key)
            if real_url:
                return real_url.strip().rstrip("/")
        return current_uri

    _ROLE_KEY_FALLBACKS = {
        "ATTACKER": ["JUDGE"],
        "EVALUATOR": ["JUDGE"],
        "TRANSLATION": ["ATTACKER", "JUDGE"],
    }

    def _has_role_specific_key(role: str) -> str:
        """Check only role-specific keys (not generic fallbacks)."""
        role_lower = role.lower()
        return (
            os.environ.get(f"{role}_API_KEY")
            or _read_secret_file(f"{role}_API_KEY")
            or _read_secret_file(f"{role_lower}-api-key")
            or _read_secret_file(f"{role_lower}_api-key")
            or _read_evalhub_secret(f"{role}_API_KEY")
            or _read_evalhub_secret(f"{role_lower}-api-key")
            or _read_evalhub_secret(f"{role_lower}_api-key")
            or ""
        )

    def _resolve_key_with_fallback(role: str) -> str:
        """Resolve API key for role, falling back to related roles before generic."""
        own_key = _has_role_specific_key(role)
        if own_key:
            return own_key
        for fallback_role in _ROLE_KEY_FALLBACKS.get(role, []):
            fallback_key = _has_role_specific_key(fallback_role)
            if fallback_key:
                return fallback_key
        return resolve_api_key(role)

    def _walk(obj: Any, role: str = "TARGET") -> None:
        if isinstance(obj, dict):
            if "api_key" in obj and isinstance(obj["api_key"], str):
                if _is_placeholder(obj["api_key"]):
                    obj["api_key"] = _resolve_key_with_fallback(role)
            if "uri" in obj and isinstance(obj["uri"], str):
                obj["uri"] = _resolve_uri(role, obj["uri"])
            for key, val in obj.items():
                child_role = role
                if key == "detector_model_config":
                    child_role = "JUDGE"
                elif key == "attack_model_config":
                    child_role = "ATTACKER"
                elif key == "evaluator_model_config":
                    child_role = "EVALUATOR"
                elif key == "langproviders" and isinstance(val, list):
                    for entry in val:
                        _walk(entry, "TRANSLATION")
                    continue
                _walk(val, child_role)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item, role)

    _walk(config, "TARGET")


# ---------------------------------------------------------------------------
# Translation langprovider configuration
# ---------------------------------------------------------------------------

_HF_LANGPROVIDERS = [
    {
        "language": "zh,en",
        "model_type": "local.LocalHFTranslator",
        "model_name": "Helsinki-NLP/opus-mt-zh-en",
    },
    {
        "language": "en,zh",
        "model_type": "local.LocalHFTranslator",
        "model_name": "Helsinki-NLP/opus-mt-en-zh",
    },
]


def _build_llm_langproviders(url: str, name: str, api_key: str = "__FROM_ENV__") -> list[dict[str, Any]]:
    """Build ``llm`` langprovider entries for zh/en pair."""
    from ..constants import DEFAULT_MODEL_TYPE

    return [
        {
            "language": "zh,en",
            "model_type": "llm",
            "translation_model_type": DEFAULT_MODEL_TYPE,
            "translation_model_name": name,
            "translation_model_config": {"uri": url, "api_key": api_key, "max_tokens": 4096, "temperature": 0.2},
        },
        {
            "language": "en,zh",
            "model_type": "llm",
            "translation_model_type": DEFAULT_MODEL_TYPE,
            "translation_model_name": name,
            "translation_model_config": {"uri": url, "api_key": api_key, "max_tokens": 4096, "temperature": 0.2},
        },
    ]


_TRANSLATION_PROBE = "probes.multilingual.TranslationIntent"


def _probe_spec_includes_translation(probe_spec: str) -> bool:
    """Return True if *probe_spec* contains the canonical TranslationIntent selector."""
    return _TRANSLATION_PROBE in {selector.strip() for selector in probe_spec.split(",")}


def build_translation_langproviders(
    benchmark_config: dict[str, Any],
    attacker_url: str = "",
    attacker_name: str = "",
    attacker_api_key: str = "",
    probe_spec: str = "",
    model_url: str = "",
) -> list[dict[str, str]] | None:
    """Resolve langproviders for the ``multilingual.TranslationIntent`` probe.

    Returns ``None`` when *probe_spec* does not include
    ``TranslationIntent``, so the caller can skip setting langproviders
    entirely (other probes don't need them).

    Resolution order (when TranslationIntent **is** present):

    1. ``translation_use_hf=True`` in *benchmark_config* -- use HuggingFace
       ``local.LocalHFTranslator`` models (Helsinki-NLP).
    2. ``translation_url`` / ``translation_api-key`` from the evalhub model
       auth secret (sidecar multi-model convention).
    3. ``intents_models.translation`` has ``url`` + ``name`` -- use a
       dedicated LLM endpoint via ``llm.LLMTranslator``.
    4. *attacker_url* / *attacker_name* available (from the resolved
       attacker model) -- reuse the attacker LLM for translation.
    5. Fallback to HF models (safety net when no LLM is available).

    API keys for LLM translators use the ``__FROM_ENV__`` placeholder
    and are resolved at pod level by :func:`_resolve_config_api_keys`
    with role ``TRANSLATION``, unless a ref token is read from the secret.
    """
    if probe_spec and not _probe_spec_includes_translation(probe_spec):
        logger.info("TranslationIntent not in probe_spec — skipping langproviders")
        return None

    from ..utils import as_bool

    if as_bool(benchmark_config.get("translation_use_hf", False)):
        logger.info("Translation mode: HF (translation_use_hf=True)")
        return list(_HF_LANGPROVIDERS)

    try:
        from evalhub.adapter.auth import read_model_auth_key

        trans_secret_url = read_model_auth_key("translation_url")
        trans_secret_key = read_model_auth_key("translation_api-key")
        if trans_secret_url or trans_secret_key:
            effective_url = trans_secret_url or model_url
            effective_key = trans_secret_key or read_model_auth_key("api-key") or "__FROM_ENV__"
            intents_models = benchmark_config.get("intents_models", {})
            if not isinstance(intents_models, dict):
                intents_models = {}
            translation_cfg = intents_models.get("translation") or {}
            trans_name = translation_cfg.get("name") or "translation-model"
            logger.info("Translation mode: secret-based LLM (%s)", trans_name)
            return _build_llm_langproviders(effective_url, trans_name, api_key=effective_key)
    except ImportError:
        pass

    intents_models = benchmark_config.get("intents_models", {})
    if not isinstance(intents_models, dict):
        intents_models = {}
    translation_cfg = intents_models.get("translation") or {}

    if translation_cfg.get("url") and translation_cfg.get("name"):
        logger.info("Translation mode: dedicated LLM (%s)", translation_cfg["name"])
        return _build_llm_langproviders(translation_cfg["url"], translation_cfg["name"])

    if attacker_url and attacker_name:
        api_key = attacker_api_key or "__FROM_ENV__"
        logger.info("Translation mode: attacker LLM (%s)", attacker_name)
        return _build_llm_langproviders(attacker_url, attacker_name, api_key=api_key)

    logger.info("Translation mode: HF fallback (no LLM available)")
    return list(_HF_LANGPROVIDERS)


# ---------------------------------------------------------------------------
# Step 1: Validation
# ---------------------------------------------------------------------------

_DANGEROUS_FLAGS = ("--rm", "--force", "--no-limit")


def validate_scan_config(config_json: str) -> None:
    """Validate config JSON structure, garak importability, and flag safety.

    Raises GarakValidationError on any failure.  Adapter-specific
    connectivity checks (S3, Files API) are done by the caller.
    """
    errors: list[str] = []

    try:
        import garak  # noqa: F401
    except ImportError:
        errors.append("garak is not installed in the base image")

    try:
        config = json.loads(config_json)
        if "plugins" not in config:
            errors.append("config_json missing required 'plugins' section")
    except json.JSONDecodeError as exc:
        errors.append(f"config_json is not valid JSON: {exc}")

    for flag in _DANGEROUS_FLAGS:
        if flag in config_json:
            errors.append(f"Dangerous flag detected: {flag}")

    if errors:
        raise GarakValidationError("Pre-flight validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


# ---------------------------------------------------------------------------
# Step 2: Taxonomy resolution
# ---------------------------------------------------------------------------


def resolve_taxonomy_data(
    content: bytes | None,
    format: str = "csv",
) -> pd.DataFrame:
    """Parse raw taxonomy bytes or return the built-in BASE_TAXONOMY.

    The caller is responsible for fetching the bytes from its storage
    backend (S3, Files API, local FS) before calling this function.
    """
    if content is not None:
        from ..intents import load_taxonomy_dataset

        text = content.decode("utf-8") if isinstance(content, bytes) else content
        taxonomy = load_taxonomy_dataset(content=text, format=format)
        logger.info(
            "Loaded custom taxonomy: %d entries, columns: %s",
            len(taxonomy),
            list(taxonomy.columns),
        )
        return taxonomy

    from ..sdg import BASE_TAXONOMY

    taxonomy = pd.DataFrame(BASE_TAXONOMY)
    logger.info("Using BASE_TAXONOMY: %d entries", len(taxonomy))
    return taxonomy


# ---------------------------------------------------------------------------
# Step 3: SDG generation
# ---------------------------------------------------------------------------


def run_sdg_generation(
    taxonomy_df: pd.DataFrame,
    sdg_model: str,
    sdg_api_base: str,
    sdg_flow_id: str = "",
    sdg_max_concurrency: int = 0,
    sdg_num_samples: int = 0,
    sdg_max_tokens: int = 0,
    api_key: str | None = None,
    sdg_provider: str = "",
) -> pd.DataFrame:
    """Run Synthetic Data Generation on a taxonomy.  Returns the raw DataFrame.

    When *api_key* is provided (e.g. a sidecar ref token), it is used
    directly.  Otherwise the key is resolved via ``resolve_api_key("sdg")``
    which reads ``SDG_API_KEY`` or ``API_KEY`` environment variables
    (injected from a Kubernetes Secret in KFP mode), falling back to
    ``"DUMMY"``.
    """
    from ..sdg import generate_sdg_dataset

    if not sdg_model or not sdg_model.strip():
        raise GarakValidationError("sdg_model is required for intents scans when SDG must run.")

    # Resolve SDG URL from mounted secret if sidecar address or empty
    effective_base = sdg_api_base
    if not effective_base or "localhost" in effective_base or "127.0.0.1" in effective_base:
        secret_url = _read_secret_file("sdg_url") or _read_evalhub_secret("sdg_url")
        if secret_url:
            effective_base = secret_url.strip().rstrip("/")
            logger.info("Resolved sdg_api_base from secret: %s", effective_base)
    if not effective_base or not effective_base.strip():
        raise GarakValidationError("sdg_api_base is required for intents scans when SDG must run.")
    sdg_api_base = effective_base

    effective_key = api_key or resolve_api_key("sdg")

    effective_flow_id = (sdg_flow_id.strip() if sdg_flow_id else "") or DEFAULT_SDG_FLOW_ID

    logger.info(
        "Running SDG: model=%s, api_base=%s, flow=%s",
        sdg_model,
        sdg_api_base,
        effective_flow_id,
    )
    sdg_result = generate_sdg_dataset(
        model=sdg_model,
        api_base=sdg_api_base,
        flow_id=effective_flow_id,
        api_key=effective_key,
        taxonomy=taxonomy_df,
        max_concurrency=sdg_max_concurrency,
        num_samples=sdg_num_samples,
        max_tokens=sdg_max_tokens,
        provider=sdg_provider,
    )
    logger.info(
        "SDG produced %d raw rows across %d categories",
        len(sdg_result.raw),
        sdg_result.raw["policy_concept"].nunique(),
    )
    return sdg_result.raw


# ---------------------------------------------------------------------------
# Step 4: Prompt normalisation
# ---------------------------------------------------------------------------


def normalize_prompts(
    raw_content: str,
    format: str = "csv",
) -> pd.DataFrame:
    """Normalise raw intents data into the (category, prompt, description) schema."""
    from ..intents import load_intents_dataset

    normalized = load_intents_dataset(content=raw_content, format=format)
    logger.info(
        "Normalised %d prompts across %d categories",
        len(normalized),
        normalized["category"].nunique(),
    )
    return normalized


# ---------------------------------------------------------------------------
# Step 5: Garak scan execution
# ---------------------------------------------------------------------------


def setup_and_run_garak(
    config_json: str,
    prompts_csv_path: Path | str | None,
    scan_dir: Path,
    timeout_seconds: int,
    max_retry_seconds: int = DEFAULT_MAX_RETRY_SECONDS,
) -> "GarakScanResult":
    """Configure and run a Garak scan with optional intent-stub generation.

    1. Parse *config_json*, set ``report_prefix`` inside *scan_dir*.
    2. If *prompts_csv_path* points to a non-empty CSV, generate intent stubs.
    3. Run ``garak --config ...`` via :func:`core.garak_runner.run_garak_scan`.
    4. Attempt AVID conversion on success.

    Returns a :class:`GarakScanResult`.  Raises :class:`GarakError` on failure.
    """
    from .garak_runner import convert_to_avid_report, run_garak_scan

    scan_dir.mkdir(parents=True, exist_ok=True)

    config: dict = json.loads(config_json)
    config.setdefault("reporting", {})["report_prefix"] = str(scan_dir / "scan")

    _resolve_config_api_keys(config)

    config_file = scan_dir / "config.json"
    config_file.write_text(json.dumps(config, indent=1))

    report_prefix = scan_dir / "scan"
    log_file = scan_dir / "scan.log"

    if prompts_csv_path is not None:
        prompts_csv_path = Path(prompts_csv_path)
        if prompts_csv_path.exists() and prompts_csv_path.stat().st_size > 0:
            from ..intents import generate_intents_from_dataset

            df = pd.read_csv(prompts_csv_path)
            if not df.empty:
                desc_col = "description" if "description" in df.columns else None
                xdg_data_home = scan_dir / "xdg_data"
                generate_intents_from_dataset(
                    df,
                    category_description_column_name=desc_col,
                    xdg_data_home=xdg_data_home,
                )
                logger.info(
                    "Generated intent stubs for %d prompts across %d categories",
                    len(df),
                    df["category"].nunique(),
                )

    logger.info(
        "Starting garak scan (timeout=%ss, max_retry_seconds=%s, output=%s)",
        timeout_seconds,
        max_retry_seconds,
        scan_dir,
    )

    scan_env = {"XDG_DATA_HOME": str(scan_dir / "xdg_data")} if prompts_csv_path is not None else None
    result = run_garak_scan(
        config_file=config_file,
        timeout_seconds=timeout_seconds,
        report_prefix=report_prefix,
        log_file=log_file,
        env=scan_env,
        max_retry_seconds=max_retry_seconds,
    )

    if result.success:
        logger.info("Garak scan completed successfully (rc=%s)", result.returncode)
        if result.log_errors:
            logger.warning(
                "Garak scan succeeded but scan.log contains %d error/warning entries",
                len(result.log_errors),
            )
        try:
            convert_to_avid_report(result.report_jsonl)
        except Exception as exc:
            logger.warning("AVID conversion failed: %s", exc)
    else:
        error_msg = (
            f"Garak scan failed (rc={result.returncode}, "
            f"timed_out={result.timed_out}): "
            f"{result.retry_budget_reason or (result.stderr[:500] if result.stderr else 'no stderr')}"
        )
        if result.log_errors:
            error_msg += "\nscan.log errors:\n" + "\n".join(result.log_errors)
        logger.error(error_msg)
        raise GarakError(error_msg)

    return result


# ---------------------------------------------------------------------------
# Step 6: Result parsing & metrics
# ---------------------------------------------------------------------------


def parse_and_build_results(
    report_content: str,
    avid_content: str | None,
    art_intents: bool,
    eval_threshold: float,
    raw_entries_by_probe: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Parse Garak reports and build the combined result dictionary.

    Calls the result_utils parsing pipeline and returns a dict suitable
    for constructing an EvaluateResponse.
    """
    from .. import result_utils

    generations, score_rows_by_probe, parsed_raw = result_utils.parse_generations_from_report_content(
        report_content, eval_threshold, art_intents=art_intents
    )
    aggregated_by_probe = result_utils.parse_aggregated_from_avid_content(avid_content or "")
    digest = result_utils.parse_digest_from_report_content(report_content)
    harness_summary = result_utils.parse_harness_summary(report_content) if art_intents else None
    harness_stub_summaries = (
        (result_utils.parse_harness_stub_summaries(report_content) or None) if art_intents else None
    )

    effective_raw = raw_entries_by_probe if raw_entries_by_probe is not None else parsed_raw

    return result_utils.combine_parsed_results(
        generations,
        score_rows_by_probe,
        aggregated_by_probe,
        eval_threshold,
        digest,
        art_intents=art_intents,
        raw_entries_by_probe=effective_raw,
        harness_summary=harness_summary,
        harness_stub_summaries=harness_stub_summaries,
        allow_legacy_intent_inference=art_intents and "EarlyStopHarness" in report_content,
    )


def log_kfp_metrics(
    metrics_output: Any,
    result_dict: dict[str, Any],
    art_intents: bool,
) -> None:
    """Log summary metrics to a KFP Metrics artifact.

    Works with ``dsl.Output[dsl.Metrics]`` — call ``metrics_output.log_metric``.
    """
    overall = result_dict.get("scores", {}).get("_overall", {}).get("aggregated_results", {})

    if art_intents:
        metrics_output.log_metric(
            "attack_success_rate",
            overall.get("attack_success_rate", 0),
        )
    else:
        metrics_output.log_metric(
            "total_attempts",
            overall.get("total_attempts", 0),
        )
        metrics_output.log_metric(
            "vulnerable_responses",
            overall.get("vulnerable_responses", 0),
        )
        metrics_output.log_metric(
            "attack_success_rate",
            overall.get("attack_success_rate", 0),
        )

    if overall.get("tbsa"):
        metrics_output.log_metric("tbsa", overall["tbsa"])

    logger.info(
        "Logged KFP metrics: %s",
        {k: v for k, v in overall.items() if k in ("attack_success_rate", "total_attempts", "tbsa")},
    )
