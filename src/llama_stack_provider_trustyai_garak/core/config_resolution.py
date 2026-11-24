"""Helpers for resolving Garak benchmark and runtime configuration."""

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any, Mapping

from ..config import GarakScanConfig
from ..garak_command_config import GarakCommandConfig

logger = logging.getLogger(__name__)


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep merge two mappings where override values take precedence."""
    merged = deepcopy(dict(base))

    for key, value in dict(override).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def resolve_scan_profile(benchmark_id: str) -> dict[str, Any]:
    """Resolve a benchmark id to a predefined scan/framework profile."""
    config = GarakScanConfig()
    all_profiles = {**config.FRAMEWORK_PROFILES, **config.SCAN_PROFILES}
    resolved = all_profiles.get(benchmark_id) or all_profiles.get(f"trustyai_garak::{benchmark_id}") or {}
    if resolved and isinstance(resolved, GarakCommandConfig):
        resolved = resolved.to_dict(exclude_none=True)
    return deepcopy(resolved)


def _set_nested_value(target: dict[str, Any], path: tuple[str, str], value: Any) -> None:
    section, field = path
    target.setdefault(section, {})
    target[section][field] = deepcopy(value)


def _selector_values(value: Any) -> list[Any]:
    """Return selector values from a string, list, or tuple without reordering."""
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]

    result: list[Any] = []
    for item in values:
        if isinstance(item, str):
            result.extend(part.strip() for part in item.split(",") if part.strip())
        elif item is not None:
            result.append(item)
    return result


def _dedupe_selectors(selectors: list[Any]) -> list[Any]:
    """Remove exact duplicate selectors while preserving the first occurrence."""
    result: list[Any] = []
    for selector in selectors:
        if selector not in result:
            result.append(selector)
    return result


def _canonical_plugin_selectors(value: Any, category: str) -> list[Any]:
    """Convert legacy plugin selectors to the canonical ``category.*`` form."""
    selectors: list[Any] = []
    prefix = f"{category}."
    repeated_prefix = f"{prefix}{category}."
    for item in _selector_values(value):
        if isinstance(item, dict):
            selectors.append(deepcopy(item))
            continue
        selector = str(item).strip()
        if selector.lower() in ("", "auto"):
            continue
        if selector.lower() in ("all", "*"):
            selectors.append(f"{category}.*")
        elif selector.lower() == "none":
            selectors.append(f"{category}.none")
        else:
            while selector.startswith(repeated_prefix):
                selector = selector[len(prefix) :]
            selectors.append(selector if selector.startswith(prefix) else f"{prefix}{selector}")
    return _dedupe_selectors(selectors)


def _tag_selectors(value: Any) -> list[Any]:
    """Convert flat probe tags to run specification mappings."""
    selectors: list[Any] = []
    for item in _selector_values(value):
        if isinstance(item, dict):
            selectors.append(deepcopy(item))
        else:
            selectors.append({"tag": str(item)})
    return _dedupe_selectors(selectors)


def _intent_selectors(value: Any) -> tuple[list[Any], list[Any]]:
    """Convert a legacy intent specification to include and exclude selectors."""
    values = _selector_values(value)
    if not values:
        return [], [{"intent": "all"}]

    include: list[Any] = []
    for item in values:
        if isinstance(item, dict):
            include.append(deepcopy(item))
            continue
        intent = str(item).strip()
        include.append({"intent": "all" if intent in ("*", "all") else intent})
    return _dedupe_selectors(include), []


def _migrate_legacy_garak_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate removed Garak configuration fields before Pydantic validation."""
    migrated = deepcopy(dict(config))
    run = migrated.setdefault("run", {})
    plugins = migrated.setdefault("plugins", {})
    if not isinstance(run, dict):
        run = migrated["run"] = {}
    if not isinstance(plugins, dict):
        plugins = migrated["plugins"] = {}

    cas = migrated.pop("cas", None)
    if isinstance(cas, dict) and "serve_detectorless_intents" in cas:
        run.setdefault("serve_detectorless_intents", cas["serve_detectorless_intents"])

    if "spec" not in run:
        include: list[Any] = []
        exclude: list[Any] = []
        if "probe_spec" in plugins:
            include.extend(_canonical_plugin_selectors(plugins["probe_spec"], "probes"))
        if "buff_spec" in plugins:
            include.extend(_canonical_plugin_selectors(plugins["buff_spec"], "buffs"))
        if "probe_tags" in run:
            include.extend(_tag_selectors(run["probe_tags"]))
        if isinstance(cas, dict) and "intent_spec" in cas:
            intent_include, intent_exclude = _intent_selectors(cas["intent_spec"])
            include.extend(intent_include)
            exclude.extend(intent_exclude)
        if include or exclude or any(key in plugins for key in ("probe_spec", "buff_spec")) or "probe_tags" in run:
            run["spec"] = {"include": _dedupe_selectors(include), "exclude": _dedupe_selectors(exclude)}

    plugins.pop("probe_spec", None)
    plugins.pop("buff_spec", None)
    run.pop("probe_tags", None)
    return migrated


def legacy_overrides_from_benchmark_config(
    benchmark_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate flat EvalHub aliases to the Garak command configuration shape."""
    overrides: dict[str, Any] = {}
    key_to_path: dict[str, tuple[str, str]] = {
        "parallel_attempts": ("system", "parallel_attempts"),
        "parallel_requests": ("system", "parallel_requests"),
        "generations": ("run", "generations"),
        "soft_probe_prompt_cap": ("run", "soft_probe_prompt_cap"),
        "seed": ("run", "seed"),
        "deprefix": ("run", "deprefix"),
        "eval_threshold": ("run", "eval_threshold"),
        "detectors": ("plugins", "detector_spec"),
        "extended_detectors": ("plugins", "extended_detectors"),
        "probe_options": ("plugins", "probes"),
        "detector_options": ("plugins", "detectors"),
        "buff_options": ("plugins", "buffs"),
        "harness_options": ("plugins", "harnesses"),
        "taxonomy": ("reporting", "taxonomy"),
    }

    for key, path in key_to_path.items():
        value = benchmark_config.get(key)
        if value is not None:
            _set_nested_value(overrides, path, value)

    selection_keys = ("probes", "buffs", "probe_tags")
    if any(key in benchmark_config and benchmark_config[key] is not None for key in selection_keys):
        include: list[Any] = []
        include.extend(_canonical_plugin_selectors(benchmark_config.get("probes"), "probes"))
        include.extend(_canonical_plugin_selectors(benchmark_config.get("buffs"), "buffs"))
        include.extend(_tag_selectors(benchmark_config.get("probe_tags")))
        _set_nested_value(overrides, ("run", "spec"), {"include": _dedupe_selectors(include), "exclude": []})

    return overrides


def _normalize_plugin_specs(config_dict: dict[str, Any]) -> None:
    """Normalize detector specifications that still use Garak's string grammar."""
    plugins = config_dict.get("plugins")
    if not isinstance(plugins, dict):
        return

    value = plugins.get("detector_spec")
    if isinstance(value, list):
        plugins["detector_spec"] = ",".join(str(item) for item in value)


def build_effective_garak_config(
    benchmark_config: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> GarakCommandConfig:
    """Build a merged Garak config with explicit precedence rules.

    Precedence from lowest to highest is provider defaults, the built-in
    profile, ``benchmark_config.garak_config``, and flat EvalHub aliases.
    """
    bc = dict(benchmark_config or {})
    profile_garak_cfg = profile.get("garak_config", {})

    explicit_garak_cfg = bc.get("garak_config", {})
    if explicit_garak_cfg is None:
        explicit_garak_cfg = {}
    if explicit_garak_cfg and not isinstance(explicit_garak_cfg, dict):
        raise ValueError("benchmark_config.garak_config must be a dictionary")
    if profile_garak_cfg and not isinstance(profile_garak_cfg, dict):
        raise ValueError("profile.garak_config must be a dictionary")

    profile_cfg = dict(profile_garak_cfg or {})
    explicit_cfg = dict(explicit_garak_cfg or {})
    if any(
        isinstance(cfg.get("cas"), dict) and ("expand_intent_tree" in cfg["cas"] or "trust_code_stubs" in cfg["cas"])
        for cfg in (profile_cfg, explicit_cfg)
    ):
        logger.warning(
            "Ignoring deprecated cas.expand_intent_tree and cas.trust_code_stubs. "
            "Garak 0.17 has no equivalent behavior."
        )

    merged = deep_merge_dicts(
        GarakCommandConfig().to_dict(exclude_none=True),
        _migrate_legacy_garak_config(profile_cfg),
    )
    explicit_cfg = _migrate_legacy_garak_config(explicit_cfg)
    merged = deep_merge_dicts(merged, explicit_cfg)
    if bc.get("num_examples") is not None:
        logger.warning(
            "num_examples is a soft per-probe prompt cap, not a total scan limit; "
            "some probes may exceed it. Use timeout_seconds to bound scan duration."
        )
        if "soft_probe_prompt_cap" not in explicit_cfg.get("run", {}):
            merged["run"]["soft_probe_prompt_cap"] = bc["num_examples"]
    merged = deep_merge_dicts(merged, legacy_overrides_from_benchmark_config(bc))
    _normalize_plugin_specs(merged)

    return GarakCommandConfig.from_dict(merged)


def resolve_timeout_seconds(
    benchmark_config: Mapping[str, Any],
    profile: Mapping[str, Any],
    default_timeout: int = 600,
) -> int:
    """Resolve timeout with explicit override, profile fallback, then default."""
    bc = dict(benchmark_config or {})

    timeouts = [bc.get("timeout_seconds"), bc.get("timeout"), profile.get("timeout")]
    for timeout in timeouts:
        if timeout is not None:
            try:
                int_timeout = int(timeout)
                if int_timeout >= 0:
                    return int_timeout
            except (TypeError, ValueError):
                pass

    return default_timeout


def resolve_max_retry_seconds(benchmark_config: Mapping[str, Any], default: int) -> int:
    """Retry budget from the benchmark parameters (``max_retry_seconds``); 0 disables it."""
    value = (benchmark_config or {}).get("max_retry_seconds")
    if value is not None:
        try:
            seconds = int(value)
            if seconds >= 0:
                return seconds
        except (TypeError, ValueError):
            pass
        logger.warning("Ignoring invalid max_retry_seconds=%r; using %s", value, default)
    return default


__all__ = [
    "build_effective_garak_config",
    "deep_merge_dicts",
    "legacy_overrides_from_benchmark_config",
    "resolve_scan_profile",
    "resolve_max_retry_seconds",
    "resolve_timeout_seconds",
]
