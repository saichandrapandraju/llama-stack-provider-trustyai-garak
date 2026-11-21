"""Tests for eval-hub Garak adapter alignment behavior."""

from __future__ import annotations

import importlib
import json
import sys
import types
from copy import deepcopy
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
from evalhub.adapter.models.job import JobSpec as SDKJobSpec

from llama_stack_provider_trustyai_garak.core.config_resolution import (
    build_effective_garak_config,
    resolve_scan_profile,
    resolve_timeout_seconds,
)
from llama_stack_provider_trustyai_garak.core.garak_runner import convert_to_avid_report


def _load_evalhub_garak_adapter(monkeypatch) -> types.ModuleType:
    """Import the adapter with lightweight SDK stubs and retained fixture artifacts."""
    monkeypatch.setenv("GARAK_RETAIN_SCAN_ARTIFACTS", "true")

    class _SimpleModel:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FrameworkAdapter:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

    class JobStatus(str, Enum):
        RUNNING = "running"
        FAILED = "failed"

    class JobPhase(str, Enum):
        INITIALIZING = "initializing"
        LOADING_DATA = "loading_data"
        RUNNING_EVALUATION = "running_evaluation"
        POST_PROCESSING = "post_processing"
        PERSISTING_ARTIFACTS = "persisting_artifacts"

    adapter_module = types.ModuleType("evalhub.adapter")
    adapter_models_job_module = types.ModuleType("evalhub.adapter.models.job")
    models_api_module = types.ModuleType("evalhub.models.api")
    evalhub_module = types.ModuleType("evalhub")
    adapter_models_module = types.ModuleType("evalhub.adapter.models")
    models_module = types.ModuleType("evalhub.models")

    adapter_module.DefaultCallbacks = _SimpleModel
    adapter_module.FrameworkAdapter = FrameworkAdapter
    adapter_module.JobCallbacks = _SimpleModel
    adapter_module.JobPhase = JobPhase
    adapter_module.JobResults = _SimpleModel
    adapter_module.JobSpec = _SimpleModel
    adapter_module.JobStatus = JobStatus
    adapter_module.JobStatusUpdate = _SimpleModel
    adapter_module.MessageInfo = _SimpleModel
    adapter_module.OCIArtifactSpec = _SimpleModel

    adapter_models_job_module.MessageInfo = _SimpleModel
    models_api_module.EvaluationResult = _SimpleModel

    adapter_auth_module = types.ModuleType("evalhub.adapter.auth")
    adapter_auth_module.read_model_auth_key = lambda _name: None

    monkeypatch.setitem(sys.modules, "evalhub", evalhub_module)
    monkeypatch.setitem(sys.modules, "evalhub.adapter", adapter_module)
    monkeypatch.setitem(sys.modules, "evalhub.adapter.auth", adapter_auth_module)
    monkeypatch.setitem(sys.modules, "evalhub.adapter.models", adapter_models_module)
    monkeypatch.setitem(sys.modules, "evalhub.adapter.models.job", adapter_models_job_module)
    monkeypatch.setitem(sys.modules, "evalhub.models", models_module)
    monkeypatch.setitem(sys.modules, "evalhub.models.api", models_api_module)

    # Ensure this import uses the stubbed evalhub modules above.
    sys.modules.pop("llama_stack_provider_trustyai_garak.evalhub.garak_adapter", None)
    return importlib.import_module("llama_stack_provider_trustyai_garak.evalhub.garak_adapter")


def test_resolve_scan_profile_accepts_prefixed_and_unprefixed_ids():
    prefixed = resolve_scan_profile("trustyai_garak::owasp_llm_top10")
    unprefixed = resolve_scan_profile("owasp_llm_top10")

    assert prefixed["name"] == "OWASP LLM Top 10"
    assert unprefixed["name"] == "OWASP LLM Top 10"
    assert unprefixed["garak_config"]["run"]["spec"] == prefixed["garak_config"]["run"]["spec"]


def test_build_effective_garak_config_honors_precedence():
    profile = resolve_scan_profile("trustyai_garak::quick")
    benchmark_config = {
        "garak_config": {
            "run": {"generations": 3},
            "plugins": {"probe_spec": ["promptinject"], "extended_detectors": False},
        },
        # Legacy key should override explicit nested key where both are set.
        "generations": 5,
        "probe_tags": "owasp:llm",
    }

    resolved = build_effective_garak_config(benchmark_config, profile)
    resolved_dict = resolved.to_dict(exclude_none=True)

    assert resolved_dict["run"]["generations"] == 5
    assert resolved_dict["run"]["spec"] == {"include": [{"tag": "owasp:llm"}], "exclude": []}
    assert "probe_tags" not in resolved_dict["run"]
    assert "probe_spec" not in resolved_dict["plugins"]
    assert "buff_spec" not in resolved_dict["plugins"]
    assert resolved_dict["plugins"]["extended_detectors"] is False


def test_resolve_timeout_seconds_precedence():
    profile = {"timeout": 3600}

    assert resolve_timeout_seconds({"timeout_seconds": 120}, profile, 600) == 120
    assert resolve_timeout_seconds({"timeout": 180}, profile, 600) == 180
    assert resolve_timeout_seconds({}, profile, 600) == 3600
    assert resolve_timeout_seconds({}, {}, 600) == 600

    assert resolve_timeout_seconds({"timeout_seconds": 0}, profile, 600) == 0
    assert resolve_timeout_seconds({"timeout": 0}, profile, 600) == 0
    assert resolve_timeout_seconds({}, {"timeout": 0}, 600) == 0

    # Negative values are rejected and fall through to the next source
    assert resolve_timeout_seconds({"timeout_seconds": -1}, profile, 600) == 3600
    assert resolve_timeout_seconds({"timeout": -1}, {}, 600) == 600
    assert resolve_timeout_seconds({}, {"timeout": -5}, 600) == 600

    # Non-numeric values are ignored and fall through
    assert resolve_timeout_seconds({"timeout_seconds": "abc"}, profile, 600) == 3600
    assert resolve_timeout_seconds({"timeout": None}, profile, 600) == 3600


@pytest.mark.parametrize(
    "benchmark_id",
    ["owasp_llm_top10", "avid", "avid_security", "avid_ethics", "avid_performance", "quality", "cwe", "intents"],
)
def test_broad_scan_prompt_cap_preserves_probe_selection(benchmark_id):
    profile = resolve_scan_profile(benchmark_id)
    resolved = build_effective_garak_config({"num_examples": 1}, profile)
    assert resolved.run.soft_probe_prompt_cap == 1
    assert resolved.run.spec.model_dump() == profile["garak_config"]["run"]["spec"]


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        ({"num_examples": 1}, 1),
        ({"num_examples": 1, "garak_config": {"run": {"soft_probe_prompt_cap": 3}}}, 3),
        ({"num_examples": 1, "soft_probe_prompt_cap": 2}, 2),
        ({"num_examples": 1, "soft_probe_prompt_cap": 2, "garak_config": {"run": {"soft_probe_prompt_cap": 3}}}, 2),
    ],
)
def test_prompt_cap_precedence(parameters, expected):
    assert build_effective_garak_config(parameters, resolve_scan_profile("avid")).run.soft_probe_prompt_cap == expected


@pytest.mark.parametrize(
    ("num_examples", "parameters", "expected_cap"),
    [
        (1, {}, 1),
        (None, {}, 256),
        (None, {"num_examples": 2}, 2),
        (1, {"num_examples": 2}, 1),
        (1, {"soft_probe_prompt_cap": 3}, 3),
        (1, {"garak_config": {"run": {"soft_probe_prompt_cap": 4}}}, 4),
        (1, {"soft_probe_prompt_cap": 3, "garak_config": {"run": {"soft_probe_prompt_cap": 4}}}, 3),
    ],
)
def test_job_spec_prompt_cap(monkeypatch, tmp_path, num_examples, parameters, expected_cap):
    job = SDKJobSpec(
        id="prompt-cap-job",
        provider_id="garak",
        benchmark_id="owasp_llm_top10",
        benchmark_index=0,
        model={"url": "https://model.example/v1", "name": "test-model"},
        parameters=deepcopy(parameters),
        callback_url="https://evalhub.example",
        num_examples=num_examples,
    )
    module = _load_evalhub_garak_adapter(monkeypatch)

    config, profile, _ = module.GarakAdapter()._build_config_from_spec(job, tmp_path / "scan")

    assert job.parameters == parameters
    assert config["run"]["soft_probe_prompt_cap"] == expected_cap
    assert config["run"]["spec"] == profile["garak_config"]["run"]["spec"]


@pytest.mark.parametrize("run", [None, [], "legacy", 42, False])
def test_prompt_cap_alias_normalizes_non_mapping_run(run):
    resolved = build_effective_garak_config(
        {"num_examples": 1, "garak_config": {"run": run}}, resolve_scan_profile("avid")
    )
    assert resolved.run.soft_probe_prompt_cap == 1
    assert resolved.run.spec.include == [{"tag": "avid-effect"}]


@pytest.mark.parametrize(
    ("sdg", "parameters"),
    [({"name": "openai/gpt-oss-120b", "provider": "openai"}, {}), ({}, {"sdg_provider": "openai"})],
)
def test_extract_sdg_provider(monkeypatch, sdg, parameters):
    module = _load_evalhub_garak_adapter(monkeypatch)
    assert module.GarakAdapter._extract_sdg_params(sdg, parameters, {})["sdg_provider"] == "openai"


def test_run_benchmark_job_uses_garak_config_file(monkeypatch, tmp_path):
    module = _load_evalhub_garak_adapter(monkeypatch)
    adapter = module.GarakAdapter()
    monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

    captured: dict[str, object] = {}

    def _fake_run_garak_scan(
        config_file,
        timeout_seconds,
        report_prefix,
        env=None,
        log_file=None,
    ):
        _ = env, log_file
        captured["config_file"] = config_file
        captured["timeout_seconds"] = timeout_seconds
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")
        return module.GarakScanResult(
            returncode=0,
            stdout="",
            stderr="",
            report_prefix=report_prefix,
        )

    monkeypatch.setattr(module, "run_garak_scan", _fake_run_garak_scan)
    monkeypatch.setattr(module, "convert_to_avid_report", lambda _path: True)
    monkeypatch.setattr(
        module.GarakAdapter,
        "_parse_results",
        lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
    )

    class _Callbacks:
        def report_status(self, _update):
            return None

        def create_oci_artifact(self, _spec):
            return SimpleNamespace(reference="oci://ref", digest="sha256:test")

    job = SimpleNamespace(
        id="job-1",
        benchmark_id="trustyai_garak::quick",
        benchmark_index=0,
        model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
        parameters={"save_sample_results": False, "timeout_seconds": 42, "generations": 2},
        exports=None,
    )

    adapter.run_benchmark_job(job, _Callbacks())

    config_path = Path(captured["config_file"])
    assert config_path.exists()

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert config_payload["reporting"]["report_prefix"].endswith("scan")
    assert config_payload["run"]["generations"] == 2
    assert captured["timeout_seconds"] == 42


def test_simple_mode_passes_hf_cache_env(monkeypatch, tmp_path):
    """When hf_cache_path is set, _run_simple passes HF_HUB_CACHE via env to run_garak_scan."""
    module = _load_evalhub_garak_adapter(monkeypatch)
    adapter = module.GarakAdapter()
    monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

    captured: dict[str, object] = {}

    def _fake_run_garak_scan(config_file, timeout_seconds, report_prefix, env=None, log_file=None):
        captured["env"] = env
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")
        return module.GarakScanResult(returncode=0, stdout="", stderr="", report_prefix=report_prefix)

    monkeypatch.setattr(module, "run_garak_scan", _fake_run_garak_scan)
    monkeypatch.setattr(module, "convert_to_avid_report", lambda _path: True)
    monkeypatch.setattr(
        module.GarakAdapter,
        "_parse_results",
        lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
    )

    class _Callbacks:
        def report_status(self, _update):
            return None

        def create_oci_artifact(self, _spec):
            return SimpleNamespace(reference="oci://ref", digest="sha256:test")

    job = SimpleNamespace(
        id="hf-cache-job",
        benchmark_id="trustyai_garak::quick",
        benchmark_index=0,
        model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
        parameters={"save_sample_results": False, "hf_cache_path": "/test_data/hf-cache"},
        exports=None,
    )

    adapter.run_benchmark_job(job, _Callbacks())
    assert captured["env"] == {"HF_HUB_CACHE": "/test_data/hf-cache"}


def test_simple_mode_no_hf_cache_passes_none_env(monkeypatch, tmp_path):
    """When hf_cache_path is not set, env=None is passed (default behavior)."""
    module = _load_evalhub_garak_adapter(monkeypatch)
    adapter = module.GarakAdapter()
    monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

    captured: dict[str, object] = {}

    def _fake_run_garak_scan(config_file, timeout_seconds, report_prefix, env=None, log_file=None):
        captured["env"] = env
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")
        return module.GarakScanResult(returncode=0, stdout="", stderr="", report_prefix=report_prefix)

    monkeypatch.setattr(module, "run_garak_scan", _fake_run_garak_scan)
    monkeypatch.setattr(module, "convert_to_avid_report", lambda _path: True)
    monkeypatch.setattr(
        module.GarakAdapter,
        "_parse_results",
        lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
    )

    class _Callbacks:
        def report_status(self, _update):
            return None

        def create_oci_artifact(self, _spec):
            return SimpleNamespace(reference="oci://ref", digest="sha256:test")

    job = SimpleNamespace(
        id="no-hf-cache-job",
        benchmark_id="trustyai_garak::quick",
        benchmark_index=0,
        model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
        parameters={
            "save_sample_results": False,
        },
        exports=None,
    )

    adapter.run_benchmark_job(job, _Callbacks())
    assert captured["env"] is None


def test_parse_results_uses_overall_without_double_count(monkeypatch, tmp_path):
    module = _load_evalhub_garak_adapter(monkeypatch)
    adapter = module.GarakAdapter()

    report_prefix = tmp_path / "scan"
    report_prefix.with_suffix(".report.jsonl").write_text(
        '{"entry_type":"attempt","status":2}\n',
        encoding="utf-8",
    )
    report_prefix.with_suffix(".avid.jsonl").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "parse_generations_from_report_content",
        lambda _content, _threshold, **_kwargs: ([], {"probe.alpha": [{}]}, {"probe.alpha": [{}]}),
    )
    monkeypatch.setattr(
        module,
        "parse_aggregated_from_avid_content",
        lambda _content: {"probe.alpha": {"total_attempts": 10, "vulnerable_responses": 3}},
    )
    monkeypatch.setattr(module, "parse_digest_from_report_content", lambda _content: {"eval": {}})
    monkeypatch.setattr(
        module,
        "combine_parsed_results",
        lambda *_args, **_kwargs: {
            "scores": {
                "probe.alpha": {
                    "aggregated_results": {
                        "total_attempts": 10,
                        "vulnerable_responses": 3,
                        "benign_responses": 7,
                        "attack_success_rate": 30.0,
                        "metadata": {},
                    }
                },
                "_overall": {
                    "aggregated_results": {
                        "total_attempts": 10,
                        "vulnerable_responses": 3,
                        "attack_success_rate": 30.0,
                        "tbsa": 4.1,
                    }
                },
            }
        },
    )

    result = module.GarakScanResult(
        returncode=0,
        stdout="",
        stderr="",
        report_prefix=report_prefix,
    )
    metrics, overall_score, num_examples, overall_summary = adapter._parse_results(result, 0.5)

    assert len(metrics) == 2
    assert metrics[0].metric_name == "attack_success_rate"
    assert metrics[0].metric_type == "ratio"
    assert metrics[0].metric_value == 0.3
    assert metrics[0].num_samples == 10
    assert metrics[1].metric_name == "probe.alpha_asr"
    assert metrics[1].metric_type == "ratio"
    assert overall_score == 0.3
    assert num_examples == 10
    assert overall_summary["tbsa"] == 4.1


def test_convert_to_avid_report_imports_garak_report(monkeypatch, tmp_path):
    report_path = tmp_path / "scan.report.jsonl"
    report_path.write_text("{}", encoding="utf-8")

    called: dict[str, bool] = {"export": False}

    class _FakeReport:
        def __init__(self, _path):
            pass

        def load(self):
            return self

        def get_evaluations(self):
            return self

        def export(self):
            called["export"] = True

    fake_garak = types.ModuleType("garak")
    fake_report_module = types.ModuleType("garak.report")
    fake_report_module.Report = _FakeReport

    monkeypatch.setitem(sys.modules, "garak", fake_garak)
    monkeypatch.setitem(sys.modules, "garak.report", fake_report_module)

    assert convert_to_avid_report(report_path) is True
    assert called["export"] is True


# ---------------------------------------------------------------------------
# KFP mode tests
# ---------------------------------------------------------------------------


class TestResolveExecutionMode:
    """Tests for _resolve_execution_mode static method."""

    def test_defaults_to_simple(self, monkeypatch):
        monkeypatch.delenv("EVALHUB_EXECUTION_MODE", raising=False)
        module = _load_evalhub_garak_adapter(monkeypatch)
        assert module.GarakAdapter._resolve_execution_mode({}) == "simple"

    def test_benchmark_config_overrides_env(self, monkeypatch):
        monkeypatch.setenv("EVALHUB_EXECUTION_MODE", "simple")
        module = _load_evalhub_garak_adapter(monkeypatch)
        assert module.GarakAdapter._resolve_execution_mode({"execution_mode": "kfp"}) == "kfp"

    def test_env_var_used_when_no_benchmark_config(self, monkeypatch):
        monkeypatch.setenv("EVALHUB_EXECUTION_MODE", "kfp")
        module = _load_evalhub_garak_adapter(monkeypatch)
        assert module.GarakAdapter._resolve_execution_mode({}) == "kfp"

    def test_unknown_mode_falls_back_to_simple(self, monkeypatch):
        monkeypatch.delenv("EVALHUB_EXECUTION_MODE", raising=False)
        module = _load_evalhub_garak_adapter(monkeypatch)
        assert module.GarakAdapter._resolve_execution_mode({"execution_mode": "unknown"}) == "simple"


class TestKFPConfig:
    """Tests for KFPConfig.from_env_and_config."""

    def test_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("KFP_ENDPOINT", "https://kfp.example.com")
        monkeypatch.setenv("KFP_NAMESPACE", "test-ns")
        monkeypatch.setenv("KFP_S3_SECRET_NAME", "my-data-connection")
        monkeypatch.setenv("AWS_S3_BUCKET", "my-bucket")

        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import KFPConfig

        cfg = KFPConfig.from_env_and_config()
        assert cfg.endpoint == "https://kfp.example.com"
        assert cfg.namespace == "test-ns"
        assert cfg.s3_secret_name == "my-data-connection"
        assert cfg.s3_bucket == "my-bucket"
        assert cfg.experiment_name == "evalhub-garak"

    def test_benchmark_config_overrides_env(self, monkeypatch):
        monkeypatch.setenv("KFP_ENDPOINT", "https://env.example.com")
        monkeypatch.setenv("KFP_NAMESPACE", "env-ns")

        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import KFPConfig

        cfg = KFPConfig.from_env_and_config(
            {
                "kfp_config": {
                    "endpoint": "https://override.example.com",
                    "namespace": "override-ns",
                    "s3_secret_name": "custom-s3-conn",
                    "s3_bucket": "override-bucket",
                }
            }
        )
        assert cfg.endpoint == "https://override.example.com"
        assert cfg.namespace == "override-ns"
        assert cfg.s3_secret_name == "custom-s3-conn"
        assert cfg.s3_bucket == "override-bucket"

    def test_missing_endpoint_raises(self, monkeypatch):
        monkeypatch.delenv("KFP_ENDPOINT", raising=False)
        monkeypatch.setenv("KFP_NAMESPACE", "ns")

        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import KFPConfig
        import pytest

        with pytest.raises(ValueError) as excinfo:
            KFPConfig.from_env_and_config()

        msg = str(excinfo.value)
        assert "KFP_ENDPOINT" in msg
        assert "kfp_config.endpoint" in msg

    def test_missing_namespace_raises(self, monkeypatch):
        monkeypatch.setenv("KFP_ENDPOINT", "https://kfp.example.com")
        monkeypatch.delenv("KFP_NAMESPACE", raising=False)

        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import KFPConfig
        import pytest

        with pytest.raises(ValueError) as excinfo:
            KFPConfig.from_env_and_config()

        msg = str(excinfo.value)
        assert "KFP_NAMESPACE" in msg
        assert "kfp_config.namespace" in msg

    def test_verify_ssl_false(self, monkeypatch):
        monkeypatch.setenv("KFP_ENDPOINT", "https://kfp.example.com")
        monkeypatch.setenv("KFP_NAMESPACE", "ns")
        monkeypatch.setenv("KFP_VERIFY_SSL", "false")

        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import KFPConfig

        cfg = KFPConfig.from_env_and_config()
        assert cfg.verify_ssl is False


class TestKFPModeExecution:
    """Tests for KFP mode in run_benchmark_job."""

    def test_kfp_mode_submits_pipeline(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        captured: dict[str, object] = {}

        report_prefix = tmp_path / "scan"
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")

        def _fake_run_via_kfp(
            self, config, callbacks, garak_config_dict, timeout, intents_params=None, eval_threshold=0.5, **kwargs
        ):
            captured["called"] = True
            captured["timeout"] = timeout
            captured["config_json"] = garak_config_dict
            captured["intents_params"] = intents_params
            captured["eval_threshold"] = eval_threshold
            return module.GarakScanResult(
                returncode=0,
                stdout="",
                stderr="",
                report_prefix=report_prefix,
            ), tmp_path

        monkeypatch.setattr(module.GarakAdapter, "_run_via_kfp", _fake_run_via_kfp)
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        class _Callbacks:
            def report_status(self, _update):
                return None

            def create_oci_artifact(self, _spec):
                return SimpleNamespace(reference="oci://ref", digest="sha256:test")

        job = SimpleNamespace(
            id="kfp-job-1",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
            parameters={"save_sample_results": False, "execution_mode": "kfp", "timeout_seconds": 99},
            exports=None,
        )

        result = adapter.run_benchmark_job(job, _Callbacks())
        assert captured["called"] is True
        assert captured["timeout"] == 99
        assert captured["eval_threshold"] == 0.5
        assert result.evaluation_metadata["execution_mode"] == "kfp"

    def test_simple_mode_does_not_call_kfp(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        kfp_called = {"value": False}

        def _fake_run_via_kfp(*args, **kwargs):
            kfp_called["value"] = True

        monkeypatch.setattr(module.GarakAdapter, "_run_via_kfp", _fake_run_via_kfp)

        report_prefix = tmp_path / "kfp-job-2" / "scan"
        report_prefix.parent.mkdir(parents=True)
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")

        def _fake_run_garak_scan(config_file, timeout_seconds, report_prefix, env=None, log_file=None):
            report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")
            return module.GarakScanResult(
                returncode=0,
                stdout="",
                stderr="",
                report_prefix=report_prefix,
            )

        monkeypatch.setattr(module, "run_garak_scan", _fake_run_garak_scan)
        monkeypatch.setattr(module, "convert_to_avid_report", lambda _path: True)
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        class _Callbacks:
            def report_status(self, _update):
                return None

            def create_oci_artifact(self, _spec):
                return SimpleNamespace(reference="oci://ref", digest="sha256:test")

        job = SimpleNamespace(
            id="kfp-job-2",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
            parameters={
                "save_sample_results": False,
            },
            exports=None,
        )

        result = adapter.run_benchmark_job(job, _Callbacks())
        assert kfp_called["value"] is False
        assert result.evaluation_metadata["execution_mode"] == "simple"

    def _setup_mlflow_test(self, monkeypatch, tmp_path, job_id, mlflow_return_value):
        """Common setup for MLflow tests: mocks garak scan, parse, and injects fake mlflow module."""
        import types
        from unittest.mock import MagicMock

        mlflow_mod = types.ModuleType("evalhub.adapter.mlflow")
        mlflow_mod.MlflowArtifact = lambda name, data, mime: SimpleNamespace(name=name, data=data, mime_type=mime)
        monkeypatch.setitem(__import__("sys").modules, "evalhub.adapter.mlflow", mlflow_mod)

        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))
        monkeypatch.setattr(module.GarakAdapter, "_run_via_kfp", lambda *a, **kw: None)

        report_prefix = tmp_path / job_id / "scan"
        report_prefix.parent.mkdir(parents=True)
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")

        def _fake_run_garak_scan(config_file, timeout_seconds, report_prefix, env=None, log_file=None):
            report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")
            return module.GarakScanResult(returncode=0, stdout="", stderr="", report_prefix=report_prefix)

        monkeypatch.setattr(module, "run_garak_scan", _fake_run_garak_scan)
        monkeypatch.setattr(module, "convert_to_avid_report", lambda _path: True)
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        mock_mlflow = MagicMock()
        mock_mlflow.save.return_value = mlflow_return_value

        class _Callbacks:
            mlflow = mock_mlflow

            def report_status(self, _update):
                return None

            def create_oci_artifact(self, _spec):
                return SimpleNamespace(reference="oci://ref", digest="sha256:test")

        job = SimpleNamespace(
            id=job_id,
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
            parameters={
                "save_sample_results": False,
            },
            exports=None,
        )
        return adapter, job, _Callbacks(), mock_mlflow

    def test_mlflow_run_id_saved_on_results(self, monkeypatch, tmp_path):
        """When callbacks.mlflow.save() returns a run ID, it is stored on results."""
        adapter, job, callbacks, mock_mlflow = self._setup_mlflow_test(
            monkeypatch, tmp_path, "mlflow-job", "run-abc-123"
        )
        result = adapter.run_benchmark_job(job, callbacks)
        mock_mlflow.save.assert_called_once()
        assert result.mlflow_run_id == "run-abc-123"

    def test_mlflow_run_id_not_set_when_save_returns_none(self, monkeypatch, tmp_path):
        """When callbacks.mlflow.save() returns None, mlflow_run_id is not set."""
        adapter, job, callbacks, mock_mlflow = self._setup_mlflow_test(monkeypatch, tmp_path, "mlflow-job-2", None)
        result = adapter.run_benchmark_job(job, callbacks)
        mock_mlflow.save.assert_called_once()
        assert not hasattr(result, "mlflow_run_id") or result.mlflow_run_id is None

    def test_mlflow_save_exception_is_non_fatal(self, monkeypatch, tmp_path):
        """When callbacks.mlflow.save() raises, the job still succeeds and mlflow_run_id is unset."""
        adapter, job, callbacks, mock_mlflow = self._setup_mlflow_test(monkeypatch, tmp_path, "mlflow-job-3", None)
        mock_mlflow.save.side_effect = Exception("boom")
        result = adapter.run_benchmark_job(job, callbacks)
        assert result is not None
        assert result.id == "mlflow-job-3"
        assert not hasattr(result, "mlflow_run_id") or result.mlflow_run_id is None


class TestS3Download:
    """Tests for S3 download logic in the adapter."""

    def test_download_results_from_s3(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)

        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_S3_ENDPOINT", "http://minio:9000")

        downloaded_files = []

        class _FakeS3Client:
            def get_paginator(self, _method):
                return self

            def paginate(self, Bucket, Prefix):
                return [
                    {
                        "Contents": [
                            {"Key": f"{Prefix}/scan.report.jsonl"},
                            {"Key": f"{Prefix}/scan.avid.jsonl"},
                            {"Key": f"{Prefix}/scan.log"},
                            {"Key": f"{Prefix}/config.json"},
                        ]
                    }
                ]

            def download_file(self, bucket, key, local_path):
                downloaded_files.append(key)
                Path(local_path).write_text(f"content of {key}")

        monkeypatch.setattr(
            module.GarakAdapter,
            "_create_s3_client",
            staticmethod(lambda **kwargs: _FakeS3Client()),
        )

        local_dir = tmp_path / "results"
        local_dir.mkdir()

        module.GarakAdapter._download_results_from_s3(
            "test-bucket",
            "evalhub-garak/job-123",
            local_dir,
        )

        assert len(downloaded_files) == 4
        assert (local_dir / "scan.report.jsonl").exists()
        assert (local_dir / "scan.avid.jsonl").exists()
        assert (local_dir / "scan.log").exists()
        assert (local_dir / "config.json").exists()

    def test_download_skips_when_no_bucket(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)

        local_dir = tmp_path / "results"
        local_dir.mkdir()

        module.GarakAdapter._download_results_from_s3("", "prefix", local_dir)
        assert list(local_dir.iterdir()) == []

    def test_download_handles_empty_listing(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)

        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")

        class _FakeS3Client:
            def get_paginator(self, _method):
                return self

            def paginate(self, Bucket, Prefix):
                return [{"Contents": []}]

        monkeypatch.setattr(
            module.GarakAdapter,
            "_create_s3_client",
            staticmethod(lambda **kwargs: _FakeS3Client()),
        )

        local_dir = tmp_path / "results"
        local_dir.mkdir()

        module.GarakAdapter._download_results_from_s3(
            "test-bucket",
            "evalhub-garak/job-empty",
            local_dir,
        )
        assert list(local_dir.iterdir()) == []


class TestResolveS3Credentials:
    """Tests for _resolve_s3_credentials merging logic."""

    def _make_kfp_config(self, **overrides):
        defaults = {
            "s3_bucket": "",
            "s3_endpoint": "",
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_kfp_config_takes_precedence_over_secret(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        kfp_cfg = self._make_kfp_config(s3_bucket="cfg-bucket", s3_endpoint="http://cfg:9000")
        secret = {
            "bucket": "secret-bucket",
            "endpoint_url": "http://secret:9000",
            "access_key": "ak",
            "secret_key": "sk",
            "region": "us-west-2",
        }
        resolved = module.GarakAdapter._resolve_s3_credentials(kfp_cfg, secret)
        assert resolved["bucket"] == "cfg-bucket"
        assert resolved["endpoint_url"] == "http://cfg:9000"
        assert resolved["access_key"] == "ak"
        assert resolved["secret_key"] == "sk"
        assert resolved["region"] == "us-west-2"

    def test_falls_back_to_secret_when_kfp_config_empty(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        kfp_cfg = self._make_kfp_config()
        secret = {
            "bucket": "secret-bucket",
            "endpoint_url": "http://secret:9000",
            "access_key": "ak",
            "secret_key": "sk",
            "region": "eu-west-1",
        }
        resolved = module.GarakAdapter._resolve_s3_credentials(kfp_cfg, secret)
        assert resolved["bucket"] == "secret-bucket"
        assert resolved["endpoint_url"] == "http://secret:9000"

    def test_falls_back_to_env_when_both_empty(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.setenv("AWS_S3_BUCKET", "env-bucket")
        monkeypatch.setenv("AWS_S3_ENDPOINT", "http://env:9000")
        kfp_cfg = self._make_kfp_config()
        resolved = module.GarakAdapter._resolve_s3_credentials(kfp_cfg, {})
        assert resolved["bucket"] == "env-bucket"
        assert resolved["endpoint_url"] == "http://env:9000"

    def test_secret_takes_precedence_over_env_when_kfp_empty(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.setenv("AWS_S3_BUCKET", "env-bucket")
        monkeypatch.setenv("AWS_S3_ENDPOINT", "http://env:9000")
        secret = {
            "bucket": "secret-bucket",
            "endpoint_url": "http://secret:9000",
            "access_key": "ak",
            "secret_key": "sk",
            "region": "eu-west-1",
        }
        kfp_cfg = self._make_kfp_config()
        resolved = module.GarakAdapter._resolve_s3_credentials(kfp_cfg, secret)
        assert resolved["bucket"] == "secret-bucket"
        assert resolved["endpoint_url"] == "http://secret:9000"

    def test_mismatch_logs_warning(self, monkeypatch, caplog):
        module = _load_evalhub_garak_adapter(monkeypatch)
        import logging

        kfp_cfg = self._make_kfp_config(s3_bucket="cfg-bucket", s3_endpoint="http://cfg:9000")
        secret = {
            "bucket": "secret-bucket",
            "endpoint_url": "http://secret:9000",
            "access_key": "",
            "secret_key": "",
            "region": "",
        }

        with caplog.at_level(logging.WARNING):
            resolved = module.GarakAdapter._resolve_s3_credentials(kfp_cfg, secret)

        assert resolved["bucket"] == "cfg-bucket"
        assert resolved["endpoint_url"] == "http://cfg:9000"
        assert any("S3 bucket mismatch" in msg for msg in caplog.messages)
        assert any("S3 endpoint_url mismatch" in msg for msg in caplog.messages)

    def test_matching_values_no_warning(self, monkeypatch, caplog):
        module = _load_evalhub_garak_adapter(monkeypatch)
        import logging

        kfp_cfg = self._make_kfp_config(s3_bucket="same-bucket", s3_endpoint="http://same:9000")
        secret = {
            "bucket": "same-bucket",
            "endpoint_url": "http://same:9000",
            "access_key": "ak",
            "secret_key": "sk",
            "region": "",
        }
        with caplog.at_level(logging.WARNING):
            resolved = module.GarakAdapter._resolve_s3_credentials(kfp_cfg, secret)

        assert resolved["bucket"] == "same-bucket"
        assert not any("mismatch" in msg for msg in caplog.messages)

    def test_none_returned_when_all_sources_empty(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
        monkeypatch.delenv("AWS_S3_ENDPOINT", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        kfp_cfg = self._make_kfp_config()
        resolved = module.GarakAdapter._resolve_s3_credentials(kfp_cfg, {})
        assert resolved["bucket"] is None
        assert resolved["endpoint_url"] is None
        assert resolved["access_key"] is None

    def test_access_key_falls_back_to_env(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-ak")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-sk")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "env-region")
        kfp_cfg = self._make_kfp_config()
        resolved = module.GarakAdapter._resolve_s3_credentials(kfp_cfg, {})
        assert resolved["access_key"] == "env-ak"
        assert resolved["secret_key"] == "env-sk"
        assert resolved["region"] == "env-region"

    def test_secret_creds_take_precedence_over_env_for_access_key(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-ak")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-sk")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "env-region")
        secret = {
            "access_key": "secret-ak",
            "secret_key": "secret-sk",
            "region": "secret-region",
        }
        kfp_cfg = self._make_kfp_config()
        resolved = module.GarakAdapter._resolve_s3_credentials(kfp_cfg, secret)
        assert resolved["access_key"] == "secret-ak"
        assert resolved["secret_key"] == "secret-sk"
        assert resolved["region"] == "secret-region"


class TestReadS3CredentialsCascade:
    """Tests for the cascading fallback chain in _read_s3_credentials_from_secret."""

    def test_step1_incluster_config_success(self, monkeypatch):
        """Step 1: Direct K8s API via load_incluster_config succeeds."""
        import base64

        module = _load_evalhub_garak_adapter(monkeypatch)

        fake_secret_data = {
            "AWS_ACCESS_KEY_ID": base64.b64encode(b"ak-incluster").decode(),
            "AWS_SECRET_ACCESS_KEY": base64.b64encode(b"sk-incluster").decode(),
            "AWS_DEFAULT_REGION": base64.b64encode(b"us-east-1").decode(),
            "AWS_S3_BUCKET": base64.b64encode(b"bucket-incluster").decode(),
            "AWS_S3_ENDPOINT": base64.b64encode(b"http://s3:9000").decode(),
        }

        class FakeV1:
            def read_namespaced_secret(self, name, ns):
                return SimpleNamespace(data=fake_secret_data)

        k8s_config = types.ModuleType("kubernetes.config")
        k8s_config.load_incluster_config = lambda: None
        k8s_config.ConfigException = type("ConfigException", (Exception,), {})

        k8s_client = types.ModuleType("kubernetes.client")
        k8s_client.CoreV1Api = FakeV1
        k8s_client.Configuration = type(
            "Configuration",
            (),
            {
                "host": "",
                "api_key": {},
                "verify_ssl": True,
                "set_default": classmethod(lambda cls, c: None),
            },
        )

        k8s_module = types.ModuleType("kubernetes")
        k8s_module.client = k8s_client
        k8s_module.config = k8s_config

        monkeypatch.setitem(sys.modules, "kubernetes", k8s_module)
        monkeypatch.setitem(sys.modules, "kubernetes.client", k8s_client)
        monkeypatch.setitem(sys.modules, "kubernetes.config", k8s_config)

        result = module.GarakAdapter._read_s3_credentials_from_secret("my-secret", "ns")
        assert result["access_key"] == "ak-incluster"
        assert result["bucket"] == "bucket-incluster"

    def test_step2_kfp_auth_token_fallback(self, monkeypatch):
        """Step 2: KFP_AUTH_TOKEN + KUBERNETES_SERVICE_HOST env vars."""
        import base64

        module = _load_evalhub_garak_adapter(monkeypatch)

        monkeypatch.setenv("KFP_AUTH_TOKEN", "fake-token")
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
        monkeypatch.setenv("KUBERNETES_SERVICE_PORT", "443")

        fake_data = {
            "AWS_ACCESS_KEY_ID": base64.b64encode(b"ak-token").decode(),
            "AWS_SECRET_ACCESS_KEY": base64.b64encode(b"sk-token").decode(),
            "AWS_DEFAULT_REGION": "",
            "AWS_S3_BUCKET": base64.b64encode(b"bucket-token").decode(),
            "AWS_S3_ENDPOINT": base64.b64encode(b"http://s3:9000").decode(),
        }

        _incluster_called = {"v": False}

        class FakeV1:
            def read_namespaced_secret(self, name, ns):
                return SimpleNamespace(data=fake_data)

        class FakeConfigException(Exception):
            pass

        k8s_config = types.ModuleType("kubernetes.config")

        def _fail_incluster():
            raise FakeConfigException("no SA token")

        k8s_config.load_incluster_config = _fail_incluster
        k8s_config.ConfigException = FakeConfigException

        k8s_client = types.ModuleType("kubernetes.client")
        k8s_client.CoreV1Api = FakeV1
        k8s_client.Configuration = type(
            "Configuration",
            (),
            {
                "host": "",
                "api_key": {},
                "verify_ssl": True,
                "set_default": classmethod(lambda cls, c: None),
            },
        )

        k8s_module = types.ModuleType("kubernetes")
        k8s_module.client = k8s_client
        k8s_module.config = k8s_config

        monkeypatch.setitem(sys.modules, "kubernetes", k8s_module)
        monkeypatch.setitem(sys.modules, "kubernetes.client", k8s_client)
        monkeypatch.setitem(sys.modules, "kubernetes.config", k8s_config)

        result = module.GarakAdapter._read_s3_credentials_from_secret("my-secret", "ns")
        assert result["access_key"] == "ak-token"
        assert result["bucket"] == "bucket-token"

    def test_step3_sidecar_proxy_fallback(self, monkeypatch):
        """Step 3: Sidecar proxy with k8s_sa_token:ref."""
        import base64

        module = _load_evalhub_garak_adapter(monkeypatch)

        monkeypatch.delenv("KFP_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)

        class FakeConfigException(Exception):
            pass

        k8s_config = types.ModuleType("kubernetes.config")
        k8s_config.load_incluster_config = lambda: (_ for _ in ()).throw(FakeConfigException("no"))
        k8s_config.ConfigException = FakeConfigException

        k8s_client = types.ModuleType("kubernetes.client")
        k8s_module = types.ModuleType("kubernetes")
        k8s_module.client = k8s_client
        k8s_module.config = k8s_config

        monkeypatch.setitem(sys.modules, "kubernetes", k8s_module)
        monkeypatch.setitem(sys.modules, "kubernetes.client", k8s_client)
        monkeypatch.setitem(sys.modules, "kubernetes.config", k8s_config)

        sidecar_data = {
            "AWS_ACCESS_KEY_ID": base64.b64encode(b"ak-sidecar").decode(),
            "AWS_SECRET_ACCESS_KEY": base64.b64encode(b"sk-sidecar").decode(),
            "AWS_DEFAULT_REGION": base64.b64encode(b"eu-west-1").decode(),
            "AWS_S3_BUCKET": base64.b64encode(b"sidecar-bucket").decode(),
            "AWS_S3_ENDPOINT": base64.b64encode(b"http://s3-sidecar:9000").decode(),
        }
        monkeypatch.setattr(
            module.GarakAdapter,
            "_read_secret_via_sidecar",
            staticmethod(lambda name, ns: sidecar_data),
        )

        result = module.GarakAdapter._read_s3_credentials_from_secret("my-secret", "ns")
        assert result["access_key"] == "ak-sidecar"
        assert result["bucket"] == "sidecar-bucket"
        assert result["endpoint_url"] == "http://s3-sidecar:9000"

    def test_step4_model_auth_fallback(self, monkeypatch):
        """Step 4: read_model_auth_key for S3 keys."""
        module = _load_evalhub_garak_adapter(monkeypatch)

        monkeypatch.delenv("KFP_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)

        class FakeConfigException(Exception):
            pass

        k8s_config = types.ModuleType("kubernetes.config")
        k8s_config.load_incluster_config = lambda: (_ for _ in ()).throw(FakeConfigException("no"))
        k8s_config.ConfigException = FakeConfigException

        k8s_client = types.ModuleType("kubernetes.client")
        k8s_module = types.ModuleType("kubernetes")
        k8s_module.client = k8s_client
        k8s_module.config = k8s_config

        monkeypatch.setitem(sys.modules, "kubernetes", k8s_module)
        monkeypatch.setitem(sys.modules, "kubernetes.client", k8s_client)
        monkeypatch.setitem(sys.modules, "kubernetes.config", k8s_config)

        monkeypatch.setattr(
            module.GarakAdapter,
            "_read_secret_via_sidecar",
            staticmethod(lambda name, ns: None),
        )

        model_auth_data = {
            "access_key": "ak-model-auth",
            "secret_key": "sk-model-auth",
            "region": "ap-south-1",
            "bucket": "model-auth-bucket",
            "endpoint_url": "http://s3-model:9000",
        }
        monkeypatch.setattr(
            module.GarakAdapter,
            "_read_s3_from_model_auth",
            staticmethod(lambda: model_auth_data),
        )

        result = module.GarakAdapter._read_s3_credentials_from_secret("my-secret", "ns")
        assert result["access_key"] == "ak-model-auth"
        assert result["bucket"] == "model-auth-bucket"
        assert result["region"] == "ap-south-1"

    def test_step5_kubeconfig_fallback(self, monkeypatch):
        """Step 5: load_kube_config for local dev."""
        import base64

        module = _load_evalhub_garak_adapter(monkeypatch)

        monkeypatch.delenv("KFP_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)

        fake_data = {
            "AWS_ACCESS_KEY_ID": base64.b64encode(b"ak-kubeconfig").decode(),
            "AWS_SECRET_ACCESS_KEY": base64.b64encode(b"sk-kubeconfig").decode(),
            "AWS_DEFAULT_REGION": "",
            "AWS_S3_BUCKET": base64.b64encode(b"kube-bucket").decode(),
            "AWS_S3_ENDPOINT": base64.b64encode(b"http://s3-local:9000").decode(),
        }

        class FakeConfigException(Exception):
            pass

        class FakeV1:
            def read_namespaced_secret(self, name, ns):
                return SimpleNamespace(data=fake_data)

        k8s_config = types.ModuleType("kubernetes.config")
        k8s_config.load_incluster_config = lambda: (_ for _ in ()).throw(FakeConfigException("no"))
        k8s_config.load_kube_config = lambda: None
        k8s_config.ConfigException = FakeConfigException

        k8s_client = types.ModuleType("kubernetes.client")
        k8s_client.CoreV1Api = FakeV1
        k8s_client.Configuration = type(
            "Configuration",
            (),
            {
                "host": "",
                "api_key": {},
                "verify_ssl": True,
                "set_default": classmethod(lambda cls, c: None),
            },
        )

        k8s_module = types.ModuleType("kubernetes")
        k8s_module.client = k8s_client
        k8s_module.config = k8s_config

        monkeypatch.setitem(sys.modules, "kubernetes", k8s_module)
        monkeypatch.setitem(sys.modules, "kubernetes.client", k8s_client)
        monkeypatch.setitem(sys.modules, "kubernetes.config", k8s_config)

        monkeypatch.setattr(
            module.GarakAdapter,
            "_read_secret_via_sidecar",
            staticmethod(lambda name, ns: None),
        )
        monkeypatch.setattr(
            module.GarakAdapter,
            "_read_s3_from_model_auth",
            staticmethod(lambda: {}),
        )

        result = module.GarakAdapter._read_s3_credentials_from_secret("my-secret", "ns")
        assert result["access_key"] == "ak-kubeconfig"
        assert result["bucket"] == "kube-bucket"

    def test_all_fallbacks_fail_returns_empty(self, monkeypatch):
        """All steps fail — returns empty dict."""
        module = _load_evalhub_garak_adapter(monkeypatch)

        monkeypatch.delenv("KFP_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)

        class FakeConfigException(Exception):
            pass

        k8s_config = types.ModuleType("kubernetes.config")
        k8s_config.load_incluster_config = lambda: (_ for _ in ()).throw(FakeConfigException("no"))
        k8s_config.load_kube_config = lambda: (_ for _ in ()).throw(Exception("no kubeconfig"))
        k8s_config.ConfigException = FakeConfigException

        k8s_client = types.ModuleType("kubernetes.client")
        k8s_client.CoreV1Api = lambda: None
        k8s_client.Configuration = type(
            "Configuration",
            (),
            {
                "host": "",
                "api_key": {},
                "verify_ssl": True,
                "set_default": classmethod(lambda cls, c: None),
            },
        )

        k8s_module = types.ModuleType("kubernetes")
        k8s_module.client = k8s_client
        k8s_module.config = k8s_config

        monkeypatch.setitem(sys.modules, "kubernetes", k8s_module)
        monkeypatch.setitem(sys.modules, "kubernetes.client", k8s_client)
        monkeypatch.setitem(sys.modules, "kubernetes.config", k8s_config)

        monkeypatch.setattr(
            module.GarakAdapter,
            "_read_secret_via_sidecar",
            staticmethod(lambda name, ns: None),
        )
        monkeypatch.setattr(
            module.GarakAdapter,
            "_read_s3_from_model_auth",
            staticmethod(lambda: {}),
        )

        result = module.GarakAdapter._read_s3_credentials_from_secret("my-secret", "ns")
        assert result == {}


class TestReadSecretViaSidecar:
    """Tests for _read_secret_via_sidecar."""

    def test_returns_none_when_k8s_url_not_set(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        auth_module = sys.modules["evalhub.adapter.auth"]
        auth_module.read_model_auth_key = lambda _name: None

        result = module.GarakAdapter._read_secret_via_sidecar("secret", "ns")
        assert result is None

    def test_calls_sidecar_with_correct_url_and_headers(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)

        auth_keys = {
            "k8s_url": "https://api.cluster.local:6443",
            "k8s_sa_token": "",
        }
        auth_module = sys.modules["evalhub.adapter.auth"]
        auth_module.read_model_auth_key = lambda name: auth_keys.get(name)

        import base64

        secret_payload = {
            "data": {
                "AWS_ACCESS_KEY_ID": base64.b64encode(b"ak").decode(),
                "AWS_S3_BUCKET": base64.b64encode(b"bkt").decode(),
            }
        }

        captured = {}

        class FakeResponse:
            def read(self):
                return json.dumps(secret_payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        original_urlopen = None

        def fake_urlopen(req, timeout=None, context=None):
            captured["url"] = req.full_url
            captured["auth"] = req.get_header("Authorization")
            return FakeResponse()

        import urllib.request

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        result = module.GarakAdapter._read_secret_via_sidecar("my-s3-secret", "evalhub-tenant")
        assert result["AWS_ACCESS_KEY_ID"] == base64.b64encode(b"ak").decode()
        assert captured["url"] == "http://localhost:8080/api/v1/namespaces/evalhub-tenant/secrets/my-s3-secret"
        assert captured["auth"] == "Bearer k8s_sa_token:ref"


class TestReadS3FromModelAuth:
    """Tests for _read_s3_from_model_auth."""

    def test_reads_standard_aws_keys_uppercase(self, monkeypatch):
        """Standard AWS_* uppercase keys (most common format)."""
        module = _load_evalhub_garak_adapter(monkeypatch)

        model_auth_keys = {
            "AWS_ACCESS_KEY_ID": "ma-ak",
            "AWS_SECRET_ACCESS_KEY": "ma-sk",
            "AWS_DEFAULT_REGION": "us-west-2",
            "AWS_S3_BUCKET": "ma-bucket",
            "AWS_S3_ENDPOINT": "http://s3-ma:9000",
        }
        auth_module = sys.modules["evalhub.adapter.auth"]
        auth_module.read_model_auth_key = lambda name: model_auth_keys.get(name)

        result = module.GarakAdapter._read_s3_from_model_auth()
        assert result["access_key"] == "ma-ak"
        assert result["secret_key"] == "ma-sk"
        assert result["region"] == "us-west-2"
        assert result["bucket"] == "ma-bucket"
        assert result["endpoint_url"] == "http://s3-ma:9000"

    def test_reads_standard_aws_keys_lowercase(self, monkeypatch):
        """Lowercase aws_* keys also work."""
        module = _load_evalhub_garak_adapter(monkeypatch)

        model_auth_keys = {
            "aws_access_key_id": "ma-ak-lower",
            "aws_secret_access_key": "ma-sk-lower",
            "aws_default_region": "eu-west-1",
            "aws_s3_bucket": "lower-bucket",
            "aws_s3_endpoint": "http://s3-lower:9000",
        }
        auth_module = sys.modules["evalhub.adapter.auth"]
        auth_module.read_model_auth_key = lambda name: model_auth_keys.get(name)

        result = module.GarakAdapter._read_s3_from_model_auth()
        assert result["access_key"] == "ma-ak-lower"
        assert result["secret_key"] == "ma-sk-lower"
        assert result["region"] == "eu-west-1"
        assert result["bucket"] == "lower-bucket"
        assert result["endpoint_url"] == "http://s3-lower:9000"

    def test_reads_custom_s3_prefix_keys(self, monkeypatch):
        """Custom s3_* keys as fallback."""
        module = _load_evalhub_garak_adapter(monkeypatch)

        model_auth_keys = {
            "s3_access_key": "ma-ak-custom",
            "s3_secret_key": "ma-sk-custom",
            "s3_region": "ap-south-1",
            "s3_bucket": "custom-bucket",
            "s3_endpoint": "http://s3-custom:9000",
        }
        auth_module = sys.modules["evalhub.adapter.auth"]
        auth_module.read_model_auth_key = lambda name: model_auth_keys.get(name)

        result = module.GarakAdapter._read_s3_from_model_auth()
        assert result["access_key"] == "ma-ak-custom"
        assert result["secret_key"] == "ma-sk-custom"
        assert result["region"] == "ap-south-1"
        assert result["bucket"] == "custom-bucket"
        assert result["endpoint_url"] == "http://s3-custom:9000"

    def test_uppercase_takes_precedence_over_lowercase(self, monkeypatch):
        """AWS_ACCESS_KEY_ID wins over aws_access_key_id."""
        module = _load_evalhub_garak_adapter(monkeypatch)

        model_auth_keys = {
            "AWS_ACCESS_KEY_ID": "upper-ak",
            "aws_access_key_id": "lower-ak",
            "s3_access_key": "custom-ak",
        }
        auth_module = sys.modules["evalhub.adapter.auth"]
        auth_module.read_model_auth_key = lambda name: model_auth_keys.get(name)

        result = module.GarakAdapter._read_s3_from_model_auth()
        assert result["access_key"] == "upper-ak"

    def test_returns_empty_when_no_keys_set(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)

        auth_module = sys.modules["evalhub.adapter.auth"]
        auth_module.read_model_auth_key = lambda _name: None

        result = module.GarakAdapter._read_s3_from_model_auth()
        assert result["access_key"] == ""
        assert result["bucket"] == ""

    def test_returns_empty_dict_when_evalhub_not_importable(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.delitem(sys.modules, "evalhub.adapter.auth", raising=False)

        result = module.GarakAdapter._read_s3_from_model_auth()
        assert result == {}


class TestPollKFPRun:
    """Tests for _poll_kfp_run static method."""

    def test_returns_on_succeeded(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)

        call_count = {"n": 0}

        class _FakeClient:
            def get_run(self, run_id):
                call_count["n"] += 1
                if call_count["n"] < 3:
                    return SimpleNamespace(state="RUNNING")
                return SimpleNamespace(state="SUCCEEDED")

        class _Callbacks:
            statuses = []

            def report_status(self, update):
                self.statuses.append(update)

        # Use 0 poll interval for test speed
        monkeypatch.setattr("time.sleep", lambda _: None)

        callbacks = _Callbacks()
        state = module.GarakAdapter._poll_kfp_run(
            _FakeClient(),
            "run-123",
            callbacks,
            poll_interval=0,
        )
        assert state == "SUCCEEDED"
        assert call_count["n"] == 3
        assert len(callbacks.statuses) == 2  # two RUNNING updates before SUCCEEDED

    def test_returns_on_failed(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)

        class _FakeClient:
            def get_run(self, run_id):
                return SimpleNamespace(state="FAILED")

        class _Callbacks:
            def report_status(self, update):
                pass

        state = module.GarakAdapter._poll_kfp_run(
            _FakeClient(),
            "run-456",
            _Callbacks(),
            poll_interval=0,
        )
        assert state == "FAILED"

    @pytest.mark.parametrize("terminal_state", ["SKIPPED", "CANCELED", "CANCELING"])
    def test_returns_immediately_on_other_terminal_states(self, monkeypatch, terminal_state):
        module = _load_evalhub_garak_adapter(monkeypatch)

        call_count = {"n": 0}

        class _FakeClient:
            def get_run(self, run_id):
                call_count["n"] += 1
                return SimpleNamespace(state=terminal_state)

        class _Callbacks:
            statuses = []

            def report_status(self, update):
                self.statuses.append(update)

        monkeypatch.setattr("time.sleep", lambda _: None)

        callbacks = _Callbacks()
        state = module.GarakAdapter._poll_kfp_run(
            _FakeClient(),
            "run-789",
            callbacks,
            poll_interval=0,
        )
        assert state == terminal_state
        assert call_count["n"] == 1

    def test_times_out_when_deadline_exceeded(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)

        clock = {"now": 0.0}
        monkeypatch.setattr("time.monotonic", lambda: clock["now"])
        monkeypatch.setattr("time.sleep", lambda _: None)

        class _FakeClient:
            def get_run(self, run_id):
                clock["now"] += 50
                return SimpleNamespace(state="RUNNING")

        class _Callbacks:
            def report_status(self, update):
                pass

        state = module.GarakAdapter._poll_kfp_run(
            _FakeClient(),
            "run-timeout",
            _Callbacks(),
            poll_interval=0,
            timeout=100,
        )
        assert state == "TIMED_OUT"


class TestJobPhaseReporting:
    """Tests for JobPhase lifecycle reporting added by the SDK contract alignment."""

    def test_run_simple_phase_sequence(self, monkeypatch, tmp_path):
        """_run_simple emits LOADING_DATA then RUNNING_EVALUATION."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()

        report_prefix = tmp_path / "scan"
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")

        def _fake_run(config_file, timeout_seconds, report_prefix, env=None, log_file=None):
            return module.GarakScanResult(
                returncode=0,
                stdout="",
                stderr="",
                report_prefix=report_prefix,
            )

        monkeypatch.setattr(module, "run_garak_scan", _fake_run)
        monkeypatch.setattr(module, "convert_to_avid_report", lambda _: True)

        statuses = []

        class _Callbacks:
            def report_status(self, update):
                statuses.append(update)

        job = SimpleNamespace(
            benchmark_id="trustyai_garak::quick",
            parameters={},
        )
        adapter._run_simple(job, _Callbacks(), {"run": {}}, tmp_path, timeout=60)

        phases = [s.phase for s in statuses]
        assert phases == [module.JobPhase.LOADING_DATA, module.JobPhase.RUNNING_EVALUATION]
        assert all(s.status == module.JobStatus.RUNNING for s in statuses)

    def test_run_simple_intents_phase_sequence(self, monkeypatch, tmp_path):
        """_run_simple_intents emits LOADING_DATA then RUNNING_EVALUATION."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()

        import pandas as pd

        fake_df = pd.DataFrame({"policy_concept": ["X"], "concept_definition": ["x"], "prompt": ["p"]})

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.core.pipeline_steps.resolve_taxonomy_data",
            lambda *a, **kw: fake_df,
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.core.pipeline_steps.run_sdg_generation",
            lambda **kw: fake_df,
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.core.pipeline_steps.normalize_prompts",
            lambda *a, **kw: fake_df,
        )

        report_prefix = tmp_path / "scan"
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.core.pipeline_steps.setup_and_run_garak",
            lambda **kw: module.GarakScanResult(
                returncode=0,
                stdout="",
                stderr="",
                report_prefix=report_prefix,
            ),
        )

        statuses = []

        class _Callbacks:
            def report_status(self, update):
                statuses.append(update)

        job = SimpleNamespace(
            benchmark_id="trustyai_garak::intents",
            parameters={},
            model=SimpleNamespace(url="http://localhost:8080", name="test-model"),
        )
        intents_params = {
            "sdg_model": "sdg-m",
            "sdg_api_base": "http://sdg:7000/v1",
        }
        adapter._run_simple_intents(job, _Callbacks(), {"run": {}}, tmp_path, 60, intents_params)

        phases = [s.phase for s in statuses]
        assert phases == [module.JobPhase.LOADING_DATA, module.JobPhase.RUNNING_EVALUATION]

    def test_run_via_kfp_phase_sequence(self, monkeypatch, tmp_path):
        """_run_via_kfp emits LOADING_DATA then RUNNING_EVALUATION."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        report_prefix = tmp_path / "kfp-job" / "scan"
        report_prefix.parent.mkdir(parents=True, exist_ok=True)
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")

        kfp_cfg = SimpleNamespace(
            endpoint="http://kfp:8080",
            namespace="ns",
            s3_secret_name="s3-secret",
            s3_prefix="prefix",
            s3_bucket="bucket",
            poll_interval_seconds=5,
            timeout_seconds=0,
            pipeline_root="",
            experiment_name="default",
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline.KFPConfig.from_env_and_config",
            staticmethod(lambda _cfg: kfp_cfg),
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline.evalhub_garak_pipeline",
            lambda **kw: None,
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.core.pipeline_steps.redact_api_keys",
            lambda d: d,
        )
        monkeypatch.setattr(
            module.GarakAdapter,
            "_create_kfp_client",
            lambda self, _cfg: SimpleNamespace(
                create_run_from_pipeline_func=lambda *a, **kw: SimpleNamespace(run_id="run-1"),
            ),
        )
        monkeypatch.setattr(
            module.GarakAdapter,
            "_poll_kfp_run",
            staticmethod(lambda *a, **kw: "SUCCEEDED"),
        )
        monkeypatch.setattr(
            module.GarakAdapter,
            "_read_s3_credentials_from_secret",
            staticmethod(lambda *a, **kw: {}),
        )
        monkeypatch.setattr(
            module.GarakAdapter,
            "_resolve_s3_credentials",
            staticmethod(
                lambda *a, **kw: {
                    "bucket": "test-bucket",
                    "endpoint_url": None,
                    "access_key": None,
                    "secret_key": None,
                    "region": None,
                }
            ),
        )
        monkeypatch.setattr(
            module.GarakAdapter,
            "_download_results_from_s3",
            lambda self, *a, **kw: None,
        )

        statuses = []

        class _Callbacks:
            def report_status(self, update):
                statuses.append(update)

        job = SimpleNamespace(
            id="kfp-job",
            benchmark_id="trustyai_garak::quick",
            model=SimpleNamespace(url="http://localhost:8000", name="m"),
            parameters={},
        )
        adapter._run_via_kfp(job, _Callbacks(), {"run": {}}, timeout=60)

        phases = [s.phase for s in statuses]
        assert phases == [module.JobPhase.LOADING_DATA, module.JobPhase.RUNNING_EVALUATION]

    @pytest.mark.parametrize("save_samples", [None, False, True])
    def test_full_lifecycle_with_oci(self, monkeypatch, tmp_path, save_samples):
        """Full run_benchmark_job with OCI exports emits the complete phase sequence."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        scan_directories = []

        def _fake_run(config_file, timeout_seconds, report_prefix, env=None, log_file=None):
            scan_directories.append(report_prefix.parent)
            report_prefix.with_suffix(".report.jsonl").write_text(
                json.dumps(
                    {
                        "entry_type": "attempt",
                        "status": 2,
                        "uuid": "attempt",
                        "probe_classname": "dan.Test",
                        "outputs": ["yes", "no"],
                        "detector_results": {"dan.DAN": [1, 0]},
                    }
                ),
                encoding="utf-8",
            )
            return module.GarakScanResult(returncode=0, stdout="", stderr="", report_prefix=report_prefix)

        monkeypatch.setattr(module, "run_garak_scan", _fake_run)
        monkeypatch.setattr(module, "convert_to_avid_report", lambda _: True)
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        statuses = []
        mlflow_files = []
        mlflow_module = types.ModuleType("evalhub.adapter.mlflow")
        mlflow_module.MlflowArtifact = lambda name, data, media: SimpleNamespace(name=name, data=data)
        monkeypatch.setitem(sys.modules, "evalhub.adapter.mlflow", mlflow_module)

        class _Callbacks:
            mlflow = SimpleNamespace(save=lambda results, config, artifacts: mlflow_files.extend(artifacts))

            def report_status(self, update):
                statuses.append(update)

            def create_oci_artifact(self, _spec):
                # The export must exist before OCI snapshots the scan directory.
                assert (scan_directories[-1] / "sample_results.jsonl").exists() == (
                    job.parameters.get("save_sample_results", True)
                )
                return SimpleNamespace(reference="oci://ref", digest="sha256:test")

            def report_results(self, _results):
                return None

        job = SimpleNamespace(
            id="oci-job",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
            parameters={} if save_samples is None else {"save_sample_results": save_samples},
            exports=SimpleNamespace(
                oci=SimpleNamespace(coordinates="quay.io/test/artifact:latest"),
            ),
        )
        adapter.run_benchmark_job(job, _Callbacks())

        assert ("sample_results.jsonl" in [a.name for a in mlflow_files]) == (save_samples is not False)
        first_directory = scan_directories[-1]
        statuses.clear()
        job.parameters = {"save_sample_results": False}
        adapter.run_benchmark_job(job, _Callbacks())
        assert scan_directories[-1] != first_directory
        assert not (scan_directories[-1] / "sample_results.jsonl").exists()
        phases = [s.phase for s in statuses if hasattr(s, "phase")]
        assert phases == [
            module.JobPhase.INITIALIZING,
            module.JobPhase.LOADING_DATA,
            module.JobPhase.RUNNING_EVALUATION,
            module.JobPhase.POST_PROCESSING,
            module.JobPhase.PERSISTING_ARTIFACTS,
        ]

    def test_full_lifecycle_without_oci(self, monkeypatch, tmp_path):
        """Without OCI exports, PERSISTING_ARTIFACTS is not emitted."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        report_prefix = tmp_path / "no-oci-job" / "scan"
        report_prefix.parent.mkdir(parents=True, exist_ok=True)
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")

        def _fake_run(config_file, timeout_seconds, report_prefix, env=None, log_file=None):
            return module.GarakScanResult(
                returncode=0,
                stdout="",
                stderr="",
                report_prefix=report_prefix,
            )

        monkeypatch.setattr(module, "run_garak_scan", _fake_run)
        monkeypatch.setattr(module, "convert_to_avid_report", lambda _: True)
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        statuses = []

        class _Callbacks:
            def report_status(self, update):
                statuses.append(update)

            def report_results(self, _results):
                return None

        job = SimpleNamespace(
            id="no-oci-job",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
            parameters={
                "save_sample_results": False,
            },
            exports=None,
        )
        adapter.run_benchmark_job(job, _Callbacks())

        phases = [s.phase for s in statuses if hasattr(s, "phase")]
        assert module.JobPhase.PERSISTING_ARTIFACTS not in phases
        assert phases == [
            module.JobPhase.INITIALIZING,
            module.JobPhase.LOADING_DATA,
            module.JobPhase.RUNNING_EVALUATION,
            module.JobPhase.POST_PROCESSING,
        ]

    def test_error_uses_message_info_not_error_info(self, monkeypatch, tmp_path):
        """Scan failure reports status=FAILED with error_message (not error)."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        report_prefix = tmp_path / "err-job" / "scan"
        report_prefix.parent.mkdir(parents=True, exist_ok=True)
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")

        def _fake_run(config_file, timeout_seconds, report_prefix, env=None, log_file=None):
            return module.GarakScanResult(
                returncode=1,
                stdout="",
                stderr="boom",
                report_prefix=report_prefix,
            )

        monkeypatch.setattr(module, "run_garak_scan", _fake_run)
        monkeypatch.setattr(module, "convert_to_avid_report", lambda _: True)

        statuses = []

        class _Callbacks:
            def report_status(self, update):
                statuses.append(update)

            def report_results(self, _results):
                return None

        job = SimpleNamespace(
            id="err-job",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
            parameters={},
            exports=None,
        )

        with pytest.raises(RuntimeError, match="boom"):
            adapter.run_benchmark_job(job, _Callbacks())

        failed = [s for s in statuses if s.status == module.JobStatus.FAILED]
        assert len(failed) == 2
        # First FAILED: scan_failed (from the result.success check)
        assert "boom" in failed[0].error_message.message
        assert failed[0].error_message.message_code == "scan_failed"
        # Second FAILED: job_failed (from the outer exception handler)
        assert failed[1].error_message.message_code == "job_failed"


class TestResolveExecutionModeNonString:
    """Test that non-string execution_mode values are handled safely."""

    def test_bool_execution_mode_falls_back_to_simple(self, monkeypatch):
        monkeypatch.delenv("EVALHUB_EXECUTION_MODE", raising=False)
        module = _load_evalhub_garak_adapter(monkeypatch)
        assert module.GarakAdapter._resolve_execution_mode({"execution_mode": True}) == "simple"

    def test_int_execution_mode_falls_back_to_simple(self, monkeypatch):
        monkeypatch.delenv("EVALHUB_EXECUTION_MODE", raising=False)
        module = _load_evalhub_garak_adapter(monkeypatch)
        assert module.GarakAdapter._resolve_execution_mode({"execution_mode": 42}) == "simple"


class TestKFPMissingS3Secret:
    """Test that missing s3_secret_name is caught early."""

    def test_run_via_kfp_raises_without_s3_secret(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))
        monkeypatch.setenv("KFP_ENDPOINT", "https://kfp.example.com")
        monkeypatch.setenv("KFP_NAMESPACE", "test-ns")
        monkeypatch.delenv("KFP_S3_SECRET_NAME", raising=False)

        class _Callbacks:
            def report_status(self, _update):
                return None

        job = SimpleNamespace(
            id="kfp-no-secret",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
            parameters={"execution_mode": "kfp"},
            exports=None,
        )

        with pytest.raises(ValueError) as excinfo:
            adapter.run_benchmark_job(job, _Callbacks())

        msg = str(excinfo.value)
        assert "KFP_S3_SECRET_NAME" in msg
        assert "kfp_config.s3_secret_name" in msg


# ---------------------------------------------------------------------------
# Intents tests
# ---------------------------------------------------------------------------

_INTENTS_MODELS_SINGLE = {
    "intents_models": {
        "judge": {"url": "http://judge:8000/v1", "name": "judge-model"},
    }
}

_INTENTS_MODELS_ALL_ROLES = {
    "intents_models": {
        "judge": {"url": "http://judge:8000/v1", "name": "judge-model", "api_key": "jk"},
        "attacker": {"url": "http://attacker:9000/v1", "name": "atk-model", "api_key": "ak"},
        "evaluator": {"url": "http://evaluator:9001/v1", "name": "eval-model", "api_key": "ek"},
        "sdg": {"url": "http://sdg:7000/v1", "name": "sdg-model", "api_key": "sk"},
    }
}


class TestBuildConfigIntentsOverrides:
    """Tests for intents-specific config overrides in _build_config_from_spec."""

    def test_intents_profile_returns_art_intents_true(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="intents-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={**_INTENTS_MODELS_SINGLE},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        _, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        assert intents_params["art_intents"] is True
        assert intents_params["sdg_flow_id"] == "major-sage-742"

    def test_single_role_fills_all_three(self, monkeypatch, tmp_path):
        """When only one of judge/attacker/evaluator is provided, all roles use it."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="intents-single-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={**_INTENTS_MODELS_SINGLE},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        judge = config_dict["plugins"]["detectors"]["judge"]
        assert judge["detector_model_name"] == "judge-model"
        assert judge["detector_model_config"]["uri"] == "http://judge:8000/v1"

        tap_cfg = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]
        assert tap_cfg["attack_model_name"] == "judge-model"
        assert tap_cfg["attack_model_config"]["uri"] == "http://judge:8000/v1"
        assert tap_cfg["evaluator_model_name"] == "judge-model"
        assert tap_cfg["evaluator_model_config"]["uri"] == "http://judge:8000/v1"

    def test_single_attacker_fills_all_three(self, monkeypatch, tmp_path):
        """Providing only 'attacker' should populate judge and evaluator too."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="intents-atk-only-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                "intents_models": {
                    "attacker": {"url": "http://atk-model:9000/v1", "name": "atk-model"},
                }
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        judge = config_dict["plugins"]["detectors"]["judge"]
        assert judge["detector_model_name"] == "atk-model"
        assert judge["detector_model_config"]["uri"] == "http://atk-model:9000/v1"

        tap_cfg = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]
        assert tap_cfg["attack_model_name"] == "atk-model"
        assert tap_cfg["evaluator_model_name"] == "atk-model"

    def test_separate_models_per_role(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="multi-model-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={**_INTENTS_MODELS_ALL_ROLES},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        judge = config_dict["plugins"]["detectors"]["judge"]
        assert judge["detector_model_name"] == "judge-model"
        assert judge["detector_model_config"]["uri"] == "http://judge:8000/v1"
        assert judge["detector_model_config"]["api_key"] == "__FROM_ENV__"

        tap_cfg = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]
        assert tap_cfg["attack_model_name"] == "atk-model"
        assert tap_cfg["attack_model_config"]["uri"] == "http://attacker:9000/v1"
        assert tap_cfg["attack_model_config"]["api_key"] == "__FROM_ENV__"
        assert tap_cfg["evaluator_model_name"] == "eval-model"
        assert tap_cfg["evaluator_model_config"]["uri"] == "http://evaluator:9001/v1"
        assert tap_cfg["evaluator_model_config"]["api_key"] == "__FROM_ENV__"

        assert intents_params["sdg_model"] == "sdg-model"
        assert intents_params["sdg_api_base"] == "http://sdg:7000/v1"

    def test_no_roles_raises_clear_error(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="no-roles-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        with pytest.raises(ValueError, match="requires model configuration"):
            adapter._build_config_from_spec(job, report_prefix)

    def test_two_roles_raises_ambiguous_error(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="two-roles-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={
                "intents_models": {
                    "judge": {"url": "http://judge:8000/v1", "name": "judge-m"},
                    "attacker": {"url": "http://atk:9000/v1", "name": "atk-m"},
                }
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        with pytest.raises(ValueError, match="Ambiguous"):
            adapter._build_config_from_spec(job, report_prefix)

    def test_missing_name_raises_clear_error(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="no-name-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={
                "intents_models": {"judge": {"url": "http://judge:8000/v1"}},
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        with pytest.raises(ValueError, match="intents_models.judge.name"):
            adapter._build_config_from_spec(job, report_prefix)

    # ------------------------------------------------------------------
    # Merge preservation tests (intents_models + garak_config overrides)
    # ------------------------------------------------------------------

    def test_multiclass_judge_config_preserved_with_intents_models(self, monkeypatch, tmp_path):
        """MulticlassJudge sub-config in garak_config survives intents_models override."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="mcj-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                **_INTENTS_MODELS_ALL_ROLES,
                "garak_config": {
                    "plugins": {
                        "detectors": {
                            "judge": {
                                "MulticlassJudge": {
                                    "system_prompt": "Custom safety evaluator",
                                    "score_key": "complied",
                                    "score_scale": 100,
                                    "confidence_cutoff": 70,
                                }
                            }
                        }
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        judge = config_dict["plugins"]["detectors"]["judge"]
        assert judge["detector_model_name"] == "judge-model"
        assert judge["detector_model_config"]["uri"] == "http://judge:8000/v1"
        assert "MulticlassJudge" in judge
        assert judge["MulticlassJudge"]["system_prompt"] == "Custom safety evaluator"
        assert judge["MulticlassJudge"]["score_key"] == "complied"
        assert judge["MulticlassJudge"]["confidence_cutoff"] == 70

    def test_extra_attack_evaluator_config_keys_preserved(self, monkeypatch, tmp_path):
        """Extra keys in attack/evaluator model configs survive intents_models override."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="extra-keys-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                **_INTENTS_MODELS_ALL_ROLES,
                "garak_config": {
                    "plugins": {
                        "probes": {
                            "tap": {
                                "TAPIntent": {
                                    "attack_model_config": {
                                        "temperature": 0.9,
                                        "top_p": 0.95,
                                    },
                                    "evaluator_model_config": {
                                        "top_p": 0.8,
                                        "presence_penalty": 0.5,
                                    },
                                }
                            }
                        }
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        tap = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]

        atk = tap["attack_model_config"]
        assert atk["uri"] == "http://attacker:9000/v1"
        assert atk["api_key"] == "__FROM_ENV__"
        assert atk["temperature"] == 0.9
        assert atk["top_p"] == 0.95

        ev = tap["evaluator_model_config"]
        assert ev["uri"] == "http://evaluator:9001/v1"
        assert ev["api_key"] == "__FROM_ENV__"
        assert ev["top_p"] == 0.8
        assert ev["presence_penalty"] == 0.5

    # ------------------------------------------------------------------
    # garak_config-only mode (no intents_models)
    # ------------------------------------------------------------------

    def test_garak_config_only_skips_intents_models_override(self, monkeypatch, tmp_path):
        """Full model config in garak_config works without intents_models."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="gc-only-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                "sdg_model": "my-sdg-model",
                "sdg_api_base": "http://sdg:7000/v1",
                "garak_config": {
                    "plugins": {
                        "detectors": {
                            "judge": {
                                "detector_model_name": "my-judge",
                                "detector_model_config": {
                                    "uri": "http://judge:8000/v1",
                                    "api_key": "sk-judge",
                                },
                                "MulticlassJudge": {
                                    "system_prompt": "Custom prompt",
                                },
                            }
                        },
                        "probes": {
                            "tap": {
                                "TAPIntent": {
                                    "attack_model_name": "my-atk",
                                    "attack_model_config": {
                                        "uri": "http://atk:9000/v1",
                                        "api_key": "sk-atk",
                                    },
                                    "evaluator_model_name": "my-eval",
                                    "evaluator_model_config": {
                                        "uri": "http://eval:9001/v1",
                                        "api_key": "sk-eval",
                                    },
                                }
                            }
                        },
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        judge = config_dict["plugins"]["detectors"]["judge"]
        assert judge["detector_model_name"] == "my-judge"
        assert judge["detector_model_config"]["uri"] == "http://judge:8000/v1"
        assert judge["detector_model_config"]["api_key"] == "sk-judge"
        assert judge["MulticlassJudge"]["system_prompt"] == "Custom prompt"

        tap = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]
        assert tap["attack_model_name"] == "my-atk"
        assert tap["attack_model_config"]["uri"] == "http://atk:9000/v1"
        assert tap["attack_model_config"]["api_key"] == "sk-atk"
        assert tap["evaluator_model_name"] == "my-eval"
        assert tap["evaluator_model_config"]["uri"] == "http://eval:9001/v1"
        assert tap["evaluator_model_config"]["api_key"] == "sk-eval"

        assert intents_params["sdg_model"] == "my-sdg-model"
        assert intents_params["sdg_api_base"] == "http://sdg:7000/v1"

    def test_partial_garak_config_missing_tap_raises(self, monkeypatch, tmp_path):
        """Judge configured in garak_config but TAP models missing raises error."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="partial-gc-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                "garak_config": {
                    "plugins": {
                        "detectors": {
                            "judge": {
                                "detector_model_name": "my-judge",
                                "detector_model_config": {"uri": "http://judge:8000/v1"},
                            }
                        }
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        with pytest.raises(ValueError, match="requires model configuration"):
            adapter._build_config_from_spec(job, report_prefix)

    # ------------------------------------------------------------------
    # Deep-merge granularity tests: override a single leaf without
    # clobbering siblings or ancestor dicts
    # ------------------------------------------------------------------

    def test_single_tapintent_field_override_preserves_profile_defaults(self, monkeypatch, tmp_path):
        """Override only ``depth`` in TAPIntent; all other profile defaults survive deep-merge."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="deep-tap-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                **_INTENTS_MODELS_ALL_ROLES,
                "garak_config": {
                    "plugins": {
                        "probes": {
                            "tap": {
                                "TAPIntent": {
                                    "depth": 20,
                                }
                            }
                        }
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        tap = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]
        assert tap["depth"] == 20, "User override for depth must take effect"
        assert tap["width"] == 10, "Profile default width must survive"
        assert tap["attack_max_attempts"] == 5, "Profile default attack_max_attempts must survive"
        assert tap["branching_factor"] == 4, "Profile default branching_factor must survive"
        assert tap["pruning"] is True, "Profile default pruning must survive"
        assert tap["attack_model_config"]["uri"] == "http://attacker:9000/v1"

    def test_extra_keys_in_detector_model_config_preserved(self, monkeypatch, tmp_path):
        """Extra keys in ``detector_model_config`` survive ``_apply_intents_models`` overlay."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="det-cfg-extra-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                **_INTENTS_MODELS_ALL_ROLES,
                "garak_config": {
                    "plugins": {
                        "detectors": {
                            "judge": {
                                "detector_model_config": {
                                    "max_tokens": 500,
                                    "temperature": 0.1,
                                }
                            }
                        }
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        det_cfg = config_dict["plugins"]["detectors"]["judge"]["detector_model_config"]
        assert det_cfg["uri"] == "http://judge:8000/v1", "intents_models uri must be injected"
        assert det_cfg["api_key"] == "__FROM_ENV__", "placeholder api_key must be injected"
        assert det_cfg["max_tokens"] == 500, "User-provided max_tokens must survive overlay"
        assert det_cfg["temperature"] == 0.1, "User-provided temperature must survive overlay"

    def test_single_multiclass_judge_field_override(self, monkeypatch, tmp_path):
        """Override only ``system_prompt`` in MulticlassJudge via garak_config;
        other MulticlassJudge fields provided alongside must survive."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="mcj-partial-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                **_INTENTS_MODELS_ALL_ROLES,
                "garak_config": {
                    "plugins": {
                        "detectors": {
                            "judge": {
                                "MulticlassJudge": {
                                    "system_prompt": "Custom evaluator prompt",
                                    "score_key": "complied",
                                    "score_scale": 100,
                                    "confidence_cutoff": 70,
                                }
                            }
                        }
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        judge = config_dict["plugins"]["detectors"]["judge"]
        assert judge["detector_model_name"] == "judge-model"
        assert judge["detector_model_config"]["uri"] == "http://judge:8000/v1"
        mcj = judge["MulticlassJudge"]
        assert mcj["system_prompt"] == "Custom evaluator prompt"
        assert mcj["score_key"] == "complied"
        assert mcj["score_scale"] == 100
        assert mcj["confidence_cutoff"] == 70

    def test_attack_model_config_extra_keys_survive_overlay(self, monkeypatch, tmp_path):
        """Profile's ``max_tokens`` in ``attack_model_config`` survives ``_apply_intents_models``."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="atk-max-tokens-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={**_INTENTS_MODELS_ALL_ROLES},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        atk = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]["attack_model_config"]
        assert atk["uri"] == "http://attacker:9000/v1"
        assert atk["api_key"] == "__FROM_ENV__"
        assert atk["max_tokens"] == 500, "Profile's max_tokens must survive intents_models overlay"

        ev = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]["evaluator_model_config"]
        assert ev["uri"] == "http://evaluator:9001/v1"
        assert ev["api_key"] == "__FROM_ENV__"
        assert ev["max_tokens"] == 10, "Profile's max_tokens must survive intents_models overlay"
        assert ev["temperature"] == 0.0, "Profile's temperature must survive intents_models overlay"

    def test_non_intents_profile_returns_art_intents_false(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="native-job",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        _, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        assert intents_params["art_intents"] is False

    def test_benchmark_config_art_intents_false_overrides_profile(self, monkeypatch, tmp_path):
        """Explicit art_intents=False in benchmark_config wins over profile (intents profile sets True)."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="override-false-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={"art_intents": False},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        _, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        assert intents_params["art_intents"] is False

    def test_benchmark_config_art_intents_true_overrides_profile(self, monkeypatch, tmp_path):
        """Explicit art_intents=True in benchmark_config wins over non-intents profile."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="override-true-job",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={
                "art_intents": True,
                **_INTENTS_MODELS_SINGLE,
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        _, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        assert intents_params["art_intents"] is True

    def test_sdg_params_from_intents_models(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="sdg-intents-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={
                **_INTENTS_MODELS_SINGLE,
                "intents_models": {
                    **_INTENTS_MODELS_SINGLE["intents_models"],
                    "sdg": {"url": "http://sdg:7000/v1", "name": "sdg-model", "api_key": "sdg-key"},
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        _, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        assert intents_params["sdg_model"] == "sdg-model"
        assert intents_params["sdg_api_base"] == "http://sdg:7000/v1"

    def test_sdg_fallback_to_flat_keys(self, monkeypatch, tmp_path):
        """When intents_models.sdg is absent, fall back to flat sdg_model/sdg_api_base."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="sdg-flat-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={
                **_INTENTS_MODELS_SINGLE,
                "sdg_model": "legacy-sdg",
                "sdg_api_base": "http://legacy-sdg:5000",
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        _, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        assert intents_params["sdg_model"] == "legacy-sdg"
        assert intents_params["sdg_api_base"] == "http://legacy-sdg:5000"


class TestTargetDefaultParameters:
    """Tests for TARGET_DEFAULT_PARAMETERS fallback (max_tokens=512)."""

    def test_default_max_tokens_when_no_model_parameters(self, monkeypatch, tmp_path):
        """When neither config.model.parameters nor benchmark_config model_parameters
        are provided, the target generator should use TARGET_DEFAULT_PARAMETERS."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="no-params-job",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        gen = config_dict["plugins"]["generators"]["openai"]["OpenAICompatible"]
        assert gen["max_tokens"] == 512

    def test_explicit_model_parameters_not_overwritten(self, monkeypatch, tmp_path):
        """When config.model.parameters is set, those values are used as-is."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="explicit-params-job",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(
                url="http://model:8000",
                name="my-llm",
                parameters={"max_tokens": 1024, "temperature": 0.7},
            ),
            parameters={},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        gen = config_dict["plugins"]["generators"]["openai"]["OpenAICompatible"]
        assert gen["max_tokens"] == 1024
        assert gen["temperature"] == 0.7

    def test_benchmark_config_model_parameters_used(self, monkeypatch, tmp_path):
        """When model has no parameters attr but benchmark_config has model_parameters,
        those are used instead of the default."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="bench-params-job",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={"model_parameters": {"max_tokens": 256}},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        gen = config_dict["plugins"]["generators"]["openai"]["OpenAICompatible"]
        assert gen["max_tokens"] == 256
        assert "temperature" not in gen


class TestTranslationLangproviders:
    """Tests for translation langprovider resolution in _build_config_from_spec."""

    def test_default_uses_attacker_llm(self, monkeypatch, tmp_path):
        """Default: single-role intents_models -> attacker LLM reused for translation."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="trans-default-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={**_INTENTS_MODELS_SINGLE},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        langproviders = config_dict["run"]["langproviders"]
        assert len(langproviders) == 2
        assert langproviders[0]["model_type"] == "llm"
        assert langproviders[0]["translation_model_config"]["uri"] == "http://judge:8000/v1"
        assert langproviders[0]["translation_model_name"] == "judge-model"
        assert langproviders[0]["translation_model_config"]["api_key"] == "__FROM_ENV__"
        assert langproviders[1]["language"] == "en,zh"

    def test_separate_attacker_used_for_translation(self, monkeypatch, tmp_path):
        """When all 3 roles provided, attacker url/name are used for translation."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="trans-atk-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={**_INTENTS_MODELS_ALL_ROLES},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        langproviders = config_dict["run"]["langproviders"]
        assert langproviders[0]["model_type"] == "llm"
        assert langproviders[0]["translation_model_config"]["uri"] == "http://attacker:9000/v1"
        assert langproviders[0]["translation_model_name"] == "atk-model"

    def test_dedicated_translation_model(self, monkeypatch, tmp_path):
        """intents_models.translation takes priority over attacker."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="trans-dedicated-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                "intents_models": {
                    **_INTENTS_MODELS_ALL_ROLES["intents_models"],
                    "translation": {"url": "http://translator:6000/v1", "name": "translator-llm"},
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        langproviders = config_dict["run"]["langproviders"]
        assert langproviders[0]["model_type"] == "llm"
        assert langproviders[0]["translation_model_config"]["uri"] == "http://translator:6000/v1"
        assert langproviders[0]["translation_model_name"] == "translator-llm"

    def test_translation_use_hf_flag(self, monkeypatch, tmp_path):
        """translation_use_hf=True forces HF models even when attacker is available."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="trans-hf-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                **_INTENTS_MODELS_ALL_ROLES,
                "translation_use_hf": True,
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        langproviders = config_dict["run"]["langproviders"]
        assert len(langproviders) == 2
        assert langproviders[0]["model_type"] == "local.LocalHFTranslator"
        assert langproviders[0]["model_name"] == "Helsinki-NLP/opus-mt-zh-en"
        assert "api_key" not in langproviders[0]

    def test_non_intents_profile_no_langproviders(self, monkeypatch, tmp_path):
        """Non-intents profiles should not get langproviders injected."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="native-no-lp-job",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        assert "langproviders" not in config_dict.get("run", {})

    def test_no_langproviders_when_translation_probe_excluded(self, monkeypatch, tmp_path):
        """When probe_spec overrides remove TranslationIntent, langproviders are not set."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="trans-excluded-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                **_INTENTS_MODELS_ALL_ROLES,
                "garak_config": {
                    "plugins": {
                        "probe_spec": "spo.SPOIntent,tap.TAPIntent",
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        assert "langproviders" not in config_dict.get("run", {})
        assert config_dict["run"]["spec"]["include"] == [
            "probes.spo.SPOIntent",
            "probes.tap.TAPIntent",
            {"intent": "all"},
        ]

    def test_list_probe_spec_with_translation_injects_langproviders(self, monkeypatch, tmp_path):
        """When probe_spec is a list containing TranslationIntent, langproviders are injected."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="list-probe-with-trans-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                **_INTENTS_MODELS_ALL_ROLES,
                "garak_config": {
                    "plugins": {
                        "probe_spec": ["spo.SPOIntent", "multilingual.TranslationIntent", "tap.TAPIntent"],
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        langproviders = config_dict["run"]["langproviders"]
        assert len(langproviders) == 2
        assert langproviders[0]["model_type"] == "llm"

    def test_list_probe_spec_without_translation_skips_langproviders(self, monkeypatch, tmp_path):
        """When probe_spec is a list without TranslationIntent, langproviders are not set."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="list-probe-no-trans-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                **_INTENTS_MODELS_ALL_ROLES,
                "garak_config": {
                    "plugins": {
                        "probe_spec": ["spo.SPOIntent", "tap.TAPIntent"],
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        assert "langproviders" not in config_dict.get("run", {})

    def test_intents_profile_no_hardcoded_hf_langproviders(self, monkeypatch, tmp_path):
        """The intents profile should no longer contain hardcoded HF langproviders."""
        from llama_stack_provider_trustyai_garak.core.config_resolution import resolve_scan_profile

        profile = resolve_scan_profile("trustyai_garak::intents")
        garak_config = profile.get("garak_config", {})
        run_config = garak_config.get("run", {})
        assert "langproviders" not in run_config


class TestReadRoleFromSecret:
    """Tests for _read_role_from_secret static method."""

    def _setup(self, monkeypatch, fake_read):
        module = _load_evalhub_garak_adapter(monkeypatch)
        auth_module = sys.modules["evalhub.adapter.auth"]
        auth_module.read_model_auth_key = fake_read
        return module

    def test_both_url_and_key_from_secret(self, monkeypatch):
        def _fake(name):
            return {"judge_url": "http://localhost:8080", "judge_api-key": "judge_api-key:ref"}.get(name)

        module = self._setup(monkeypatch, _fake)
        url, key = module.GarakAdapter._read_role_from_secret("judge", "http://fallback:8080")
        assert url == "http://localhost:8080"
        assert key == "judge_api-key:ref"

    def test_key_present_url_absent_uses_model_url(self, monkeypatch):
        def _fake(name):
            return {"judge_api-key": "judge_api-key:ref"}.get(name)

        module = self._setup(monkeypatch, _fake)
        url, key = module.GarakAdapter._read_role_from_secret("judge", "http://localhost:8080")
        assert url == "http://localhost:8080"
        assert key == "judge_api-key:ref"

    def test_url_present_key_absent_uses_generic_api_key(self, monkeypatch):
        def _fake(name):
            return {"judge_url": "http://localhost:8080", "api-key": "api-key:ref"}.get(name)

        module = self._setup(monkeypatch, _fake)
        url, key = module.GarakAdapter._read_role_from_secret("judge", "http://fallback:8080")
        assert url == "http://localhost:8080"
        assert key == "api-key:ref"

    def test_neither_url_nor_key_returns_none(self, monkeypatch):
        module = self._setup(monkeypatch, lambda name: None)
        url, key = module.GarakAdapter._read_role_from_secret("judge", "http://localhost:8080")
        assert url is None
        assert key is None

    def test_url_present_no_generic_key_returns_none_for_key(self, monkeypatch):
        def _fake(name):
            return {"sdg_url": "http://localhost:8080"}.get(name)

        module = self._setup(monkeypatch, _fake)
        url, key = module.GarakAdapter._read_role_from_secret("sdg", "http://fallback:8080")
        assert url == "http://localhost:8080"
        assert key is None

    def test_secret_url_wins_over_job_request_url(self, monkeypatch):
        def _fake(name):
            return {"judge_url": "http://localhost:8080", "judge_api-key": "judge_api-key:ref"}.get(name)

        module = self._setup(monkeypatch, _fake)
        url, key = module.GarakAdapter._read_role_from_secret("judge", "http://real-model:9000/v1")
        assert url == "http://localhost:8080"
        assert key == "judge_api-key:ref"


class TestResolveIntentsApiKey:
    """Tests for _resolve_intents_api_key static method."""

    def test_direct_api_key_takes_priority(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.setenv("JUDGE_API_KEY", "from-env")

        result = module.GarakAdapter._resolve_intents_api_key(
            "judge",
            {"api_key": "direct-key"},
        )
        assert result == "direct-key"

    def test_api_key_env_from_config(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.setenv("MY_CUSTOM_KEY", "custom-val")

        result = module.GarakAdapter._resolve_intents_api_key(
            "judge",
            {"api_key_env": "MY_CUSTOM_KEY"},
        )
        assert result == "custom-val"

    def test_convention_env_var(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.setenv("JUDGE_API_KEY", "conv-key")

        result = module.GarakAdapter._resolve_intents_api_key("judge", {})
        assert result == "conv-key"

    def test_generic_env_fallback(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        monkeypatch.setenv("OPENAICOMPATIBLE_API_KEY", "generic-key")

        result = module.GarakAdapter._resolve_intents_api_key("judge", {})
        assert result == "generic-key"

    def test_returns_dummy_when_nothing_found(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAICOMPATIBLE_API_KEY", raising=False)

        result = module.GarakAdapter._resolve_intents_api_key("judge", {})
        assert result == "DUMMY"

    def _install_auth_stub(self, monkeypatch, fake_read_fn):
        """Install a stub evalhub.adapter.auth module with a fake read_model_auth_key."""
        auth_module = types.ModuleType("evalhub.adapter.auth")
        auth_module.read_model_auth_key = fake_read_fn
        monkeypatch.setitem(sys.modules, "evalhub.adapter.auth", auth_module)

    def test_api_key_name_takes_precedence_over_env_vars(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.setenv("JUDGE_API_KEY", "env-judge")
        monkeypatch.setenv("OPENAICOMPATIBLE_API_KEY", "env-openai")

        calls = []

        def fake_read(name):
            calls.append(name)
            return "secret-from-custom" if name == "custom-secret" else None

        self._install_auth_stub(monkeypatch, fake_read)

        result = module.GarakAdapter._resolve_intents_api_key(
            "judge",
            {"api_key_name": "custom-secret"},
        )
        assert result == "secret-from-custom"
        assert calls == ["custom-secret"]

    def test_role_specific_secret_when_no_explicit_key_or_env(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAICOMPATIBLE_API_KEY", raising=False)

        def fake_read(name):
            return "secret-judge" if name == "judge-api-key" else None

        self._install_auth_stub(monkeypatch, fake_read)

        result = module.GarakAdapter._resolve_intents_api_key("judge", {})
        assert result == "secret-judge"

    def test_generic_secret_when_role_specific_missing(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAICOMPATIBLE_API_KEY", raising=False)

        def fake_read(name):
            return "generic-secret" if name == "api-key" else None

        self._install_auth_stub(monkeypatch, fake_read)

        result = module.GarakAdapter._resolve_intents_api_key("judge", {})
        assert result == "generic-secret"

    def test_api_key_env_takes_precedence_over_role_specific_env(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.setenv("JUDGE_API_KEY", "env-judge")
        monkeypatch.setenv("CUSTOM_KEY", "env-custom")

        result = module.GarakAdapter._resolve_intents_api_key(
            "judge",
            {"api_key_env": "CUSTOM_KEY"},
        )
        assert result == "env-custom"

    def test_direct_api_key_wins_over_secret(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)

        def fake_read(name):
            return "should-not-be-used"

        self._install_auth_stub(monkeypatch, fake_read)

        result = module.GarakAdapter._resolve_intents_api_key(
            "judge",
            {"api_key": "direct-wins", "api_key_name": "some-secret"},
        )
        assert result == "direct-wins"


class TestSimpleIntentsMode:
    """Tests for intents benchmark running in simple (non-KFP) mode."""

    def _make_adapter_and_job(self, monkeypatch, tmp_path, parameters=None):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="simple-intents-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                "save_sample_results": False,
                "execution_mode": "simple",
                **_INTENTS_MODELS_ALL_ROLES,
                **(parameters or {}),
            },
            exports=None,
        )
        return module, adapter, job

    def test_simple_intents_dispatches_to_run_simple_intents(self, monkeypatch, tmp_path):
        """Verify art_intents + simple mode routes to _run_simple_intents."""
        module, adapter, job = self._make_adapter_and_job(monkeypatch, tmp_path)

        captured = {}
        report_prefix = tmp_path / "simple-intents-job" / "scan"
        report_prefix.parent.mkdir(parents=True, exist_ok=True)
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")

        def _fake_run_simple_intents(self, config, callbacks, garak_config_dict, scan_dir, timeout, intents_params):
            captured["called"] = True
            captured["intents_params"] = intents_params
            return module.GarakScanResult(
                returncode=0,
                stdout="",
                stderr="",
                report_prefix=report_prefix,
            )

        monkeypatch.setattr(module.GarakAdapter, "_run_simple_intents", _fake_run_simple_intents)
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        class _Callbacks:
            def report_status(self, _update):
                return None

        adapter.run_benchmark_job(job, _Callbacks())
        assert captured.get("called") is True
        assert captured["intents_params"]["art_intents"] is True

    def test_simple_intents_default_taxonomy_and_sdg(self, monkeypatch, tmp_path):
        """Full pipeline: default taxonomy -> SDG -> normalize -> scan."""
        module, adapter, job = self._make_adapter_and_job(monkeypatch, tmp_path)

        scan_dir = tmp_path / "simple-intents-job"
        scan_dir.mkdir(parents=True, exist_ok=True)

        import pandas as pd

        fake_raw_df = pd.DataFrame(
            {
                "policy_concept": ["Fraud", "Fraud"],
                "concept_definition": ["Fraud prompts", "Fraud prompts"],
                "prompt": ["Commit fraud?", "How to scam?"],
            }
        )

        sdg_captured = {}

        def _fake_run_sdg_generation(taxonomy_df, sdg_model, sdg_api_base, **kwargs):
            sdg_captured["taxonomy_len"] = len(taxonomy_df)
            sdg_captured["sdg_model"] = sdg_model
            sdg_captured["sdg_api_base"] = sdg_api_base
            sdg_captured.update(kwargs)
            return fake_raw_df

        def _fake_setup_and_run(config_json, prompts_csv_path, scan_dir, timeout_seconds):
            sdg_captured["prompts_csv_path"] = str(prompts_csv_path)
            sdg_captured["config_json"] = config_json
            report_prefix = scan_dir / "scan"
            report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")
            return module.GarakScanResult(
                returncode=0,
                stdout="",
                stderr="",
                report_prefix=report_prefix,
            )

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.core.pipeline_steps.run_sdg_generation",
            _fake_run_sdg_generation,
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.core.pipeline_steps.setup_and_run_garak",
            _fake_setup_and_run,
        )
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        class _Callbacks:
            def report_status(self, _update):
                return None

        adapter.run_benchmark_job(job, _Callbacks())
        scan_dir = next(scan_dir.glob("execution-*"))

        assert sdg_captured["taxonomy_len"] == 8  # BASE_TAXONOMY has 8 entries
        assert sdg_captured["sdg_model"] == "sdg-model"
        assert sdg_captured["sdg_api_base"] == "http://sdg:7000/v1"
        assert sdg_captured["sdg_flow_id"] == "major-sage-742"
        assert sdg_captured["sdg_max_concurrency"] == 10
        assert sdg_captured["sdg_num_samples"] == 0
        assert sdg_captured["sdg_max_tokens"] == 0
        assert (scan_dir / "sdg_raw_output.csv").exists()
        assert (scan_dir / "sdg_normalized_output.csv").exists()
        assert sdg_captured.get("prompts_csv_path") is not None

    def test_simple_intents_custom_taxonomy_from_local_path(self, monkeypatch, tmp_path):
        """Custom taxonomy loaded from a local file path (pre-downloaded by SDK)."""
        module, adapter, job = self._make_adapter_and_job(monkeypatch, tmp_path)

        taxonomy_file = tmp_path / "test_data" / "taxonomy.csv"
        taxonomy_file.parent.mkdir(parents=True, exist_ok=True)
        taxonomy_file.write_text(
            "policy_concept,concept_definition\nCustomHarm,Custom harm definition\n",
            encoding="utf-8",
        )
        job.parameters["policy_s3_key"] = str(taxonomy_file)

        scan_dir = tmp_path / "simple-intents-job"
        scan_dir.mkdir(parents=True, exist_ok=True)

        import pandas as pd

        sdg_captured = {}

        def _fake_run_sdg_generation(taxonomy_df, **kwargs):
            sdg_captured["taxonomy_concepts"] = taxonomy_df["policy_concept"].tolist()
            return pd.DataFrame(
                {
                    "policy_concept": ["CustomHarm"],
                    "concept_definition": ["Custom harm definition"],
                    "prompt": ["Do something harmful"],
                }
            )

        def _fake_setup_and_run(config_json, prompts_csv_path, scan_dir, timeout_seconds):
            report_prefix = scan_dir / "scan"
            report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")
            return module.GarakScanResult(
                returncode=0,
                stdout="",
                stderr="",
                report_prefix=report_prefix,
            )

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.core.pipeline_steps.run_sdg_generation",
            _fake_run_sdg_generation,
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.core.pipeline_steps.setup_and_run_garak",
            _fake_setup_and_run,
        )
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        class _Callbacks:
            def report_status(self, _update):
                return None

        adapter.run_benchmark_job(job, _Callbacks())
        assert sdg_captured["taxonomy_concepts"] == ["CustomHarm"]

    def test_simple_intents_bypass_sdg_from_local_path(self, monkeypatch, tmp_path):
        """Bypass SDG: load pre-generated prompts from a local file."""
        module, adapter, job = self._make_adapter_and_job(monkeypatch, tmp_path)

        intents_file = tmp_path / "test_data" / "prompts.csv"
        intents_file.parent.mkdir(parents=True, exist_ok=True)
        intents_file.write_text(
            "category,prompt,description\nharm,Do bad things,Harm prompts\n",
            encoding="utf-8",
        )
        job.parameters["intents_s3_key"] = str(intents_file)

        scan_dir = tmp_path / "simple-intents-job"
        scan_dir.mkdir(parents=True, exist_ok=True)

        setup_captured = {}

        def _fake_setup_and_run(config_json, prompts_csv_path, scan_dir, timeout_seconds):
            import pandas as pd

            setup_captured["prompts"] = pd.read_csv(prompts_csv_path)
            report_prefix = scan_dir / "scan"
            report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")
            return module.GarakScanResult(
                returncode=0,
                stdout="",
                stderr="",
                report_prefix=report_prefix,
            )

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.core.pipeline_steps.setup_and_run_garak",
            _fake_setup_and_run,
        )
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        class _Callbacks:
            def report_status(self, _update):
                return None

        adapter.run_benchmark_job(job, _Callbacks())
        scan_dir = next(scan_dir.glob("execution-*"))

        assert (scan_dir / "sdg_raw_output.csv").exists()
        assert (scan_dir / "sdg_normalized_output.csv").exists()
        assert len(setup_captured["prompts"]) == 1
        assert setup_captured["prompts"]["category"].iloc[0] == "harm"

    def test_simple_intents_missing_taxonomy_file_raises(self, monkeypatch, tmp_path):
        """Missing taxonomy file raises FileNotFoundError."""
        module, adapter, job = self._make_adapter_and_job(monkeypatch, tmp_path)
        job.parameters["policy_s3_key"] = "/nonexistent/taxonomy.csv"

        class _Callbacks:
            def report_status(self, _update):
                return None

        with pytest.raises(FileNotFoundError, match="Taxonomy file not found"):
            adapter.run_benchmark_job(job, _Callbacks())

    def test_simple_intents_missing_intents_file_raises(self, monkeypatch, tmp_path):
        """Missing bypass prompts file raises FileNotFoundError."""
        module, adapter, job = self._make_adapter_and_job(monkeypatch, tmp_path)
        job.parameters["intents_s3_key"] = "/nonexistent/prompts.csv"

        class _Callbacks:
            def report_status(self, _update):
                return None

        with pytest.raises(FileNotFoundError, match="Intents file not found"):
            adapter.run_benchmark_job(job, _Callbacks())

    def test_simple_intents_missing_sdg_model_raises(self, monkeypatch, tmp_path):
        """SDG path without sdg_model raises ValueError."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="simple-intents-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                "execution_mode": "simple",
                "intents_models": {
                    "judge": {"url": "http://judge:8000/v1", "name": "judge-model"},
                    "attacker": {"url": "http://attacker:9000/v1", "name": "atk-model"},
                    "evaluator": {"url": "http://evaluator:9001/v1", "name": "eval-model"},
                },
            },
            exports=None,
        )

        class _Callbacks:
            def report_status(self, _update):
                return None

        with pytest.raises(ValueError, match="sdg_model.*sdg_api_base"):
            adapter.run_benchmark_job(job, _Callbacks())

    def test_simple_intents_persists_artifacts_to_scan_dir(self, monkeypatch, tmp_path):
        """Verify sdg_raw_output.csv and sdg_normalized_output.csv are in scan_dir."""
        module, adapter, job = self._make_adapter_and_job(monkeypatch, tmp_path)

        intents_file = tmp_path / "test_data" / "prompts.csv"
        intents_file.parent.mkdir(parents=True, exist_ok=True)
        intents_file.write_text(
            "category,prompt,description\nfraud,Scam people,Fraud desc\nviolence,Attack,Violence desc\n",
            encoding="utf-8",
        )
        job.parameters["intents_s3_key"] = str(intents_file)

        scan_dir = tmp_path / "simple-intents-job"
        scan_dir.mkdir(parents=True, exist_ok=True)

        def _fake_setup_and_run(config_json, prompts_csv_path, scan_dir, timeout_seconds):
            report_prefix = scan_dir / "scan"
            report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")
            return module.GarakScanResult(
                returncode=0,
                stdout="",
                stderr="",
                report_prefix=report_prefix,
            )

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.core.pipeline_steps.setup_and_run_garak",
            _fake_setup_and_run,
        )
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        class _Callbacks:
            def report_status(self, _update):
                return None

        adapter.run_benchmark_job(job, _Callbacks())
        scan_dir = next(scan_dir.glob("execution-*"))

        raw_csv = scan_dir / "sdg_raw_output.csv"
        norm_csv = scan_dir / "sdg_normalized_output.csv"
        assert raw_csv.exists()
        assert norm_csv.exists()

        import pandas as pd

        norm_df = pd.read_csv(norm_csv)
        assert "category" in norm_df.columns
        assert "prompt" in norm_df.columns
        assert len(norm_df) == 2

    def test_simple_intents_sdg_uses_secret_url_and_key(self, monkeypatch, tmp_path):
        """SDG reads sdg_url and sdg_api-key from secret when available."""
        module, adapter, job = self._make_adapter_and_job(monkeypatch, tmp_path)

        auth_module = sys.modules["evalhub.adapter.auth"]
        auth_module.read_model_auth_key = lambda name: {
            "sdg_url": "http://localhost:8080/v1",
            "sdg_api-key": "sdg_api-key:ref",
        }.get(name)

        scan_dir = tmp_path / "simple-intents-job"
        scan_dir.mkdir(parents=True, exist_ok=True)

        import pandas as pd

        fake_raw_df = pd.DataFrame({"policy_concept": ["X"], "concept_definition": ["x"], "prompt": ["p"]})

        sdg_captured = {}

        def _fake_run_sdg_generation(taxonomy_df, sdg_model, sdg_api_base, **kwargs):
            sdg_captured["sdg_api_base"] = sdg_api_base
            sdg_captured["api_key"] = kwargs.get("api_key")
            return fake_raw_df

        def _fake_setup_and_run(config_json, prompts_csv_path, scan_dir, timeout_seconds):
            report_prefix = scan_dir / "scan"
            report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")
            return module.GarakScanResult(returncode=0, stdout="", stderr="", report_prefix=report_prefix)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.core.pipeline_steps.run_sdg_generation",
            _fake_run_sdg_generation,
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.core.pipeline_steps.setup_and_run_garak",
            _fake_setup_and_run,
        )
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        class _Callbacks:
            def report_status(self, _update):
                return None

        adapter.run_benchmark_job(job, _Callbacks())

        assert sdg_captured["sdg_api_base"] == "http://localhost:8080/v1"
        assert sdg_captured["api_key"] == "sdg_api-key:ref"


class TestKFPIntentsMode:
    """Tests for intents mode through run_benchmark_job with KFP."""

    def test_kfp_intents_passes_params_to_pipeline(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        captured: dict[str, object] = {}
        report_prefix = tmp_path / "scan"
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")

        def _fake_run_via_kfp(
            self, config, callbacks, garak_config_dict, timeout, intents_params=None, eval_threshold=0.5, **kwargs
        ):
            captured["intents_params"] = intents_params
            captured["eval_threshold"] = eval_threshold
            return module.GarakScanResult(
                returncode=0,
                stdout="",
                stderr="",
                report_prefix=report_prefix,
            ), tmp_path

        monkeypatch.setattr(module.GarakAdapter, "_run_via_kfp", _fake_run_via_kfp)
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        class _Callbacks:
            def report_status(self, _update):
                return None

            def create_oci_artifact(self, _spec):
                return SimpleNamespace(reference="oci://ref", digest="sha256:test")

        job = SimpleNamespace(
            id="kfp-intents-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={
                "save_sample_results": False,
                "execution_mode": "kfp",
                **_INTENTS_MODELS_ALL_ROLES,
            },
            exports=None,
        )

        result = adapter.run_benchmark_job(job, _Callbacks())

        assert captured["intents_params"]["art_intents"] is True
        assert captured["intents_params"]["sdg_model"] == "sdg-model"
        assert result.evaluation_metadata["art_intents"] is True

    def test_art_html_generated_for_intents(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))
        monkeypatch.setenv("KFP_ENDPOINT", "http://kfp:8080")
        monkeypatch.setenv("KFP_NAMESPACE", "test-ns")

        scan_dir = tmp_path / "intents-html-job"
        scan_dir.mkdir(parents=True)
        report_prefix = scan_dir / "scan"
        report_prefix.with_suffix(".report.jsonl").write_text(
            '{"entry_type":"attempt","status":2}\n',
            encoding="utf-8",
        )

        def _fake_run_via_kfp(
            self, config, callbacks, garak_config_dict, timeout, intents_params=None, eval_threshold=0.5, **kwargs
        ):
            return module.GarakScanResult(
                returncode=0,
                stdout="",
                stderr="",
                report_prefix=report_prefix,
            ), scan_dir

        monkeypatch.setattr(module.GarakAdapter, "_run_via_kfp", _fake_run_via_kfp)
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        html_generated = {"called": False}

        def _fake_generate_art_report(content, **kwargs):
            html_generated["called"] = True
            return "<html><body>ART Report</body></html>"

        monkeypatch.setattr(module, "generate_art_report", _fake_generate_art_report)

        class _Callbacks:
            def report_status(self, _update):
                return None

            def create_oci_artifact(self, _spec):
                return SimpleNamespace(reference="oci://ref", digest="sha256:test")

        job = SimpleNamespace(
            id="intents-html-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={"save_sample_results": False, **_INTENTS_MODELS_SINGLE, "execution_mode": "kfp"},
            exports=None,
        )

        adapter.run_benchmark_job(job, _Callbacks())

        assert html_generated["called"] is True
        html_path = scan_dir / "scan.intents.html"
        assert html_path.exists()
        assert "ART Report" in html_path.read_text()

    def test_kfp_raises_when_art_intents_but_no_sdg_model(self, monkeypatch, tmp_path):
        """_run_via_kfp raises ValueError when art_intents=True with no sdg_model (sdg_model always required)."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))
        monkeypatch.setenv("KFP_ENDPOINT", "https://kfp.example.com")
        monkeypatch.setenv("KFP_NAMESPACE", "test-ns")
        monkeypatch.setenv("KFP_S3_SECRET_NAME", "data-conn")

        class _Callbacks:
            def report_status(self, _update):
                return None

            def create_oci_artifact(self, _spec):
                return SimpleNamespace(reference="oci://ref", digest="sha256:test")

        job = SimpleNamespace(
            id="kfp-no-data-source",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={
                "execution_mode": "kfp",
                **_INTENTS_MODELS_SINGLE,
            },
            exports=None,
        )

        with pytest.raises(ValueError, match="sdg_model"):
            adapter.run_benchmark_job(job, _Callbacks())


class TestParseResultsIntentsMode:
    """Tests for _parse_results with art_intents=True."""

    def test_parse_results_with_art_intents(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()

        report_prefix = tmp_path / "scan"
        report_prefix.with_suffix(".report.jsonl").write_text(
            '{"entry_type":"attempt","status":2}\n',
            encoding="utf-8",
        )
        report_prefix.with_suffix(".avid.jsonl").write_text("", encoding="utf-8")

        monkeypatch.setattr(
            module,
            "parse_generations_from_report_content",
            lambda _content, _threshold, **_kwargs: (
                [],
                {"spo.SPOIntent": [{}]},
                {"spo.SPOIntent": [{"detector_results": {}, "notes": {}}]},
            ),
        )
        monkeypatch.setattr(
            module,
            "parse_aggregated_from_avid_content",
            lambda _content: {},
        )
        monkeypatch.setattr(
            module,
            "parse_digest_from_report_content",
            lambda _content: {},
        )
        monkeypatch.setattr(
            module,
            "combine_parsed_results",
            lambda *_args, **kwargs: {
                "scores": {
                    "spo.SPOIntent": {
                        "aggregated_results": {
                            "total_attempts": 20,
                            "unsafe_stubs": 3,
                            "safe_stubs": 7,
                            "attack_success_rate": 30.0,
                            "intent_breakdown": {
                                "S001": {
                                    "total_attempts": 10,
                                    "total_stubs": 5,
                                    "unsafe_stubs": 2,
                                    "safe_stubs": 3,
                                    "attack_success_rate": 40.0,
                                },
                                "S002": {
                                    "total_attempts": 10,
                                    "total_stubs": 5,
                                    "unsafe_stubs": 1,
                                    "safe_stubs": 4,
                                    "attack_success_rate": 20.0,
                                },
                            },
                            "metadata": {},
                        }
                    },
                    "_overall": {
                        "aggregated_results": {
                            "total_attempts": 20,
                            "attack_success_rate": 30.0,
                            "intent_breakdown": {},
                        }
                    },
                }
            },
        )

        result = module.GarakScanResult(
            returncode=0,
            stdout="",
            stderr="",
            report_prefix=report_prefix,
        )
        metrics, overall_score, num_examples, _ = adapter._parse_results(
            result,
            0.5,
            art_intents=True,
        )

        assert len(metrics) == 2
        assert metrics[0].metric_name == "attack_success_rate"
        assert metrics[0].metric_type == "ratio"
        assert metrics[0].metric_value == 0.3
        assert metrics[0].num_samples == 20
        assert metrics[1].metric_name == "spo.SPOIntent_asr"
        assert metrics[1].metric_type == "ratio"
        assert metrics[1].metric_value == 0.3
        assert metrics[1].num_samples is None
        assert metrics[1].metadata["total_attempts"] == 20
        assert metrics[1].metadata["unsafe_stubs"] == 3
        assert metrics[1].metadata["safe_stubs"] == 7
        assert "intent_breakdown" in metrics[1].metadata
        assert metrics[1].metadata["intent_breakdown"]["S001"]["unsafe_stubs"] == 2
        assert overall_score == 0.3
        assert num_examples == 20


# ---------------------------------------------------------------------------
# Targeted KFP component tests
# ---------------------------------------------------------------------------


class _FakeArtifact:
    """Minimal stand-in for KFP dsl.Output / dsl.Input artifacts."""

    def __init__(self, path: str):
        self.path = path


class _FakeMetrics:
    """Minimal stand-in for dsl.Output[dsl.Metrics]."""

    def __init__(self):
        self.logged: dict[str, float] = {}

    def log_metric(self, name: str, value: float):
        self.logged[name] = value


def _get_component_fn(component_func):
    """Extract the raw Python function from a KFP @dsl.component."""
    return getattr(component_func, "python_func", component_func)


class TestValidateComponent:
    """Targeted tests for the validate KFP component."""

    def test_valid_config_and_s3_passes(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import validate

        # Mock garak module so import succeeds
        monkeypatch.setitem(sys.modules, "garak", types.ModuleType("garak"))

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")

        head_bucket_called = {}

        def _fake_create_s3_client():
            client = SimpleNamespace(
                head_bucket=lambda Bucket: head_bucket_called.update({"bucket": Bucket}),
            )
            return client

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )

        config_json = json.dumps({"plugins": {"probe_spec": ["test"]}, "reporting": {}})
        fn = _get_component_fn(validate)
        result = fn(config_json=config_json)
        assert result.valid is True
        assert head_bucket_called["bucket"] == "test-bucket"

    def test_malformed_config_json_raises(self, monkeypatch):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import validate
        from llama_stack_provider_trustyai_garak.errors import GarakValidationError

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        def _fake_create_s3_client():
            return SimpleNamespace(head_bucket=lambda Bucket: None)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )

        fn = _get_component_fn(validate)
        with pytest.raises(GarakValidationError, match="not valid JSON"):
            fn(config_json="not-valid-json{{{")

    def test_missing_plugins_section_raises(self, monkeypatch):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import validate
        from llama_stack_provider_trustyai_garak.errors import GarakValidationError

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        def _fake_create_s3_client():
            return SimpleNamespace(head_bucket=lambda Bucket: None)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )

        fn = _get_component_fn(validate)
        with pytest.raises(GarakValidationError, match="plugins"):
            fn(config_json=json.dumps({"reporting": {}}))

    def test_missing_s3_bucket_raises(self, monkeypatch):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import validate
        from llama_stack_provider_trustyai_garak.errors import GarakValidationError

        # Mock garak module so import succeeds
        monkeypatch.setitem(sys.modules, "garak", types.ModuleType("garak"))

        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)

        fn = _get_component_fn(validate)
        with pytest.raises(GarakValidationError, match="AWS_S3_BUCKET"):
            fn(config_json=json.dumps({"plugins": {}}))

    def test_unreachable_s3_bucket_raises(self, monkeypatch):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import validate
        from llama_stack_provider_trustyai_garak.errors import GarakValidationError

        # Mock garak module so import succeeds
        monkeypatch.setitem(sys.modules, "garak", types.ModuleType("garak"))

        monkeypatch.setenv("AWS_S3_BUCKET", "bad-bucket")

        def _fake_create_s3_client():
            def _fail(**kwargs):
                raise ConnectionError("unreachable")

            return SimpleNamespace(head_bucket=_fail)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )

        fn = _get_component_fn(validate)
        with pytest.raises(GarakValidationError, match="not reachable"):
            fn(config_json=json.dumps({"plugins": {}}))


class TestResolveTaxonomyComponent:
    """Tests for the resolve_taxonomy KFP component."""

    def test_no_policy_emits_base_taxonomy(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import resolve_taxonomy
        import pandas as pd

        artifact = _FakeArtifact(str(tmp_path / "taxonomy.csv"))

        fn = _get_component_fn(resolve_taxonomy)
        fn(policy_s3_key="", policy_format="csv", taxonomy_dataset=artifact)

        df = pd.read_csv(artifact.path)
        assert len(df) > 0
        assert "policy_concept" in df.columns

    def test_custom_taxonomy_from_s3(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import resolve_taxonomy
        import pandas as pd

        artifact = _FakeArtifact(str(tmp_path / "taxonomy.csv"))

        taxonomy_csv = "policy_concept,concept_definition\nCustom,Custom def\n"

        def _fake_create_s3_client():
            def _get_object(Bucket, Key):
                return {"Body": SimpleNamespace(read=lambda: taxonomy_csv.encode())}

            return SimpleNamespace(get_object=_get_object)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )
        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        fn = _get_component_fn(resolve_taxonomy)
        fn(
            policy_s3_key="path/to/taxonomy.csv",
            policy_format="csv",
            taxonomy_dataset=artifact,
        )

        df = pd.read_csv(artifact.path)
        assert df["policy_concept"].iloc[0] == "Custom"

    def test_invalid_s3_uri_raises(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import resolve_taxonomy
        from llama_stack_provider_trustyai_garak.errors import GarakValidationError

        artifact = _FakeArtifact(str(tmp_path / "taxonomy.csv"))

        fn = _get_component_fn(resolve_taxonomy)
        with pytest.raises(GarakValidationError, match="Invalid policy_s3_key"):
            fn(
                policy_s3_key="s3://bucket-only",
                policy_format="csv",
                taxonomy_dataset=artifact,
            )


class TestSdgGenerateComponent:
    """Tests for the sdg_generate KFP component."""

    def test_non_intents_writes_empty_marker(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import sdg_generate

        taxonomy = _FakeArtifact(str(tmp_path / "taxonomy.csv"))
        Path(taxonomy.path).write_text("policy_concept,concept_definition\nA,B\n")
        sdg_out = _FakeArtifact(str(tmp_path / "sdg.csv"))

        fn = _get_component_fn(sdg_generate)
        fn(
            art_intents=False,
            intents_s3_key="",
            sdg_model="",
            sdg_api_base="",
            sdg_flow_id="",
            sdg_max_concurrency=10,
            sdg_num_samples=0,
            sdg_max_tokens=0,
            taxonomy_dataset=taxonomy,
            sdg_dataset=sdg_out,
        )

        assert Path(sdg_out.path).read_text() == ""

    def test_bypass_mode_writes_empty_marker(self, monkeypatch, tmp_path):
        """When intents_s3_key is set, sdg_generate writes empty (bypass handled by prepare_prompts)."""
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import sdg_generate

        taxonomy = _FakeArtifact(str(tmp_path / "taxonomy.csv"))
        Path(taxonomy.path).write_text("policy_concept,concept_definition\nA,B\n")
        sdg_out = _FakeArtifact(str(tmp_path / "sdg.csv"))

        fn = _get_component_fn(sdg_generate)
        fn(
            art_intents=True,
            intents_s3_key="path/to/intents.csv",
            sdg_model="",
            sdg_api_base="",
            sdg_flow_id="",
            sdg_max_concurrency=10,
            sdg_num_samples=0,
            sdg_max_tokens=0,
            taxonomy_dataset=taxonomy,
            sdg_dataset=sdg_out,
        )

        assert Path(sdg_out.path).read_text() == ""

    def test_intents_no_sdg_model_raises(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import sdg_generate
        from llama_stack_provider_trustyai_garak.errors import GarakValidationError

        taxonomy = _FakeArtifact(str(tmp_path / "taxonomy.csv"))
        Path(taxonomy.path).write_text("policy_concept,concept_definition\nA,B\n")
        sdg_out = _FakeArtifact(str(tmp_path / "sdg.csv"))

        fn = _get_component_fn(sdg_generate)
        with pytest.raises(GarakValidationError, match="sdg_model"):
            fn(
                art_intents=True,
                intents_s3_key="",
                sdg_model="",
                sdg_api_base="",
                sdg_flow_id="",
                sdg_max_concurrency=10,
                sdg_num_samples=0,
                sdg_max_tokens=0,
                taxonomy_dataset=taxonomy,
                sdg_dataset=sdg_out,
            )

    def test_sdg_missing_api_base_raises(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import sdg_generate
        from llama_stack_provider_trustyai_garak.errors import GarakValidationError

        taxonomy = _FakeArtifact(str(tmp_path / "taxonomy.csv"))
        Path(taxonomy.path).write_text("policy_concept,concept_definition\nA,B\n")
        sdg_out = _FakeArtifact(str(tmp_path / "sdg.csv"))

        fn = _get_component_fn(sdg_generate)
        with pytest.raises(GarakValidationError, match="sdg_api_base"):
            fn(
                art_intents=True,
                intents_s3_key="",
                sdg_model="some-model",
                sdg_api_base="",
                sdg_flow_id="",
                sdg_max_concurrency=10,
                sdg_num_samples=0,
                sdg_max_tokens=0,
                taxonomy_dataset=taxonomy,
                sdg_dataset=sdg_out,
            )

    def test_sdg_outputs_raw_artifact(self, monkeypatch, tmp_path):
        """SDG path writes only the raw artifact (normalisation is in prepare_prompts)."""
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import sdg_generate
        from llama_stack_provider_trustyai_garak.sdg import SDGResult
        import pandas as pd

        taxonomy = _FakeArtifact(str(tmp_path / "taxonomy.csv"))
        Path(taxonomy.path).write_text("policy_concept,concept_definition\nHarm,Harm def\n")
        sdg_out = _FakeArtifact(str(tmp_path / "sdg.csv"))

        raw_df = pd.DataFrame(
            {
                "policy_concept": ["Harm"],
                "concept_definition": ["Harm def"],
                "prompt": ["generated"],
                "demographics_pool": [["Teens"]],
            }
        )
        norm_df = pd.DataFrame(
            {
                "category": ["harm"],
                "prompt": ["generated"],
                "description": ["Harm def"],
            }
        )

        def _fake_generate_sdg(
            model,
            api_base,
            flow_id,
            api_key="dummy",
            taxonomy=None,
            max_concurrency=10,
            num_samples=0,
            max_tokens=0,
            provider="",
        ):
            return SDGResult(raw=raw_df, normalized=norm_df)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.sdg.generate_sdg_dataset",
            _fake_generate_sdg,
        )

        fn = _get_component_fn(sdg_generate)
        fn(
            art_intents=True,
            intents_s3_key="",
            sdg_model="test-model",
            sdg_api_base="http://sdg:8000",
            sdg_flow_id="test-flow",
            sdg_max_concurrency=10,
            sdg_num_samples=0,
            sdg_max_tokens=0,
            taxonomy_dataset=taxonomy,
            sdg_dataset=sdg_out,
        )

        raw_result = pd.read_csv(sdg_out.path)
        assert "policy_concept" in raw_result.columns
        assert "prompt" in raw_result.columns
        assert len(raw_result) == 1

    def test_non_default_sdg_params_plumbed_through(self, monkeypatch, tmp_path):
        """Non-default sdg_max_concurrency/num_samples/max_tokens reach generate_sdg_dataset."""
        from unittest.mock import MagicMock
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import sdg_generate
        from llama_stack_provider_trustyai_garak.sdg import SDGResult
        import pandas as pd

        taxonomy = _FakeArtifact(str(tmp_path / "taxonomy.csv"))
        Path(taxonomy.path).write_text("policy_concept,concept_definition\nA,B\n")
        sdg_out = _FakeArtifact(str(tmp_path / "sdg.csv"))

        raw_df = pd.DataFrame({"policy_concept": ["A"], "prompt": ["p"], "concept_definition": ["B"]})
        mock_generate = MagicMock(return_value=SDGResult(raw=raw_df, normalized=raw_df))
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.sdg.generate_sdg_dataset",
            mock_generate,
        )

        fn = _get_component_fn(sdg_generate)
        fn(
            art_intents=True,
            intents_s3_key="",
            sdg_model="test-model",
            sdg_api_base="http://sdg:8000",
            sdg_provider="openai",
            sdg_flow_id="test-flow",
            sdg_max_concurrency=3,
            sdg_num_samples=25,
            sdg_max_tokens=4096,
            taxonomy_dataset=taxonomy,
            sdg_dataset=sdg_out,
        )

        mock_generate.assert_called_once()
        kwargs = mock_generate.call_args.kwargs
        assert kwargs["max_concurrency"] == 3
        assert kwargs["num_samples"] == 25
        assert kwargs["max_tokens"] == 4096
        assert kwargs["provider"] == "openai"

    def test_default_sdg_params_when_zeros(self, monkeypatch, tmp_path):
        """When sdg params are 0 (default), they are passed as 0 and generate_sdg_dataset handles the fallback."""
        from unittest.mock import MagicMock
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import sdg_generate
        from llama_stack_provider_trustyai_garak.sdg import SDGResult
        import pandas as pd

        taxonomy = _FakeArtifact(str(tmp_path / "taxonomy.csv"))
        Path(taxonomy.path).write_text("policy_concept,concept_definition\nA,B\n")
        sdg_out = _FakeArtifact(str(tmp_path / "sdg.csv"))

        raw_df = pd.DataFrame({"policy_concept": ["A"], "prompt": ["p"], "concept_definition": ["B"]})
        mock_generate = MagicMock(return_value=SDGResult(raw=raw_df, normalized=raw_df))
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.sdg.generate_sdg_dataset",
            mock_generate,
        )

        fn = _get_component_fn(sdg_generate)
        fn(
            art_intents=True,
            intents_s3_key="",
            sdg_model="test-model",
            sdg_api_base="http://sdg:8000",
            sdg_flow_id="test-flow",
            sdg_max_concurrency=0,
            sdg_num_samples=0,
            sdg_max_tokens=0,
            taxonomy_dataset=taxonomy,
            sdg_dataset=sdg_out,
        )

        mock_generate.assert_called_once()
        kwargs = mock_generate.call_args.kwargs
        assert kwargs["max_concurrency"] == 0
        assert kwargs["num_samples"] == 0
        assert kwargs["max_tokens"] == 0


class TestPreparePromptsComponent:
    """Tests for the prepare_prompts KFP component."""

    def test_non_intents_writes_empty_marker(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import prepare_prompts

        sdg_in = _FakeArtifact(str(tmp_path / "sdg.csv"))
        Path(sdg_in.path).write_text("")
        prompts_out = _FakeArtifact(str(tmp_path / "prompts.csv"))

        fn = _get_component_fn(prepare_prompts)
        fn(
            art_intents=False,
            s3_prefix="evalhub-garak-kfp/job1",
            intents_s3_key="",
            intents_format="csv",
            sdg_dataset=sdg_in,
            prompts_dataset=prompts_out,
        )

        assert Path(prompts_out.path).read_text() == ""

    def test_sdg_path_normalises_and_uploads(self, monkeypatch, tmp_path):
        """SDG ran: reads raw artifact, uploads raw + normalised to S3, outputs normalised."""
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import prepare_prompts
        import pandas as pd

        raw_csv = "policy_concept,concept_definition,prompt\nHarm,Harmful stuff,Do bad\n"
        sdg_in = _FakeArtifact(str(tmp_path / "sdg.csv"))
        Path(sdg_in.path).write_text(raw_csv)
        prompts_out = _FakeArtifact(str(tmp_path / "prompts.csv"))

        uploaded: dict[str, str] = {}

        def _fake_create_s3_client():
            def _put_object(Bucket, Key, Body):
                uploaded[Key] = Body.decode("utf-8") if isinstance(Body, bytes) else Body

            def _upload_file(filepath, bucket, key):
                uploaded[key] = Path(filepath).read_text()

            return SimpleNamespace(put_object=_put_object, upload_file=_upload_file)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )
        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        fn = _get_component_fn(prepare_prompts)
        fn(
            art_intents=True,
            s3_prefix="evalhub-garak-kfp/job123",
            intents_s3_key="",
            intents_format="csv",
            sdg_dataset=sdg_in,
            prompts_dataset=prompts_out,
        )

        assert "evalhub-garak-kfp/job123/sdg_raw_output.csv" in uploaded
        assert "evalhub-garak-kfp/job123/sdg_normalized_output.csv" in uploaded

        result_df = pd.read_csv(prompts_out.path)
        assert "category" in result_df.columns
        assert "prompt" in result_df.columns
        assert len(result_df) == 1

        raw_uploaded = uploaded["evalhub-garak-kfp/job123/sdg_raw_output.csv"]
        assert "policy_concept" in raw_uploaded

    def test_bypass_fetches_and_preserves_raw(self, monkeypatch, tmp_path):
        """Bypass mode: fetches user file from S3, uploads original as raw, normalises."""
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import prepare_prompts
        import pandas as pd

        user_csv = "policy_concept,concept_definition,prompt,demographics_pool\nHarm,Bad stuff,Do harm,\"['Teens']\"\n"

        sdg_in = _FakeArtifact(str(tmp_path / "sdg.csv"))
        Path(sdg_in.path).write_text("")
        prompts_out = _FakeArtifact(str(tmp_path / "prompts.csv"))

        uploaded: dict[str, str] = {}

        def _fake_create_s3_client():
            def _get_object(Bucket, Key):
                return {"Body": SimpleNamespace(read=lambda: user_csv.encode())}

            def _put_object(Bucket, Key, Body):
                uploaded[Key] = Body.decode("utf-8") if isinstance(Body, bytes) else Body

            def _upload_file(filepath, bucket, key):
                uploaded[key] = Path(filepath).read_text()

            return SimpleNamespace(
                get_object=_get_object,
                put_object=_put_object,
                upload_file=_upload_file,
            )

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )
        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        fn = _get_component_fn(prepare_prompts)
        fn(
            art_intents=True,
            s3_prefix="evalhub-garak-kfp/job456",
            intents_s3_key="user/intents.csv",
            intents_format="csv",
            sdg_dataset=sdg_in,
            prompts_dataset=prompts_out,
        )

        raw_uploaded = uploaded.get("evalhub-garak-kfp/job456/sdg_raw_output.csv", "")
        assert "demographics_pool" in raw_uploaded

        result_df = pd.read_csv(prompts_out.path)
        assert list(result_df.columns) == ["category", "prompt", "description"]
        assert len(result_df) == 1

    def test_empty_sdg_and_no_bypass_writes_empty(self, monkeypatch, tmp_path):
        """No SDG output and no bypass key: writes empty marker."""
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import prepare_prompts

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        sdg_in = _FakeArtifact(str(tmp_path / "sdg.csv"))
        Path(sdg_in.path).write_text("")
        prompts_out = _FakeArtifact(str(tmp_path / "prompts.csv"))

        fn = _get_component_fn(prepare_prompts)
        fn(
            art_intents=True,
            s3_prefix="evalhub-garak-kfp/job1",
            intents_s3_key="",
            intents_format="csv",
            sdg_dataset=sdg_in,
            prompts_dataset=prompts_out,
        )

        assert Path(prompts_out.path).read_text() == ""


class TestAdapterMutualExclusivity:
    """Tests for mutual exclusivity of policy_s3_key and intents_s3_key in adapter."""

    _KFP_ENV = {
        "KFP_ENDPOINT": "http://kfp:8080",
        "KFP_NAMESPACE": "test-ns",
        "KFP_S3_SECRET_NAME": "test-secret",
        "AWS_S3_BUCKET": "test-bucket",
    }

    def test_both_policy_and_intents_s3_key_raises(self, monkeypatch):
        """Both policy_s3_key and intents_s3_key set raises ValueError."""
        from llama_stack_provider_trustyai_garak.evalhub.garak_adapter import GarakAdapter

        for k, v in self._KFP_ENV.items():
            monkeypatch.setenv(k, v)

        adapter = GarakAdapter.__new__(GarakAdapter)

        with pytest.raises(ValueError, match="mutually exclusive"):
            adapter._run_via_kfp(
                config=SimpleNamespace(
                    id="test-job",
                    benchmark_id="test",
                    parameters={},
                ),
                callbacks=SimpleNamespace(report_status=lambda *a, **kw: None),
                garak_config_dict={"plugins": {}},
                timeout=300,
                intents_params={
                    "art_intents": True,
                    "policy_s3_key": "path/to/policy.csv",
                    "intents_s3_key": "path/to/intents.csv",
                    "sdg_model": "model",
                    "sdg_api_base": "http://base",
                },
                eval_threshold=0.5,
            )

    def test_bypass_mode_no_sdg_params_accepted(self, monkeypatch):
        """intents_s3_key set without sdg_model/sdg_api_base should not raise SDG validation."""
        from llama_stack_provider_trustyai_garak.evalhub.garak_adapter import GarakAdapter

        for k, v in self._KFP_ENV.items():
            monkeypatch.setenv(k, v)

        adapter = GarakAdapter.__new__(GarakAdapter)

        ip = {
            "art_intents": True,
            "policy_s3_key": "",
            "intents_s3_key": "path/to/intents.csv",
            "intents_format": "csv",
            "sdg_model": "",
            "sdg_api_base": "",
        }

        try:
            adapter._run_via_kfp(
                config=SimpleNamespace(
                    id="test-job",
                    benchmark_id="test",
                    parameters={},
                ),
                callbacks=SimpleNamespace(report_status=lambda *a, **kw: None),
                garak_config_dict={"plugins": {}},
                timeout=300,
                intents_params=ip,
                eval_threshold=0.5,
            )
        except ValueError as e:
            assert "sdg_model" not in str(e), f"Should not require sdg_model in bypass mode: {e}"
            assert "sdg_api_base" not in str(e), f"Should not require sdg_api_base in bypass mode: {e}"
        except Exception:
            pass  # Other errors (KFP client creation, etc.) are expected

    def test_sdg_mode_requires_sdg_params(self, monkeypatch):
        """art_intents=True without intents_s3_key requires sdg_model."""
        from llama_stack_provider_trustyai_garak.evalhub.garak_adapter import GarakAdapter

        for k, v in self._KFP_ENV.items():
            monkeypatch.setenv(k, v)

        adapter = GarakAdapter.__new__(GarakAdapter)

        with pytest.raises(ValueError, match="sdg_model"):
            adapter._run_via_kfp(
                config=SimpleNamespace(
                    id="test-job",
                    benchmark_id="test",
                    parameters={},
                ),
                callbacks=SimpleNamespace(report_status=lambda *a, **kw: None),
                garak_config_dict={"plugins": {}},
                timeout=300,
                intents_params={
                    "art_intents": True,
                    "policy_s3_key": "",
                    "intents_s3_key": "",
                    "sdg_model": "",
                    "sdg_api_base": "",
                },
                eval_threshold=0.5,
            )


class TestArtifactMetadataSurfacing:
    """Tests that artifact S3 paths are surfaced in evaluation_metadata."""

    def test_intents_kfp_job_includes_only_verified_artifacts(self, monkeypatch):
        """art_intents=True + KFP mode -> only artifacts verified in S3 are included."""
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import DEFAULT_S3_PREFIX
        from llama_stack_provider_trustyai_garak.constants import EXECUTION_MODE_KFP
        from unittest.mock import MagicMock
        import os

        job_id = "test-job-123"
        bucket = "my-bucket"
        s3_prefix = f"{DEFAULT_S3_PREFIX}/{job_id}"

        monkeypatch.setenv("AWS_S3_BUCKET", bucket)

        existing_keys = {
            f"{s3_prefix}/sdg_raw_output.csv",
            f"{s3_prefix}/sdg_normalized_output.csv",
            f"{s3_prefix}/scan.report.jsonl",
        }

        mock_s3 = MagicMock()

        def _head_object(Bucket, Key):
            if Key not in existing_keys:
                raise Exception("Not found")

        mock_s3.head_object = MagicMock(side_effect=_head_object)

        mock_create = MagicMock(return_value=mock_s3)
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.garak_adapter.create_s3_client",
            mock_create,
            raising=False,
        )

        _kfp_ov = {}
        _prefix = _kfp_ov.get("s3_prefix", os.getenv("KFP_S3_PREFIX", DEFAULT_S3_PREFIX))
        _bucket = _kfp_ov.get("s3_bucket", os.getenv("AWS_S3_BUCKET", ""))

        artifact_keys = {
            "sdg_raw_output": f"{s3_prefix}/sdg_raw_output.csv",
            "sdg_normalized_output": f"{s3_prefix}/sdg_normalized_output.csv",
            "intents_html_report": f"{s3_prefix}/scan.intents.html",
            "scan_report": f"{s3_prefix}/scan.report.jsonl",
            "scan_hitlog": f"{s3_prefix}/scan.hitlog.jsonl",
        }
        verified = {}
        if _bucket:
            s3 = mock_create()
            for name, key in artifact_keys.items():
                try:
                    s3.head_object(Bucket=_bucket, Key=key)
                    verified[name] = f"s3://{_bucket}/{key}"
                except Exception:
                    pass

        assert len(verified) == 3
        assert "sdg_raw_output" in verified
        assert "sdg_normalized_output" in verified
        assert "scan_report" in verified
        assert "intents_html_report" not in verified
        assert "scan_hitlog" not in verified
        assert verified["sdg_raw_output"] == f"s3://{bucket}/{s3_prefix}/sdg_raw_output.csv"

    def test_intents_kfp_job_bucket_from_kfp_config(self, monkeypatch):
        """s3_bucket from kfp_config takes precedence over env var."""
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import DEFAULT_S3_PREFIX
        import os

        monkeypatch.setenv("AWS_S3_BUCKET", "env-bucket")
        kfp_ov = {"s3_bucket": "config-bucket"}
        _bucket = kfp_ov.get("s3_bucket", os.getenv("AWS_S3_BUCKET", ""))

        assert _bucket == "config-bucket"

    def test_non_intents_job_no_artifacts(self):
        """art_intents=False -> no artifacts dict in evaluation_metadata."""
        from llama_stack_provider_trustyai_garak.constants import EXECUTION_MODE_KFP

        art_intents = False
        execution_mode = EXECUTION_MODE_KFP

        eval_meta = {
            "framework": "garak",
            "art_intents": art_intents,
            "execution_mode": execution_mode,
        }

        if art_intents and execution_mode == EXECUTION_MODE_KFP:
            eval_meta["artifacts"] = {}

        assert "artifacts" not in eval_meta


class TestWriteKfpOutputsComponent:
    """Targeted tests for the write_kfp_outputs KFP component."""

    def test_missing_s3_bucket_fails_component(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import write_kfp_outputs

        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)

        metrics = _FakeMetrics()
        html = _FakeArtifact(str(tmp_path / "report.html"))

        fn = _get_component_fn(write_kfp_outputs)
        with pytest.raises(RuntimeError, match="AWS_S3_BUCKET"):
            fn(
                s3_prefix="test/prefix",
                eval_threshold=0.5,
                art_intents=False,
                summary_metrics=metrics,
                html_report=html,
            )

    def test_empty_report_fails_component(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import write_kfp_outputs

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        def _fake_create_s3_client():
            def _get_object(**kwargs):
                return {"Body": SimpleNamespace(read=lambda: b"")}

            return SimpleNamespace(get_object=_get_object, upload_file=lambda *a: None)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )

        metrics = _FakeMetrics()
        html = _FakeArtifact(str(tmp_path / "report.html"))

        fn = _get_component_fn(write_kfp_outputs)
        with pytest.raises(RuntimeError, match="report file is empty or absent"):
            fn(
                s3_prefix="test/prefix",
                eval_threshold=0.5,
                art_intents=False,
                summary_metrics=metrics,
                html_report=html,
            )

    def test_native_probes_logs_metrics_and_html(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import write_kfp_outputs

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        report_content = '{"entry_type":"attempt","status":2}\n'
        html_content = "<html><body>Native Report</body></html>"

        def _fake_create_s3_client():
            def _get_object(Bucket, Key):
                if Key.endswith(".report.jsonl"):
                    return {"Body": SimpleNamespace(read=lambda: report_content.encode())}
                if Key.endswith(".avid.jsonl"):
                    return {"Body": SimpleNamespace(read=lambda: b"")}
                if Key.endswith(".report.html"):
                    return {"Body": SimpleNamespace(read=lambda: html_content.encode())}
                raise Exception(f"unexpected key: {Key}")

            return SimpleNamespace(get_object=_get_object, upload_file=lambda *a: None)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_generations_from_report_content",
            lambda content, threshold, **kwargs: ([], {}, {}),
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_aggregated_from_avid_content",
            lambda content: {},
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_digest_from_report_content",
            lambda content: {},
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.combine_parsed_results",
            lambda *args, **kwargs: {
                "scores": {
                    "_overall": {
                        "aggregated_results": {
                            "total_attempts": 100,
                            "vulnerable_responses": 15,
                            "attack_success_rate": 15.0,
                        }
                    }
                }
            },
        )

        metrics = _FakeMetrics()
        html = _FakeArtifact(str(tmp_path / "report.html"))

        fn = _get_component_fn(write_kfp_outputs)
        fn(
            s3_prefix="test/prefix",
            eval_threshold=0.5,
            art_intents=False,
            summary_metrics=metrics,
            html_report=html,
        )

        assert metrics.logged["total_attempts"] == 100
        assert metrics.logged["vulnerable_responses"] == 15
        assert metrics.logged["attack_success_rate"] == 15.0
        assert "Native Report" in Path(html.path).read_text()

    def test_intents_mode_uploads_html_to_s3(self, monkeypatch, tmp_path):
        """Intents mode: uploads intents HTML to S3 job folder."""
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import write_kfp_outputs

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        report_content = '{"entry_type":"attempt","status":2}\n'

        uploaded_keys: list[str] = []

        def _fake_create_s3_client():
            def _get_object(Bucket, Key):
                if Key.endswith(".report.jsonl"):
                    return {"Body": SimpleNamespace(read=lambda: report_content.encode())}
                return {"Body": SimpleNamespace(read=lambda: b"")}

            def _upload_file(filepath, bucket, key):
                uploaded_keys.append(key)

            return SimpleNamespace(get_object=_get_object, upload_file=_upload_file)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_generations_from_report_content",
            lambda content, threshold, **kwargs: ([], {}, {}),
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_aggregated_from_avid_content",
            lambda content: {},
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_digest_from_report_content",
            lambda content: {},
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.combine_parsed_results",
            lambda *args, **kwargs: {"scores": {"_overall": {"aggregated_results": {"attack_success_rate": 25.0}}}},
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.generate_art_report",
            lambda content, **kw: "<html>ART</html>",
        )

        metrics = _FakeMetrics()
        html = _FakeArtifact(str(tmp_path / "report.html"))

        fn = _get_component_fn(write_kfp_outputs)
        fn(
            s3_prefix="evalhub-garak/job123",
            eval_threshold=0.5,
            art_intents=True,
            summary_metrics=metrics,
            html_report=html,
        )

        assert metrics.logged["attack_success_rate"] == 25.0
        assert "ART" in Path(html.path).read_text()
        assert "evalhub-garak/job123/scan.intents.html" in uploaded_keys

    def test_intents_mode_logs_asr_metric(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import write_kfp_outputs

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        report_content = '{"entry_type":"attempt","status":2}\n'

        def _fake_create_s3_client():
            def _get_object(Bucket, Key):
                if Key.endswith(".report.jsonl"):
                    return {"Body": SimpleNamespace(read=lambda: report_content.encode())}
                return {"Body": SimpleNamespace(read=lambda: b"")}

            return SimpleNamespace(get_object=_get_object, upload_file=lambda *a: None)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_generations_from_report_content",
            lambda content, threshold, **kwargs: ([], {}, {}),
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_aggregated_from_avid_content",
            lambda content: {},
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_digest_from_report_content",
            lambda content: {},
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.combine_parsed_results",
            lambda *args, **kwargs: {
                "scores": {
                    "_overall": {
                        "aggregated_results": {
                            "attack_success_rate": 25.0,
                        }
                    }
                }
            },
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.generate_art_report",
            lambda content, **kw: "<html>ART</html>",
        )

        metrics = _FakeMetrics()
        html = _FakeArtifact(str(tmp_path / "report.html"))

        fn = _get_component_fn(write_kfp_outputs)
        fn(
            s3_prefix="test/prefix",
            eval_threshold=0.5,
            art_intents=True,
            summary_metrics=metrics,
            html_report=html,
        )

        assert metrics.logged["attack_success_rate"] == 25.0
        assert "total_attempts" not in metrics.logged
        assert "ART" in Path(html.path).read_text()

    def test_parse_failure_writes_fallback_html_and_fails_component(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import write_kfp_outputs

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        def _fake_create_s3_client():
            def _get_object(Bucket, Key):
                if Key.endswith(".report.jsonl"):
                    return {"Body": SimpleNamespace(read=lambda: b'{"data": true}\n')}
                return {"Body": SimpleNamespace(read=lambda: b"")}

            return SimpleNamespace(get_object=_get_object, upload_file=lambda *a: None)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_generations_from_report_content",
            lambda content, threshold, **kwargs: (_ for _ in ()).throw(RuntimeError("parse boom")),
        )

        metrics = _FakeMetrics()
        html = _FakeArtifact(str(tmp_path / "report.html"))

        fn = _get_component_fn(write_kfp_outputs)
        with pytest.raises(RuntimeError, match="parse boom"):
            fn(
                s3_prefix="test/prefix",
                eval_threshold=0.5,
                art_intents=False,
                summary_metrics=metrics,
                html_report=html,
            )

        html_content = Path(html.path).read_text()
        assert "Report generation failed" in html_content


class TestResolveKfpModelUrl:
    """Tests for _resolve_kfp_model_url model URL resolution chain."""

    def test_non_localhost_url_passes_through(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        result = module.GarakAdapter._resolve_kfp_model_url("https://model.example.com/v1", {})
        assert result == "https://model.example.com/v1"

    def test_explicit_kfp_config_model_url_wins(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        bc = {"kfp_config": {"model_url": "https://override.example.com/v1"}}
        result = module.GarakAdapter._resolve_kfp_model_url("http://localhost:8080", bc)
        assert result == "https://override.example.com/v1"

    def test_model_auth_secret_fallback(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        auth_mod = sys.modules["evalhub.adapter.auth"]
        auth_mod.read_model_auth_key = lambda name: (
            "https://from-secret.example.com/v1" if name == "model_url" else None
        )
        result = module.GarakAdapter._resolve_kfp_model_url("http://localhost:8080", {})
        assert result == "https://from-secret.example.com/v1"

    def test_intents_models_judge_fallback(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        bc = {
            "intents_models": {
                "judge": {"url": "https://judge.example.com/v1", "name": "j"},
            }
        }
        result = module.GarakAdapter._resolve_kfp_model_url("http://localhost:8080", bc)
        assert result == "https://judge.example.com/v1"

    def test_intents_models_skips_localhost_urls(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        bc = {
            "intents_models": {
                "judge": {"url": "http://localhost:8080/v1", "name": "j"},
                "attacker": {"url": "https://real-attacker.example.com/v1", "name": "a"},
            }
        }
        result = module.GarakAdapter._resolve_kfp_model_url("http://localhost:8080", bc)
        assert result == "https://real-attacker.example.com/v1"

    def test_returns_localhost_when_no_fallbacks(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        result = module.GarakAdapter._resolve_kfp_model_url("http://localhost:8080", {})
        assert result == "http://localhost:8080"

    def test_warns_when_localhost_remains_in_kfp_mode(self, monkeypatch, caplog):
        import logging

        module = _load_evalhub_garak_adapter(monkeypatch)
        bc = {"kfp_config": {"namespace": "test-ns"}}
        with caplog.at_level(logging.WARNING):
            result = module.GarakAdapter._resolve_kfp_model_url("http://localhost:8080", bc)
        assert result == "http://localhost:8080"
        assert "sidecar address" in caplog.text
        assert "kfp_config.model_url" in caplog.text

    def test_no_warning_without_kfp_config(self, monkeypatch, caplog):
        import logging

        module = _load_evalhub_garak_adapter(monkeypatch)
        with caplog.at_level(logging.WARNING):
            module.GarakAdapter._resolve_kfp_model_url("http://localhost:8080", {})
        assert "sidecar address" not in caplog.text

    def test_127_0_0_1_treated_as_sidecar(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        bc = {"kfp_config": {"model_url": "https://real.example.com"}}
        result = module.GarakAdapter._resolve_kfp_model_url("http://127.0.0.1:8080/v1", bc)
        assert result == "https://real.example.com"

    def test_end_to_end_generator_uses_intents_url(self, monkeypatch, tmp_path):
        """When model.url is localhost and intents_models has a real URL,
        the generator URI in the final config should use the intents URL."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="kfp-sidecar-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://localhost:8080", name="my-model"),
            parameters={
                **_INTENTS_MODELS_SINGLE,
                "kfp_config": {"namespace": "test-ns"},
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        generator_uri = config_dict["plugins"]["generators"]["openai"]["OpenAICompatible"]["uri"]
        assert "localhost" not in generator_uri
        assert generator_uri == "http://judge:8000/v1"

    def test_precedence_kfp_config_over_intents_models(self, monkeypatch, tmp_path):
        """kfp_config.model_url should take precedence over intents_models."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="kfp-explicit-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://localhost:8080", name="my-model"),
            parameters={
                **_INTENTS_MODELS_SINGLE,
                "kfp_config": {
                    "namespace": "test-ns",
                    "model_url": "https://explicit-override.example.com/v1",
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        generator_uri = config_dict["plugins"]["generators"]["openai"]["OpenAICompatible"]["uri"]
        assert generator_uri == "https://explicit-override.example.com/v1"


@pytest.mark.parametrize("save_samples", [None, False, True])
@pytest.mark.parametrize("upload_fails", [False, True])
@pytest.mark.parametrize("bucket_override", [None, "config-bucket"])
def test_kfp_sample_export_uploaded_before_reference(
    monkeypatch, tmp_path, caplog, save_samples, upload_fails, bucket_override
):
    """KFP publishes only uploaded sample files, including with no OCI/MLflow."""
    from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import KFPConfig

    module = _load_evalhub_garak_adapter(monkeypatch)
    monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))
    adapter = module.GarakAdapter()
    scan_dir = tmp_path / "job" / "execution-test"
    scan_dir.mkdir(parents=True)
    report_prefix = scan_dir / "scan"
    report_prefix.with_suffix(".report.jsonl").write_text(
        json.dumps(
            {
                "entry_type": "attempt",
                "status": 2,
                "uuid": "id",
                "probe_classname": "dan.Test",
                "outputs": ["yes", "no"],
                "detector_results": {"dan.DAN": [[1], [0]]},
            }
        )
    )
    monkeypatch.setattr(
        adapter,
        "_run_via_kfp",
        lambda *a, **kw: (
            module.GarakScanResult(returncode=0, stdout="", stderr="", report_prefix=report_prefix),
            scan_dir,
        ),
    )
    monkeypatch.setattr(adapter, "_parse_results", lambda *a, **kw: ([], None, 2, {}))
    monkeypatch.setattr(
        KFPConfig,
        "from_env_and_config",
        lambda *a: SimpleNamespace(
            s3_secret_name="test-secret", namespace="test", s3_prefix="runs", s3_bucket=bucket_override or ""
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_read_s3_credentials_from_secret",
        lambda *a: dict(
            bucket="secret-bucket",
            endpoint_url="http://s3.example",
            access_key="fake",
            secret_key="fake",
            region="test",
        ),
    )
    expected_bucket = bucket_override or "secret-bucket"
    uploads = []

    def upload_file(filename, bucket, key):
        if upload_fails:
            raise RuntimeError("S3 upload failed")
        uploads.append((bucket, key, Path(filename).read_bytes()))

    monkeypatch.setattr(adapter, "_create_s3_client", lambda **kw: SimpleNamespace(upload_file=upload_file))
    parameters = {"execution_mode": "kfp", "kfp_config": {"s3_prefix": "runs"}}
    if save_samples is not None:
        parameters["save_sample_results"] = save_samples
    job = SimpleNamespace(
        id="job",
        benchmark_id="trustyai_garak::quick",
        benchmark_index=0,
        model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
        parameters=parameters,
        exports=None,
    )
    callbacks = SimpleNamespace(report_status=lambda update: None)
    results = adapter.run_benchmark_job(job, callbacks)
    assert results.num_examples_evaluated == 2
    artifacts = results.evaluation_metadata["artifacts"]
    assert artifacts["scan_report"] == f"s3://{expected_bucket}/runs/job/execution-test/scan.report.jsonl"
    if upload_fails and save_samples is not False:
        assert "Skipping KFP sample results upload" in caplog.text
    if save_samples is False or upload_fails:
        assert not uploads
        assert "sample_results" not in artifacts
    else:
        assert uploads[0][:2] == (expected_bucket, "runs/job/execution-test/sample_results.jsonl")
        assert len(uploads[0][2].splitlines()) == 2
        assert artifacts["sample_results"] == f"s3://{expected_bucket}/runs/job/execution-test/sample_results.jsonl"


def test_intents_save_samples_by_default(monkeypatch, tmp_path):
    """Omitting the flag persists intent stubs rather than rejecting intents jobs."""
    module, adapter, job = TestSimpleIntentsMode()._make_adapter_and_job(monkeypatch, tmp_path)
    job.parameters.pop("save_sample_results")
    directories = []

    def run_intents(config, callbacks, garak_config_dict, scan_dir, timeout, intents_params):
        directories.append(scan_dir)
        report_prefix = scan_dir / "scan"
        report_prefix.with_suffix(".report.jsonl").write_text(
            json.dumps(
                {
                    "entry_type": "harness_stub_summary",
                    "intent": "fraud",
                    "source_stub": "stub",
                    "accepted": True,
                }
            )
        )
        return module.GarakScanResult(returncode=0, stdout="", stderr="", report_prefix=report_prefix)

    monkeypatch.setattr(adapter, "_run_simple_intents", run_intents)
    monkeypatch.setattr(adapter, "_parse_results", lambda *a, **kw: ([], 1.0, 1, {}))
    adapter.run_benchmark_job(job, SimpleNamespace(report_status=lambda update: None))
    record = json.loads((directories[0] / "sample_results.jsonl").read_text())
    assert record["sample_unit"] == "source_stub"
    assert record["attack_success_rate"] == 1.0


@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("failure", [None, "scan", "export", "empty", "disk", "oci"])
def test_sample_export_failures_and_scratch_lifecycle(monkeypatch, tmp_path, caplog, retain, failure):
    """Auxiliary export failures preserve metrics; scratch outlives persistence only when retained."""
    from llama_stack_provider_trustyai_garak.evalhub import sample_results

    module = _load_evalhub_garak_adapter(monkeypatch)
    monkeypatch.delenv("GARAK_RETAIN_SCAN_ARTIFACTS")
    if retain:
        monkeypatch.setenv("GARAK_RETAIN_SCAN_ARTIFACTS", "true")
    monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))
    adapter = module.GarakAdapter()
    directories, statuses, persisted = [], [], []

    def run_scan(config, callbacks, garak_config_dict, scan_dir, timeout):
        directories.append(scan_dir)
        report = {
            "entry_type": "attempt",
            "status": 2,
            "uuid": "attempt",
            "probe_classname": "dan.Test",
            "outputs": ["yes", "no"],
            "detector_results": {"primary": [1, 0]},
        }
        if failure == "export":
            report["detector_results"] = {}
        prefix = scan_dir / "scan"
        prefix.with_suffix(".report.jsonl").write_text("" if failure == "empty" else json.dumps(report))
        return module.GarakScanResult(
            returncode=1 if failure == "scan" else 0,
            stdout="",
            stderr="scan failure" if failure == "scan" else "",
            report_prefix=prefix,
        )

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    if failure == "disk":
        monkeypatch.setattr(sample_results, "save_sample_results", fail_write)
    monkeypatch.setattr(adapter, "_run_simple", run_scan)
    monkeypatch.setattr(adapter, "_parse_results", lambda *a, **kw: (["metric"], 0.75, 2, {}))
    mlflow = types.ModuleType("evalhub.adapter.mlflow")
    mlflow.MlflowArtifact = lambda name, data, media: SimpleNamespace(name=name, data=data)
    monkeypatch.setitem(sys.modules, "evalhub.adapter.mlflow", mlflow)

    def persist_oci(spec):
        assert spec.files_path == directories[0]
        assert (directories[0] / "scan.report.jsonl").exists()
        if failure == "oci":
            raise RuntimeError("OCI unavailable")
        persisted.append("oci")
        return SimpleNamespace(reference="oci://ref", digest="sha256:test")

    def persist_mlflow(results, config, artifacts):
        assert directories[0].exists()
        assert ("sample_results.jsonl" in [artifact.name for artifact in artifacts]) == (
            failure not in {"export", "empty", "disk"}
        )
        persisted.append("mlflow")

    callbacks = SimpleNamespace(
        report_status=statuses.append, create_oci_artifact=persist_oci, mlflow=SimpleNamespace(save=persist_mlflow)
    )
    job = SimpleNamespace(
        id="cleanup-job",
        benchmark_id="trustyai_garak::quick",
        benchmark_index=0,
        model=SimpleNamespace(url="http://model.example/v1", name="model"),
        parameters={},
        exports=SimpleNamespace(oci=SimpleNamespace(coordinates="registry.example/artifact:test")),
    )
    if failure in {"scan", "oci"}:
        with pytest.raises(RuntimeError):
            adapter.run_benchmark_job(job, callbacks)
    else:
        result = adapter.run_benchmark_job(job, callbacks)
        assert result.overall_score == 0.75
        assert result.results == ["metric"]
        assert persisted == ["oci", "mlflow"]
        assert not any(getattr(status, "status", None) == module.JobStatus.FAILED for status in statuses)
        if failure in {"export", "empty", "disk"}:
            assert "Skipping sample results export" in caplog.text
    assert directories[0].exists() == retain


@pytest.mark.parametrize("failed", [False, True])
def test_kfp_execution_uses_owned_scratch_and_cleans_it(monkeypatch, tmp_path, failed):
    """KFP shares the caller's scratch directory, including when submission fails."""
    module = _load_evalhub_garak_adapter(monkeypatch)
    monkeypatch.delenv("GARAK_RETAIN_SCAN_ARTIFACTS")
    monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))
    adapter = module.GarakAdapter()
    directories = []

    def run_kfp(config, callbacks, garak_config_dict, timeout, scan_dir, **kwargs):
        directories.append(scan_dir)
        assert scan_dir.exists()
        if failed:
            raise RuntimeError("pipeline failed")
        prefix = scan_dir / "scan"
        prefix.with_suffix(".report.jsonl").write_text("{}")
        return module.GarakScanResult(returncode=0, stdout="", stderr="", report_prefix=prefix), scan_dir

    monkeypatch.setattr(adapter, "_run_via_kfp", run_kfp)
    monkeypatch.setattr(adapter, "_parse_results", lambda *a, **kw: ([], 0.25, 0, {}))
    job = SimpleNamespace(
        id="kfp-cleanup",
        benchmark_id="trustyai_garak::quick",
        benchmark_index=0,
        model=SimpleNamespace(url="http://model.example/v1", name="model"),
        parameters={"execution_mode": "kfp"},
        exports=None,
    )
    callbacks = SimpleNamespace(report_status=lambda update: None)
    if failed:
        with pytest.raises(RuntimeError, match="pipeline failed"):
            adapter.run_benchmark_job(job, callbacks)
    else:
        assert adapter.run_benchmark_job(job, callbacks).overall_score == 0.25
    assert not directories[0].exists()
    assert not list((tmp_path / "kfp-cleanup").glob("execution-*"))
