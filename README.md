# TrustyAI Garak: LLM Red Teaming for Eval-Hub

Automated vulnerability scanning and red teaming for Large Language Models using [Garak](https://github.com/NVIDIA/garak). This package provides a `FrameworkAdapter` for the [eval-hub](https://github.com/opendatahub-io/eval-hub) evaluation platform, enabling garak scans to run as Kubernetes Jobs or via Kubeflow Pipelines.

## What It Does

- 🔍 **Vulnerability Assessment**: Red-team LLMs for prompt injection, jailbreaks, toxicity, bias and other vulnerabilities
- 📋 **Compliance**: OWASP LLM Top 10, AVID taxonomy benchmarks
- 🎯 **Intents-based Testing**: Policy taxonomy + SDG + TAPIntent for targeted risk assessment (simple and KFP modes)
- ☁️ **Cloud-Native**: Runs on OpenShift AI / Kubernetes as eval-hub jobs
- 📊 **Detailed Reports**: JSONL, HTML, and AVID-format reports with MLflow integration

## Execution Modes

| Mode | How Garak Runs | Intents Support | Use Case |
|------|---------------|-----------------|----------|
| **Simple** | Directly in the eval-hub K8s Job pod | Yes | Standard and intents scans, no extra infrastructure |
| **KFP** | K8s Job submits to Kubeflow Pipelines, polls status | Yes | Running garak and SDG on Data Science Pipelines, with artifacts in S3 |

Both modes run the intents benchmark through the same steps (taxonomy → SDG → prompts →
scan → results). Simple mode runs them one after another in the job pod; a taxonomy or
pre-generated prompts file comes from the job's test data (`test_data_ref`, mounted at
`/test_data`), and `policy_s3_key` / `intents_s3_key` take its local path. KFP mode runs
them as pipeline components that pass files through S3.

## Installation

```bash
# Core (eval-hub adapter with KFP support)
pip install llama-stack-provider-trustyai-garak

# With SDG support (for intents workflows)
pip install "llama-stack-provider-trustyai-garak[sdg]"

# Development
pip install "llama-stack-provider-trustyai-garak[dev]"
```

## Container Image

```bash
# Build the eval-hub adapter image
docker build -f Containerfile -t trustyai-garak:dev .
```

The container runs as:
```bash
# Simple mode (garak in same pod)
CMD ["python", "-m", "llama_stack_provider_trustyai_garak.evalhub"]

# KFP mode (garak in a separate KFP pod)
CMD ["python", "-m", "llama_stack_provider_trustyai_garak.evalhub.kfp_adapter"]
```

## Benchmark Profiles

Predefined scan profiles available via `benchmark_id`:

| Profile | Description |
|---------|-------------|
| `quick` | Single DAN probe for fast testing |
| `owasp_llm_top10` | OWASP Top 10 for LLM Applications |
| `avid` | AVID taxonomy — all vulnerabilities |
| `avid_security` | AVID — security vulnerabilities |
| `avid_ethics` | AVID — ethical concerns |
| `avid_performance` | AVID — performance issues |
| `quality` | Violence, profanity, toxicity, hate speech |
| `cwe` | Common Weakness Enumeration |
| `intents` | Intents-based risk assessment (policy taxonomy + SDG + TAPIntent) |

## Job Spec Configuration

The adapter reads a `JobSpec` from a mounted ConfigMap:

```json
{
  "id": "scan-001",
  "provider_id": "garak",
  "benchmark_id": "quick",
  "model": {
    "url": "https://my-model-endpoint.example.com/v1",
    "name": "my-model"
  },
  "parameters": {
    "probes": "dan.Dan_11_0",
    "execution_mode": "simple"
  }
}
```

## Parameters

The SDK's top-level `JobSpec.num_examples` is also honored as a soft per-probe
prompt cap. It takes precedence over the legacy `parameters.num_examples` alias.
Explicit `soft_probe_prompt_cap` settings, either flat or under `garak_config.run`,
retain their precedence over the sample-limit alias.

Key `parameters` in the job spec:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `probes` | Comma-separated probe names | Profile default |
| `probe_tags` | Tag-based probe filtering | — |
| `save_sample_results` | Persist per-sample JSONL; set explicitly to `false` to disable | `true` |
| `execution_mode` | `simple` or `kfp` | `simple` |
| `timeout_seconds` | Scan timeout | Profile default |
| `soft_probe_prompt_cap` | Preferred prompt limit per probe; some probes may exceed it | `256` |
| `num_examples` | Compatibility alias for the soft per-probe prompt cap; explicit caps take precedence | — |
| `eval_threshold` | Vulnerability threshold (0.0–1.0) | `0.5` |
| `model_type` | Garak generator type | `openai.OpenAICompatible` |
| `garak_config` | Full garak config dict (deep-merged onto profile) | — |

Unless `save_sample_results: false` is set, the adapter writes
`sample_results.jsonl` before OCI and MLflow persistence. KFP mode also uploads
it to S3 and includes its URI in artifact metadata. Each execution uses a
separate local directory and KFP S3 prefix, so retries cannot reuse stale files.

Standard scans export one record per completed response with a report-local
`sample_id` (`attempt UUID:response index`), `probe`, `response`, `detector`,
`detector_score`, and numeric `attack_success_rate` (0 or 1). Each probe uses the
detector with the most response-level threshold hits, matching standard AVID's
conservative ASR aggregation. Nested score arrays are flattened; incomplete or
non-finite scores skip the sample export with a warning. Sample-export and KFP
sample-upload failures preserve the completed evaluation and its aggregate results;
failed uploads are not advertised in S3 artifact metadata. Metadata and intermediate
attempt snapshots are excluded.

The JSONL mean measures response-level success for the selected detector. It is
not generally equal to the reported metric: the fallback aggregate counts
attempts with a hit from **any** detector. Detectors disagreeing or attempts
containing multiple responses can therefore produce different values. AVID-based
aggregation uses its own per-probe aggregation and weighting.

Attempt UUIDs are generated by Garak for each scan. Standard sample IDs are stable
only within that report and copies of its export, not across retries or reruns.
Calibration must use IDs from the same exported report (for example, a labeled
subset); do not join calibration from an independent scan by these IDs.

Intents export one observation per `(intent, source_stub)` with a stable hashed
`sample_id`, `intent`, `source_stub`, `sample_unit: source_stub`, and numeric
`attack_success_rate`. Final harness summaries are authoritative; older reports
combine acceptance across attempts for the same stub. Exports stream to a
temporary file and are published atomically after validation.

Scan files now live under `GARAK_SCAN_DIR/<job_id>/execution-<random>/`, including
`scan.report.jsonl` and `sample_results.jsonl`; consumers of the old flat
`<job_id>/scan.report.jsonl` layout must use the execution directory instead.
The adapter removes its execution directory after all persistence callbacks
finish, on success or failure. Set `GARAK_RETAIN_SCAN_ARTIFACTS=true` to retain
local files for debugging; operators enabling retention must manage disk usage.
Use configured OCI/MLflow artifacts or KFP S3 URIs for durable access.

Prompt caps do not limit the number of selected probes or guarantee a total request count.

SDG accepts bare model IDs such as `Qwen/Qwen3-VL-30B-A3B-Instruct` and routes
them through LiteLLM's `openai/` provider at `sdg_api_base`. Explicit LiteLLM
provider prefixes are preserved.

For a raw model ID whose namespace is also a LiteLLM provider, supply the
provider separately: `sdg_model: "openai/gpt-oss-120b"` with
`sdg_provider: "openai"` preserves the complete `openai/gpt-oss-120b` ID in the
endpoint request. The same setting is available as
`intents_models.sdg.provider`. When an explicit provider is set, the complete
model value is treated as a raw ID; otherwise legacy provider-prefixed values
keep their existing meaning.

## Results

The adapter reports `EvaluationResult` per probe with:
- `attack_success_rate`: Percentage of successful attacks
- `vulnerable_responses`: Count of vulnerable responses
- `total_attempts`: Total probe attempts

Overall metrics include TBSA (Tier-Based Security Aggregate) when available.

## Development

```bash
make test        # Unit tests
make e2e         # EvalHub contract tests: real eval-hub server + garak (tests/e2e/README.md)
make coverage    # With coverage report
make lint        # ruff check
make format      # ruff format
```

## Support & Documentation

- 📖 **Garak Docs**: https://reference.garak.ai/en/latest/index.html
- 💬 **Issues**: https://github.com/trustyai-explainability/llama-stack-provider-trustyai-garak/issues
