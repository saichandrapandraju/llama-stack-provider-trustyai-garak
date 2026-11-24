# Architecture

## Eval-Hub Garak Adapter

This repo provides a `FrameworkAdapter` for the eval-hub SDK used by the RHOAI
evaluation platform. It integrates [Garak](https://github.com/NVIDIA/garak)
red-teaming scans with Kubernetes-based evaluation jobs.

### Execution Modes

| Mode | Module | How It Works |
|------|--------|-------------|
| **Simple** | `evalhub/garak_adapter.py` | Garak runs directly inside the eval-hub K8s job pod. No KFP. |
| **KFP** | `evalhub/kfp_adapter.py` | K8s job submits a KFP pipeline, polls for completion, pulls artifacts from S3. Supports intents/SDG. |

Entry points: `evalhub/garak_adapter.py` → `GarakAdapter.main()`,
`evalhub/kfp_adapter.py` → `GarakKFPAdapter` (forces KFP mode).

Can also be invoked as `python -m llama_stack_provider_trustyai_garak.evalhub`.

## Intents Benchmark (Key Feature)

The `intents` benchmark uses SDG and adversarial probes for targeted risk assessment:

1. Loads a policy taxonomy dataset (CSV/JSON)
2. Runs SDG (synthetic data generation) via sdg-hub to generate test prompts
3. Scans the model with TAPIntent probes (tree-of-attacks with persuasion)
4. Evaluates responses with MulticlassJudge detector

Both execution modes support intents. They run the same steps from
`core/pipeline_steps.py`:

```
taxonomy → SDG → prompts → scan → parse
```

- **Simple mode** (`GarakAdapter._run_simple_intents`): the steps run one after
  another in the eval-hub job pod. A taxonomy (`policy_s3_key`) or pre-generated
  prompts file (`intents_s3_key`) is a local path under `/test_data`, which the
  eval-hub SDK fills from the job's `test_data_ref`.
- **KFP mode** (`evalhub/kfp_pipeline.py`): each step is a KFP component, and
  files pass between them through S3. The pipeline starts with a `validate`
  component (`validate_scan_config`: garak is importable, the config has a
  `plugins` section, no blocked flags), which simple mode does not run.

### Intents Model Configuration

Intents requires four model endpoints: judge, attacker, evaluator, and target.
These can be configured two ways:

1. **`intents_models` shorthand** — Provide model name/URL per role. The overlay
   fills in garak config using `x.get("key") or default` pattern (respects
   user-configured values, only fills empty slots). `api_key` is always set to
   `__FROM_ENV__` for K8s Secret injection.

2. **`garak_config` only** — Fully configure all model endpoints directly in
   `garak_config.plugins`. No overlay is applied.

## Code Layout

```
src/llama_stack_provider_trustyai_garak/
├── core/                    # Shared logic used by all modes
│   ├── config_resolution.py # Deep-merge user overrides onto benchmark profiles
│   ├── command_builder.py   # Build garak CLI args for OpenAI-compatible endpoints
│   ├── garak_runner.py      # Subprocess runner for garak CLI
│   └── pipeline_steps.py    # Six-step pipeline (validate→taxonomy→SDG→prompts→scan→parse)
│
├── evalhub/                 # Eval-Hub integration
│   ├── garak_adapter.py     # FrameworkAdapter: benchmark resolution, intents overlay, callbacks
│   ├── kfp_adapter.py       # KFP-specific adapter (forces KFP execution mode)
│   ├── kfp_pipeline.py      # Eval-hub KFP pipeline with S3 artifact flow
│   └── s3_utils.py          # S3/Data Connection client
│
├── garak_command_config.py  # Pydantic models for garak YAML config
├── config.py                # Scan profiles (OWASP, AVID, intents, etc.)
├── intents.py               # Policy taxonomy dataset loading (SDG/intents flows)
├── sdg.py                   # Synthetic data generation via sdg-hub
├── result_utils.py          # Parse garak outputs, TBSA scoring, HTML reports
├── constants.py             # Execution mode constants, default values
├── errors.py                # Exception hierarchy
├── utils.py                 # XDG path helpers, HTTP client
└── resources/               # Jinja2 templates and Vega chart specs
```

## Config Resolution Flow

```
Benchmark Profile (predefined in config.py)
        │
        ▼
  deep_merge_dicts(profile, user_overrides)
        │
        ▼
  Effective garak_config
        │
        ▼
  intents_models overlay (only for intents benchmark)
    - model type/name/uri: set if empty, preserve if user-configured
    - api_key: always forced to __FROM_ENV__ (K8s Secret)
    - max_tokens/temperature: setdefault (preserve existing)
        │
        ▼
  Final GarakCommandConfig → config.json file → garak --config
```

## Module Summary

| Module | Purpose | Used By |
|--------|---------|---------|
| `garak_command_config.py` | Pydantic models for garak YAML config | All modes |
| `config.py` | Scan profiles (GarakScanConfig, TapIntentConfig) | All modes |
| `intents.py` | Policy taxonomy dataset loading | Intents (both modes) |
| `sdg.py` | Synthetic data generation via sdg-hub | Intents (both modes) |
| `result_utils.py` | Parse garak outputs, TBSA scoring, HTML reports | All modes |
| `constants.py` | KFP image keys, execution mode constants | All modes |

## Dependency Extras

| Extra | Contents | Use Case |
|-------|----------|----------|
| (none) | Core adapter + KFP + boto3 + eval-hub-sdk | Production |
| `[sdg]` | nest_asyncio + sdg-hub | SDG without garak |
| `[inline]` | `[sdg]` + garak (pinned version) | Container builds |
| `[test]` | pytest + pytest-asyncio | CI test runner |
| `[dev]` | `[test]` + ruff + pre-commit | Local development |
