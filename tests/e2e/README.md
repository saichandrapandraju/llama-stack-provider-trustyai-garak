# EvalHub contract e2e tests

These tests run the real stack, without a cluster:

```
pytest ──REST──▶ eval-hub-server -local ──sh -c──▶ garak adapter ──▶ garak ──▶ mock model
                        ▲                               │
                        └──────── /events callbacks ────┘
```

The unit tests mock garak and the eval-hub SDK. These don't, so they catch what
the unit tests can't:

- **eval-hub server or SDK changes**: the job spec, status and result callbacks,
  and how the server stores metrics and applies `pass_criteria`.
- **Catalog drift**: a benchmark eval-hub offers that has no adapter profile, or
  a different primary score.
- **New benchmarks**: the sweep runs every benchmark that has no dedicated test
  (all except `quick` and `intents`) with no test changes. That includes a profile
  added to the adapter before eval-hub's catalog lists it: the tests serve it under
  a stand-in catalog entry (copied from a real taxonomy entry), sweep it as
  `<id>-adapter-only`, and print a warning naming it until it is in eval-hub's
  `garak.yaml`. A benchmark that needs extra setup, as `intents` does, needs its
  own test.
- **garak changes**: garak really runs, and the tests check that every probe it
  queued reported a result.

The model is [`mock_openai_server.py`](mock_openai_server.py), an OpenAI-compatible
server with fixed answers. It picks its answer from the model name:

- **Targets:** `mock-refuser` always refuses and `mock-complier` always plays along
  with jailbreaks, so a test can assert the attack success rate as well as completion.
- **Intents helpers:** `mock-sdg`, `mock-attacker`, `mock-evaluator`, `mock-judge`
  and `mock-translator` answer in the formats sdg-hub, TAP, MulticlassJudge and
  garak's translator parse. That lets the **full intents profile** run end to end
  (SDG, SPO, translation, TAP, judge) in seconds. The judge and evaluator score a
  response as compliant only if it carries a marker that only `mock-complier` emits,
  so the refuser scores 0 and the complier scores above 0.

The mock requires an API key, so every job also tests credential handling.

## Where versions come from

Nothing is hardcoded in the tests or the workflow:

| What | Source |
| --- | --- |
| Runtime packages, including garak and eval-hub-sdk | `requirements.txt` |
| garak | The fork's git tag for the version locked in `requirements.txt` |
| Test tools: pytest and `eval-hub-server` (the server binary the tests run) | The `test` extra in `pyproject.toml` |
| The garak catalog (benchmarks, primary score, pass criteria) | eval-hub's `config/providers/garak.yaml` at the tag matching the installed `eval-hub-server` |

[`scripts/install-locked-deps.sh`](../../scripts/install-locked-deps.sh) installs
the locked set from public indexes; it never uses the production index that
`requirements.txt` names. With `--extras test` it adds the test tools,
constrained by the lock, so they can't change a locked package. Don't use
`pip install ".[test]"`: that re-resolves the project's dependencies against
public metadata and silently changes locked versions (kubernetes 36 becomes 30).

Bumping eval-hub means changing the `eval-hub-server` pin in the `test` extra.
The catalog follows the version automatically.

## Running locally

```bash
scripts/install-locked-deps.sh --extras test
pip install --no-deps -e .

make e2e             # the contract suite, about a minute
make e2e-sweep       # every taxonomy benchmark, capped as on PRs (minutes; downloads detector models)
make e2e-sweep-full  # every taxonomy benchmark, full profile as shipped (as nightly)
```

To test an image instead, the way the PR job does, see the header of
[`Containerfile`](Containerfile) in this directory.

| Variable | Purpose |
| --- | --- |
| `EVALHUB_SERVER_BIN` | eval-hub server binary to use, e.g. one built from an eval-hub checkout (`go build -o eval-hub ./cmd/eval_hub`). Defaults to `eval-hub-server` on `PATH`. |
| `E2E_GARAK_PROVIDER_YAML` | Path or URL of the garak provider definition. Defaults to eval-hub's catalog at the tag matching the installed `eval-hub-server`. Required with `EVALHUB_SERVER_BIN`. |
| `E2E_REQUIRED=1` | Fail instead of skip when the stack isn't installed. CI and `make e2e` set it. |
| `E2E_SWEEP` | `capped` runs the taxonomy benchmarks with one prompt per probe, one generation and extended detectors off. `full` runs each profile as shipped. Unset skips them. |
| `E2E_ARTIFACTS_DIR` | Where the server log, each job's `job.json` and the adapter `jobrun.log` are copied. |

Without the stack installed, these tests are skipped. `make test` excludes them.

## In CI

Two workflows: [`e2e-evalhub.yml`](../../.github/workflows/e2e-evalhub.yml) runs on PRs and pushes to main, and [`e2e-nightly.yml`](../../.github/workflows/e2e-nightly.yml) runs nightly, on demand, or on a labelled PR.

| Job | When | What |
| --- | --- | --- |
| `image` | PRs, pushes to main | Builds the provider image from the commit and runs the tests inside it, as its runtime user, against the adapter installed in the image. Covers the contract suite, full intents on the mock, and every taxonomy benchmark capped. |
| `canary (evalhub-main)` | nightly | The locked set, with eval-hub-sdk `main`, the server built from eval-hub `main` and the garak catalog from `main` swapped in |
| `canary (garak-head)` | nightly | The locked set, with the garak fork's `automated-red-teaming` branch head swapped in |
| `sweep` | nightly | Every taxonomy benchmark, full profile as shipped, inside the image |

Detector models and garak data are cached. The nightly full sweep, which downloads every detector, saves a fresh cache each night, and PR jobs restore the newest one for the current `requirements.txt`. A PR job saves only when no cache could be restored.
Dependency changes are tested when a PR changes `requirements.txt`. The nightly
canaries change only the component under test, so a failure points at that component.

If a nightly run fails, the workflow opens an issue labelled `e2e-nightly`, or
comments on the open one. Logs are uploaded as run artifacts.

To run the nightly jobs against a PR before it merges, add the `run-nightly-e2e`
label. They run on that event and on every later push while the label stays,
which is useful for garak bumps, dependency changes and SDK upgrades. The
"open an issue on failure" step runs only on the schedule.

## Testing an existing image

The suite can run inside any build of the provider, not just one built from the
checkout. The shared step
[`.github/actions/e2e-image`](../../.github/actions/e2e-image/action.yml) either
builds `Containerfile` or pulls a given image, then adds the test extra on top.

- **Manually:** run *Tier 2 - EvalHub E2E* (contract tests and capped benchmarks)
  or *Tier 2 - EvalHub E2E (nightly)* (full sweep) from the Actions tab, and set
  `image`. For example, `quay.io/opendatahub/odh-trustyai-garak-lls-provider-dsp:dev`
  is the ODH image.
- **From another workflow:** `uses: ./.github/workflows/e2e-evalhub.yml` with
  `image`, `runtime_user`, `ref`, and for a private image `registry` plus the
  `registry_username`/`registry_password` secrets.

Set `runtime_user` to the user the image runs as: `1001` for the upstream and
ODH images, `0` for the downstream Konflux image, which ends as `USER root`.

Scheduled runs (the nightly jobs and the garak drift check) run only in the
upstream repo. Midstream and downstream get these files by sync, but don't run
the schedule.

## Known bugs pinned by these tests

Both of these tests are marked `xfail(strict=True)`. When the bug is fixed the
test passes unexpectedly, which fails the run, so remember to remove the marker
when you fix one.

- `test_unreachable_model_fails_the_job` and `test_rejected_credential_fails_the_job`:
  failed scans are reported as `completed` with `pass: true`
  ([#190](https://github.com/trustyai-explainability/llama-stack-provider-trustyai-garak/issues/190)).

## Not covered here

KFP mode needs Data Science Pipelines. It's covered by the pipeline compile test
in the unit suite (`tests/test_kfp_pipeline_compile.py`) and by cluster testing.
The Kubernetes path (sidecar, operator, RBAC) is the next tier
([#194](https://github.com/trustyai-explainability/llama-stack-provider-trustyai-garak/issues/194)).
