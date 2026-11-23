"""Compile the eval-hub KFP pipeline without a cluster.

KFP mode is otherwise only exercised on an OpenShift cluster with Data Science
Pipelines. Compiling catches broken component signatures, unserialisable defaults
and wiring mistakes in unit CI, and checks the base-image override that a PR image
is injected through.

The compile runs in a subprocess because ``kfp_pipeline`` resolves the component
image at import time (env var, then a ConfigMap on the current kube context), so
the test pins the image and hides any local kubeconfig.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

kfp = pytest.importorskip("kfp")

TEST_IMAGE = "example.invalid/garak-provider:ci-compile"
EXPECTED_COMPONENTS = {
    "validate",
    "resolve-taxonomy",
    "sdg-generate",
    "prepare-prompts",
    "garak-scan",
    "write-kfp-outputs",
}

_COMPILE = """
import sys
from kfp import compiler
from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import evalhub_garak_pipeline
compiler.Compiler().compile(evalhub_garak_pipeline, sys.argv[1])
"""


@pytest.fixture(scope="module")
def compiled_pipeline(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("kfp") / "pipeline.yaml"
    env = {
        **os.environ,
        "KUBEFLOW_GARAK_BASE_IMAGE": TEST_IMAGE,
        "KUBECONFIG": str(out.parent / "no-kubeconfig"),
    }
    env.pop("KUBERNETES_SERVICE_HOST", None)
    proc = subprocess.run(
        [sys.executable, "-c", _COMPILE, str(out)], env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"pipeline failed to compile:\n{proc.stderr[-4000:]}"
    # The first document is the pipeline spec; kfp-kubernetes appends a platform spec document.
    return next(yaml.safe_load_all(Path(out).read_text()))


def test_pipeline_has_every_component(compiled_pipeline: dict) -> None:
    executors = compiled_pipeline["deploymentSpec"]["executors"]
    names = {name.removeprefix("exec-") for name in executors}
    assert names == EXPECTED_COMPONENTS


def test_every_component_uses_the_injected_image(compiled_pipeline: dict) -> None:
    images = {
        name: spec["container"]["image"] for name, spec in compiled_pipeline["deploymentSpec"]["executors"].items()
    }
    assert set(images.values()) == {TEST_IMAGE}, images


def _adapter_pipeline_arguments() -> set[str]:
    """Keys the adapter submits: the ``pipeline_args`` literal plus later ``pipeline_args[...] =``."""
    import ast

    import llama_stack_provider_trustyai_garak.evalhub.garak_adapter as adapter_module

    tree = ast.parse(Path(adapter_module.__file__).read_text())
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "pipeline_args":
            assert isinstance(node.value, ast.Dict)
            keys.update(k.value for k in node.value.keys if isinstance(k, ast.Constant))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and getattr(target.value, "id", None) == "pipeline_args"
                    and isinstance(target.slice, ast.Constant)
                ):
                    keys.add(target.slice.value)
    assert keys, "could not find pipeline_args in garak_adapter.py; update this test"
    return keys


def test_pipeline_inputs_match_what_the_adapter_submits(compiled_pipeline: dict) -> None:
    """KFP rejects unknown arguments and requires inputs without defaults."""
    inputs = compiled_pipeline["root"]["inputDefinitions"]["parameters"]
    submitted = _adapter_pipeline_arguments()
    unknown = submitted - set(inputs)
    assert not unknown, f"adapter submits arguments the pipeline does not declare: {sorted(unknown)}"
    required = {name for name, spec in inputs.items() if not spec.get("isOptional")}
    missing = required - submitted
    assert not missing, f"pipeline requires inputs the adapter never submits: {sorted(missing)}"
