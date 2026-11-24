# Kubeflow ConfigMap keys and defaults for base image resolution
GARAK_PROVIDER_IMAGE_CONFIGMAP_NAME = "trustyai-service-operator-config"
GARAK_PROVIDER_IMAGE_CONFIGMAP_KEY = (
    "garak-provider-image"  # from https://github.com/opendatahub-io/opendatahub-operator/pull/2567
)
DEFAULT_GARAK_PROVIDER_IMAGE = "quay.io/trustyai/trustyai-garak-lls-provider-dsp@sha256:c960230103493b8dece955012b0a34105c185dd4ac5a3034460b50efa8303084"
KUBEFLOW_CANDIDATE_NAMESPACES = ["redhat-ods-applications", "opendatahub"]

# Default values
DEFAULT_TIMEOUT = 0
# Stop a scan when garak has retried failed model requests (5xx, 429, timeouts) for this long
# with no progress. Unlike DEFAULT_TIMEOUT it doesn't depend on scan size or model latency.
DEFAULT_MAX_RETRY_SECONDS = 1800
DEFAULT_MODEL_TYPE = "openai.OpenAICompatible"
DEFAULT_EVAL_THRESHOLD = 0.5

# evalhub adapter
EXECUTION_MODE_SIMPLE = "simple"
EXECUTION_MODE_KFP = "kfp"
TARGET_DEFAULT_PARAMETERS = {"max_tokens": 512}
# XDG variables
XDG_CACHE_HOME = "/tmp/.cache"
XDG_DATA_HOME = "/tmp/.local/share"
XDG_CONFIG_HOME = "/tmp/.config"

# SDG variables
DEFAULT_SDG_FLOW_ID = "major-sage-742"
DEFAULT_SDG_MAX_CONCURRENCY = 10
DEFAULT_SDG_NUM_SAMPLES = 0
DEFAULT_SDG_NUM_SAMPLES_BLOCK_NAME = "replicate_rows"
DEFAULT_SDG_MAX_TOKENS = 0
DEFAULT_SDG_MAX_TOKENS_BLOCK_NAME = "generate_adversarial_prompt"
