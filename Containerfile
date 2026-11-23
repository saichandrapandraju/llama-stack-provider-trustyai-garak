FROM registry.access.redhat.com/ubi9/python-312:latest
WORKDIR /opt/app-root

USER root

# Install the exact dependency set locked in requirements.txt (the single source of
# truth for versions) from public indexes: CPU torch, garak from the fork's git tag,
# everything else from PyPI. The production index in requirements.txt is reserved for
# product builds. Deps are their own layer, so source changes don't reinstall them.
# pyproject.toml comes along so the installer fails the build on a stale lock.
COPY requirements.txt pyproject.toml scripts/install-locked-deps.sh ./
RUN PIP_EXTRA_ARGS=--no-cache-dir ./install-locked-deps.sh requirements.txt && \
    rm install-locked-deps.sh

# Set XDG environment variables to use /tmp (always writable) for garak to write to
ENV XDG_CACHE_HOME=/tmp/.cache
ENV XDG_DATA_HOME=/tmp/.local/share
ENV XDG_CONFIG_HOME=/tmp/.config

# Allow nltk to use proxies to download words corpora for translation
ENV NLTK_ALLOW_PROXIED_URLOPEN=1

# Install our package on top of the locked deps
COPY src src
RUN pip install --no-cache-dir --no-deps .

# Drop to non-root for runtime (UBI9 uses 1001)
USER 1001

# Default CMD for eval-hub (runs as K8s Job, simple mode — garak in same pod)
# For KFP mode (garak in a separate KFP pod), use:
#   CMD ["python", "-m", "llama_stack_provider_trustyai_garak.evalhub.kfp_adapter"]
CMD ["python", "-m", "llama_stack_provider_trustyai_garak.evalhub"]