#!/bin/bash
set -euo pipefail

# setup-dev.sh - Prepare offline development environment for Energy Transformer
# This script installs system packages, Poetry, and all Python dependencies.
# Run once with internet access. Subsequent runs are safe and skip installed items.

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# Detect OS (apt-based assumed)
if ! command -v apt-get >/dev/null; then
  echo "Error: apt-get not available. Please install dependencies manually." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential curl git python3 python3-venv python3-dev

# Install Poetry if missing
if ! command -v poetry >/dev/null; then
  curl -sSL https://install.python-poetry.org | python3 - --yes
  export PATH="$HOME/.local/bin:$PATH"
fi

# Ensure Poetry uses in-project virtualenv
poetry config virtualenvs.in-project true --local

# Install project dependencies with all extras
poetry install --with dev,examples,docs --no-interaction

# Verification
poetry run python - <<'PY'
import torch
import energy_transformer as et
print("Energy Transformer", et.__version__)
print("PyTorch", torch.__version__)
PY

# Run a short test suite to confirm environment works
poetry run pytest tests/unit -k "init" -q

cat <<'MSG'
Setup complete. Activate the virtual environment with:
  source .venv/bin/activate
MSG

