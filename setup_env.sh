#!/usr/bin/env bash
# Create / activate py310 environment for DiSIINet
set -euo pipefail

ENV_NAME="${DISIINET_ENV:-py310}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda env: $ENV_NAME (Python 3.10)"
  conda create -n "$ENV_NAME" python=3.10 pip -y
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install -r "$PROJECT_ROOT/requirements.txt"
echo "Environment ready: $ENV_NAME"
echo "Run: conda activate $ENV_NAME && cd $PROJECT_ROOT"
