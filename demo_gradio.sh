#!/usr/bin/bash
set -eo pipefail

export PATH="$HOME/miniforge3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate test_geco3

python demo_gradio.py

