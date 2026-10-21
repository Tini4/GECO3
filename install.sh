#!/usr/bin/bash
set -eo pipefail

export PATH="$HOME/miniforge3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n test_geco3 python=3.10 -y
conda activate test_geco3

conda install -c nvidia cuda-toolkit=12.6 -y
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

cd ./Deformable-DETR/models/ops
CUDA_VISIBLE_DEVICES=0 pip install --no-build-isolation .
cd ../../..

cp -r ./Deformable-DETR/models/ops ./models/ops

pip install hydra-core
pip install scikit-image
pip install pycocotools
pip install einops
pip install "numpy<2"

pip install gradio==4.44.1 fastapi==0.112.2 starlette==0.37.2
pip install gradio_image_prompter

pip install huggingface-hub==0.34.3
pip install --force-reinstall "pydantic<2.11"

