#!/bin/bash

export PATH="$HOME/miniforge3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate test_geco3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python inference.py \
--model_name GECO2FSCD \
--data_path ./data/fsc147 \
--model_path ./checkpoints/ \
--backbone resnet50 \
--reduction 16 \
--image_size 1024 \
--emb_dim 256 \
--num_heads 8 \
--kernel_dim 1 \
--num_objects 3 \
--batch_size 1
