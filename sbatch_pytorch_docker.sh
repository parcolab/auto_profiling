#!/bin/bash
#SBATCH -p A100

CMD=$1
docker run -it --rm \ 
    --gpus device=$CUDA_VISIBLE_DEVICES \
    -v /home/hhk971/data:/data \
    nvcr.io/nvidia/pytorch:20.10-py3 \
    $CMD
