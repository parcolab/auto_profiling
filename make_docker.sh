#!/bin/bash
#SBATCH -p A100
docker run --gpus all \
    -it \
    -v /home/hhk971/data:/data \
    --ipc=host \
    --name=hhkA100_8 \
    nvcr.io/nvidia/pytorch:20.10-py3 bash
    
