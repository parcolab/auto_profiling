#!/bin/bash
#SBATCH -n 10
#SBATCH -p gpu-2080Ti
BATCH=$1
GPUS=$2
SBN=$3
export CUDA_VISIBLE_DEVICES=3
nsys profile -f true -o efficient_det_B16 -c cudaProfilerApi --stop-on-range-end true --export sqlite python train_prof.py --dataset VOC --dataset_root /home/shared/VOCdevkit/ --network efficientdet-d0 --batch_size 16
#nsys profile -f true -o outputs/1029/Resnet50-B$BATCH-GPU$GPU-$SBN -c cudaProfilerApi --stop-on-range-end true --export sqlite python ./multiproc.py --nproc_per_node 4 ./main.py --arch resnet50 -b $BATCH --training-only -p 10 --prof 100 --epochs 1 $SBN /home/shared/ILSVRC2012/
#nvprof -o EfficientDet_B$BATCH-GPUS$GPUS%p.sql --profile-from-start off --profile-child-processes python train_prof.py --dataset VOC --dataset_root /home/shared/VOCdevkit/ --network efficientdet-d0 --batch_size 16 --multiprocessing-distributed
