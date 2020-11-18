#!/bin/bash
#SBATCH -n 20
#SBATCH -p gpu-2080Ti
BATCH=$1
GPUS=$2
SBN=$3

#nsys profile -f true -o outputs/1008/Resnet50-B$BATCH-GPU$GPU-$SBN -c cudaProfilerApi --stop-on-range-end true --export sqlite python ./multiproc.py --nproc_per_node 4 ./main.py --arch resnet50 -b $BATCH --training-only -p 10 --prof 100 --epochs 1 $SBN /tmp/ILSVRC2012/
#nvprof -o outputs/1029/Resnext101-B$BATCH-GPU$GPUS-$SBN%p.sql --profile-from-start off --profile-child-processes python ./multiproc.py --nproc_per_node $GPUS ./main.py --arch resnext101-32x4d -b $BATCH --training-only -p 10 --prof 100 --epochs 1 $SBN /home/shared/ILSVRC2012/
dlprof -f true --mode=pytorch --reports=kernel --iter_start=30  python ./multiproc.py --nproc_per_node $GPUS ./main.py --arch resnext101-32x4d -b $BATCH --training-only -p 10 --prof 100 --epochs 1 $SBN /home/shared/ILSVRC2012/
