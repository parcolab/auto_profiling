#!/bin/bash
#SBATCH -n 10
#SBATCH -p gpu-2080Ti
#SBATCH --gres=gpu:1
BATCH=$1
GPUS=$2
SBN=$3
#export CUDA_VISIBLE_DEVICES=3
#nsys profile -f true -o outputs/1029/Resnet50-B$BATCH-GPU$GPU-$SBN -c cudaProfilerApi --stop-on-range-end true --export sqlite python ./multiproc.py --nproc_per_node 4 ./main.py --arch resnet50 -b $BATCH --training-only -p 10 --prof 100 --epochs 1 $SBN /home/shared/ILSVRC2012/
#nvprof -o output/EfficientDet_B$BATCH-Single%p.sql --profile-from-start off --profile-child-processes python train_prof.py --dataset VOC --dataset_root /home/shared/VOCdevkit/ --network efficientdet-d0 --batch_size 8 
#sh output/make_csv.sh
#dlprof --mode=pytorch --nsys_profile_range=true --reports=kernel --formats=csv --force true  python train_prof.py --dataset VOC --dataset_root /home/shared/VOCdevkit/ --network efficientdet-d0 --batch_size 16
export DYNAMIC_KERNEL_LIMIT_START=302272
export DYNAMIC_KERNEL_LIMIT_END=351540
#LD_PRELOAD=/home/hhk971/advanced_computer_architecture_project/accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so 
python train.py --dataset VOC --dataset_root /home/shared/VOCdevkit/ --network efficientdet-d0 --batch_size 16
