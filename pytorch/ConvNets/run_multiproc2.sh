#!/bin/bash
#SBATCH -n 5
#SBATCH -p gpu-2080Ti
#SBATCH --gres=gpu:2
BATCH=$1
GPUS=2
MODEL=$2
PORT=$3
#SBN=$3

mkdir -p outputs/$MODEL-$GPUS-$BATCH
#nsys profile -f true -o outputs/1029/Resnet50-B$BATCH-GPU$GPU-$SBN -c cudaProfilerApi --stop-on-range-end true --export sqlite python ./multiproc.py --nproc_per_node 4 ./main.py --arch resnet50 -b $BATCH --training-only -p 10 --prof 100 --epochs 1 $SBN /home/shared/ILSVRC2012/
#nvprof -o outputs/1014/Resnet50-mixed-B$BATCH-GPU$GPUS-$SBN%p.sql --profile-from-start off --profile-child-processes python ./multiproc.py --nproc_per_node $GPUS ./main.py --arch resnet50 -b $BATCH --training-only -p 10 --prof 100 --epochs 1 $SBN --amp /tmp/ILSVRC2012/
dlprof --mode=pytorch --reports=iteration  --iter_start=30 --iter_stop=129  --force true --output_path outputs/$MODEL-$GPUS-$BATCH python ./multiproc.py --nproc_per_node $GPUS  --master_port $PORT ./main.py --arch $MODEL -b $BATCH --training-only -p 10 --prof 140 --epochs 1 --prof-output outputs/$MODEL-$GPUS-$BATCH /home/shared/ILSVRC2012/
#export DYNAMIC_KERNEL_LIMIT_START=128677
#export DYNAMIC_KERNEL_LIMIT_END=145696
#LD_PRELOAD=/home/hhk971/advanced_computer_architecture_project/accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so python ./main.py --arch resnet50 -b 32 --training-only -p 10 --prof 50 --epochs 1 --data-backend pytorch /tmp/ILSVRC2012/
