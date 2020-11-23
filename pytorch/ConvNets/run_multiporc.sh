#!/bin/bash
#SBATCH -n 20
#SBATCH -p gpu-2080Ti
BATCH=$1
GPUS=$2
SBN=$3

#nsys profile -f true -o outputs/1029/Resnet50-B$BATCH-GPU$GPU-$SBN -c cudaProfilerApi --stop-on-range-end true --export sqlite python ./multiproc.py --nproc_per_node 4 ./main.py --arch resnet50 -b $BATCH --training-only -p 10 --prof 100 --epochs 1 $SBN /home/shared/ILSVRC2012/
#nvprof -o outputs/1014/Resnet50-mixed-B$BATCH-GPU$GPUS-$SBN%p.sql --profile-from-start off --profile-child-processes python ./multiproc.py --nproc_per_node $GPUS ./main.py --arch resnet50 -b $BATCH --training-only -p 10 --prof 100 --epochs 1 $SBN --amp /tmp/ILSVRC2012/
#dlprof --mode=pytorch --reports=iteration  --iter_start=30 --iter_stop=129 python ./main.py --arch resnet101 -b 32 --training-only -p 10 --prof 200 --epochs 1 /home/shared/ILSVRC2012/
LD_PRELOAD=/home/hhk971/advanced_computer_architecture_project/accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so python ./main.py --arch resnext101-32x4d -b 32 --training-only -p 10 --prof 200 --epochs 1 --data-backend pytorch /home/shared/ILSVRC2012/
