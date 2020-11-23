#!/bin/bash
BATCH=$1
GPUS=$2
MODEL=$3
SBN=$4

cd "$(dirname $0)"
OUTPUT_PATH=/data/outputs/$MODEL-$GPUS-$BATCH-$SBN
mkdir -p $OUTPUT_PATH
#dlprof --mode=pytorch --reports=iteration  --iter_start=30 --iter_stop=129  --force true --output_path /data/outputs/$MODEL-$GPUS-$BATCH-$SBN  python ./multiproc.py --nproc_per_node $GPUS  --master_port 8888 ./main.py --arch $MODEL -b $BATCH --training-only -p 10 --prof 140 --epochs 1 $SBN /data/ILSVR2012/ILSVRC2012
nsys profile -c cudaProfilerApi --stop-on-range-end true -t cuda,nvtx -s none --show-output=true -f true --export sqlite -o $OUTPUT_PATH/nsys_profile python ./multiproc.py --nproc_per_node $GPUS  --master_port 8888 ./main.py --arch $MODEL -b $BATCH --training-only -p 10 --prof 140 --epochs 1 $SBN /data/ILSVR2012/ILSVRC2012
