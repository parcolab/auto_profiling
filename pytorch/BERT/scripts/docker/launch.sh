#!/bin/bash
PROFILE_CMD=$1
CMD=${2:-/bin/bash}
NV_VISIBLE_DEVICES="all"
NV_VISIBLE_DEVICES="all"
DOCKER_BRIDGE="host"

echo "$PROFILE_CMD $CMD"
docker run --rm \
  --gpus device=$NV_VISIBLE_DEVICES \
  --net=$DOCKER_BRIDGE \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e LD_LIBRARY_PATH='/workspace/install/lib/' \
  -v $PWD:/workspace/bert \
  -v $PWD/results:/results -v /home/hhk971/data:/data bert bash scripts/run_pretraining.sh "$PROFILE_CMD" $CMD
