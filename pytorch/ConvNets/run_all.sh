#!/bin/bash
GPUS=$1
sbatch run_multiporc.sh 64 $GPUS "--sync-batch-norm"
sbatch run_multiporc.sh 128 $GPUS "--sync-batch-norm"
sbatch run_multiporc.sh 256 $GPUS "--sync-batch-norm"
sbatch run_multiporc.sh 64 $GPUS 
sbatch run_multiporc.sh 128 $GPUS 
sbatch run_multiporc.sh 256 $GPUS
