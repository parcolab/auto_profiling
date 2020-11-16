#!/bin/bash
#SBATCH -p gpu-2080Ti
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH -o 'results/sbatch/slurm-%j.out'

COMMAND=$1
WORKING_DIR=$2
OUTPUT_PATH=$3
cd $WORKING_DIR
dlprof --mode=pytorch --reports=kernel,detail --formats=csv  --force=true  --iter_start=30 --iter_stop=130 --output_path=$OUTPUT_PATH $COMMAND
