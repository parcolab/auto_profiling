#!/bin/bash
#SBATCH -p gpu-2080Ti
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH -o 'results/sbatch/slurm-%j.out'

COMMAND=$1
WORKING_DIR=$2
OUTPUT_PATH=$3
cd $WORKING_DIR
ncu --nvtx --profile-from-start on -f --csv --page raw  -o $OUTPUT_PATH $COMMAND > $OUTPUT_PATH.csv
