#!/bin/bash
#SBATCH -p A100
#SBATCH -o results/sbatch/slurm-%j.out
docker exec hhkA100_8 sh /data/auto_profiling/docker_scripts.sh "$1" "$2"

