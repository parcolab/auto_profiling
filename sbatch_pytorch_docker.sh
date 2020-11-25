#!/bin/bash
#SBATCH -p A100

docker exec hhkA100_8 sh /data/auto_profiling/docker_scripts.sh "$1" "$2"

