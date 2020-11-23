#!/bin/bash
#SBATCH -n 20
#SBATCH -p gpu-2080Ti

nsys profile -f true -o outputs/1012/test  --export sqlite python ./main.py --arch resnet50 -b 8 --training-only -p 10 --prof 100 --epochs 1  /tmp/ILSVRC2012/
