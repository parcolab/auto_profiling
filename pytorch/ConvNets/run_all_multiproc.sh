#!/bin/bash

sbatch run_multiproc1.sh 16 resnet101
sbatch run_multiproc1.sh 32 resnet101
sbatch run_multiproc1.sh 64 resnet101
sbatch run_multiproc1.sh 128 resnet101

sbatch run_multiproc1.sh 16 resnext101-32x4d
sbatch run_multiproc1.sh 32 resnext101-32x4d
sbatch run_multiproc1.sh 64 resnext101-32x4d
sbatch run_multiproc1.sh 128 resnext101-32x4d

sbatch run_multiproc2.sh 16 resnet101 29500
sbatch run_multiproc2.sh 32 resnet101 29501
sbatch run_multiproc2.sh 64 resnet101 29502
sbatch run_multiproc2.sh 128 resnet101 29503

sbatch run_multiproc2.sh 16 resnext101-32x4d 29504
sbatch run_multiproc2.sh 32 resnext101-32x4d 29505
sbatch run_multiproc2.sh 64 resnext101-32x4d 29506
sbatch run_multiproc2.sh 128 resnext101-32x4d 29507

sbatch run_multiproc4.sh 16 resnet101 29508
sbatch run_multiproc4.sh 32 resnet101 29509
sbatch run_multiproc4.sh 64 resnet101 29510
sbatch run_multiproc4.sh 128 resnet101 29511

sbatch run_multiproc4.sh 16 resnext101-32x4d 29512
sbatch run_multiproc4.sh 32 resnext101-32x4d 29513
sbatch run_multiproc4.sh 64 resnext101-32x4d 29514
sbatch run_multiproc4.sh 128 resnext101-32x4d 29515


