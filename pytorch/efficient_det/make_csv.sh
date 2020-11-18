#!/bin/bash

for f in *.sql; 
do
    sbatch make_csv_sbatch.sh $f
done
