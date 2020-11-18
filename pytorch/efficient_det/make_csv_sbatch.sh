#!/bin/bash
#SBATCH -p allcpu
f=$1
dict_file="${f%.sql}.dict"
csv_file="${f%.sql}.csv"
echo "Start parse " $f 
python -m pyprof.parse "$f" > "$dict_file"
python -m pyprof.prof --csv "$dict_file" > "$csv_file"

