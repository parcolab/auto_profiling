#!/bin/bash/
WORKING_DIR=$1
CMD=$2
export LD_LIBRARY_PATH=/opt/conda/lib/python3.6/site-packages/torch/lib/:$LD_LIBRARY_PATH
cd $WORKING_DIR
$CMD

