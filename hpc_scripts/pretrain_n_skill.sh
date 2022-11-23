#!/bin/bash

echo agent ${1}
for SKILL_DIM in 5 10 20 50 100 200 500 1000
do
    sbatch ./hpc_scripts/pretrain.sh $SKILL_DIM ${1}
done