#!/bin/bash

for SEED in {1..5}
do
    echo $SEED
    sbatch ./hpc_scripts/sac_train.sh $SEED
done
