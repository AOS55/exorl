#!/bin/bash

for SEED in 1 2 3 4 5
do
    sbatch ./hpc_scripts/pretrain_smm.sh $SEED
done
