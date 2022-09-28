#!/bin/bash

for SEED in {1..10}
do  
    echo $SEED
    sbatch ./hpc_scripts/pretrain_seed.sh $SEED
done