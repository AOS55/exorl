#!/bin/bash

for SEED in {2..20}
do  
    echo $SEED
    sbatch ./hpc_scripts/pretrain_seed.sh $SEED
done