#!/bin/bash

for ENT_COEF in 0.01 0.1 1.0 10.0
do
    sbatch ./hpc_scripts/pretrain_smm.sh $ENT_COEF ${1}
done
