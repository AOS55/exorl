#!/bin/bash

for SKILL_DIM in 5 10 20 50 100 200 500
do
    sbatch ./hpc_scripts/pretrain_diayn.sh $SKILL_DIM
done