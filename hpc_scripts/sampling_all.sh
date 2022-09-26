#!/bin/bash

for AGENT in aps diayn disagreement icm_apt icm proto rnd smm
do
    sbatch ./hpc_scripts/sampling.sh $AGENT
done
