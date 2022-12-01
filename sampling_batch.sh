#!/bin/bash

echo seed ${1}
for SKILL_DIM in 50
    do for AGENT in smm
        do for SNAPSHOT_TS in 10000 50000 100000 500000 1000000 1500000 2000000 3000000 4000000 5000000 6000000 7000000 8000000 9000000 10000000 11000000 12000000 13000000 14000000 15000000 16000000
            do python prioritized_sampling.py seed=${1} skill_dim=$SKILL_DIM agent=$AGENT obs_type=states task=SimplePointBot_goal data_type=unsupervised snapshot_ts=$SNAPSHOT_TS
        done
    done
done