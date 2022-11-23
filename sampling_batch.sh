#!/bin/bash

echo seed ${1}
for SKILL_DIM in 5 10 20 50 100 200 500 1000
    do for AGENT in smm
        do for SNAPSHOT_TS in 10000 50000 100000 500000 1000000 2000000 3000000 4000000 5000000 6000000
            do python prioritized_sampling.py seed=${1} skill_dim=$SKILL_DIM agent=$AGENT obs_type=states task=SimplePointBot_goal data_type=unsupervised snapshot_ts=$SNAPSHOT_TS
        done
    done
done