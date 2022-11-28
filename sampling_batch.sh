#!/bin/bash

echo seed ${1}
for SKILL_DIM in 20
    do for AGENT in smm
        do for SNAPSHOT_TS in 7000000 8000000 9000000 10000000 11000000 12000000 13000000 14000000 15000000 16000000
            do python prioritized_sampling.py seed=${1} skill_dim=$SKILL_DIM agent=$AGENT obs_type=states task=SimplePointBot_goal data_type=unsupervised snapshot_ts=$SNAPSHOT_TS
        done
    done
done