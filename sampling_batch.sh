#!/bin/bash

echo seed ${1}
for SKILL_DIM in 128 256 512
    do for AGENT in diayn
        do for SNAPSHOT_TS in 10000 50000 100000 500000 1000000
            do python prioritized_sampling.py seed=${1} skill_dim=$SKILL_DIM agent=$AGENT obs_type=states task=SimplePointBot_goal data_type=unsupervised snapshot_ts=$SNAPSHOT_TS
        done
    done
done