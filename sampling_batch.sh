#!/bin/bash

echo seed ${1}
echo skill dim ${2}
for AGENT in smm
    do for SNAPSHOT_TS in 10000 50000 100000 500000 1000000 1500000 2000000
        do python prioritized_sampling.py seed=${1} skill_dim=${2} agent=$AGENT obs_type=states task=SimplePointBot_goal data_type=unsupervised snapshot_ts=$SNAPSHOT_TS
    done
done