#!/bin/bash

export DISPLAY=:0
export HYDRA_FULL_ERROR=1

for alpha in 0.001 0.01 0.1 1.0 10.0 do
    python train_sac.py domain=SimplePointBot alpha=$alpha &
done
