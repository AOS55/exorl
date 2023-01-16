#!/bin/bash

export DISPLAY=:0
export HYDRA_FULL_ERROR=1

for seed in 1 2 3 4 5 do
    python pretrain.py agent=smm domain=SimpleVelocityBot seed=$seed &
done
