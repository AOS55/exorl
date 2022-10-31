#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --partition gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=pretrain_spb
#SBATCH --mem=100G
#SBATCH --output=hpc_output/pretrain%j.log

echo ${1} starting
ln -f hpc_output/pretrain${SLURM_JOB_ID}.log hpc_output/pretrain${1}.log

source ./hpc_scripts/setup.sh
python pretrain.py agent=${1} domain=SimplePointBot obs_type=states

rm hpc_output/pretrain${SLURM_JOB_ID}.log