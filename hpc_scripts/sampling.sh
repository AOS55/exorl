#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --partition gpu
#SBATCH --time=4:00:00
#SBATCH --job-name=sampling_spb
#SBATCH --mem=100G
#SBATCH --output=hpc_output/sampling%j.log

echo ${1} starting
ln -f hpc_output/sampling${SLURM_JOB_ID}.log

source ./hpc_scripts/setup.sh
python sampling.py agent=${1} task=SimplePointBot_goal obs_type=states

rm hpc_output/sampling${SLURM_JOB_ID}.log
