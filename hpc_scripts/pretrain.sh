#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition gpu
#SBATCH --time=6:00:00
#SBATCH --job-name=pretrain_spb
#SBATCH --mem=20G
#SBATCH --output=hpc_output/pretrain%j.log

echo ${1} starting
echo agent ${2}
ln -f hpc_output/pretrain${SLURM_JOB_ID}.log hpc_output/pretrain${1}.log

source ./hpc_scripts/setup.sh
python pretrain.py agent=${2} skill_dim=${1} domain=SimplePointBot obs_type=states

rm hpc_output/pretrain${SLURM_JOB_ID}.log

