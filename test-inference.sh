#!/bin/bash
#SBATCH --job-name=llm
#SBATCH --partition=mlgpu_devel
#SBATCH --time=1:00:00

#SBATCH --gpus=1

#SBATCH --account=ag_hiskp_funcke
#SBATCH --output=/home/s6jakrei_hpc/logs/%x-%j.out

source $HOME/.venv/bin/activate
python -u code/test-inference.py