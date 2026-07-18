#!/bin/bash
#SBATCH --account=def-vganesh
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128000
#SBATCH --time=0-18:00
#SBATCH --job-name=vllm
#SBATCH --output=./llm_SFT_miscFiles/vllm_%j.out  # optional: save stdout/stderr

# Load any required modules
module load gcc arrow/15.0.1 opencv/4.11.0

# Activate your env
source ~/lean_env/bin/activate

python ./llm_SFT/train.py