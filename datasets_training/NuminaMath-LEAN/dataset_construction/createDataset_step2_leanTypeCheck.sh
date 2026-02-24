#!/bin/bash
#SBATCH --account=def-vganesh
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-16:00
#SBATCH --job-name=lean_dataset
#SBATCH --output=tmpFolder/lean_dataset_%A_%a.out
#SBATCH --error=tmpFolder/lean_dataset_%A_%a.err
#SBATCH --array=0-104

# Load required modules
module load gcc arrow/15.0.1

# Activate Lean virtual environment
source ~/lean_env/bin/activate

# Compute strt_indx for this task (in multiples of 1000)
strt_indx=$(( SLURM_ARRAY_TASK_ID * 1000 ))

# Pad to 6 digits for filenames
strt_indx_padded=$(printf "%06d" $strt_indx)

# Run Python script
python ./datasets_training/NuminaMath-LEAN/dataset_construction/createDataset_step2_leanTypeCheck.py \
       --strt_indx $strt_indx \
       | tee ./datasets_training/NuminaMath-LEAN/step2Parts/LOG_step2_leanTypeCheck_${strt_indx_padded}.txt
