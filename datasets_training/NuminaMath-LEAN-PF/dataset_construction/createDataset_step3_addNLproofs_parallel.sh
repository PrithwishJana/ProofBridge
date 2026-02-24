#!/bin/bash
#SBATCH --account=def-vganesh
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --job-name=lean_datasetStep3
#SBATCH --output=tmpFolder/lean_datasetStep3_%A_%a.out
#SBATCH --error=tmpFolder/lean_datasetStep3_%A_%a.err
#SBATCH --array=0-104

# Load required modules
module load gcc arrow/15.0.1 opencv/4.11.0

# Activate Lean virtual environment
source ~/lean_env/bin/activate

# Compute strt_indx for this task (in multiples of 1000)
strt_indx=$(( SLURM_ARRAY_TASK_ID * 1000 ))

# Pad to 6 digits for filenames
strt_indx_padded=$(printf "%06d" $strt_indx)

# Run Python script
python ./datasets_training/NuminaMath-LEAN/dataset_construction/createDataset_step3_addNLproofs.py \
       --strt_indx $strt_indx 2>&1 \
       | tee ./datasets_training/NuminaMath-LEAN/step3Parts/LOG_step3_addNLproofs_${strt_indx_padded}.txt
