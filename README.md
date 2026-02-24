# ProofBridge: Auto-Formalization of Natural Language Proofs in Lean via Joint Embeddings

Published at ICLR-2026 (14th International Conference on Learning Representations), 23-27 April 2026,  Rio de Janeiro, Brazil

Pre-print and supplementary available at: https://arxiv.org/abs/2306.06755

## Step 1: Load required modules (reqd. for ComputeCanada)
```bash
module load gcc arrow/15.0.1 opencv/4.11.0
```

## Step 2: Create & activate Python virtual environment

(do this only once; afterward you just source ~/lean_env/bin/activate)
```bash
cd ./LEAN_interaction
python -m venv ~/lean_env
source ~/lean_env/bin/activate
```

## Step 3: Install Python dependencies
```bash
pip install --upgrade pip
pip install pandas tqdm datasets lockfile rapidfuzz
pip install poetry
pip install vllm
pip install trl
pip install wandb
poetry lock
poetry install
```

## Step 4: Enter Poetry shell (if using poetry)
```bash
poetry shell
```

## Step 5: Install Lean via elan
```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
source $HOME/.elan/env
```


## Step 6: Set default Lean version:
```bash
elan default leanprover/lean4:4.15.0
lean --version  # should print Lean (version 4.15.0)
lake --version  # should work too and print version 4.15.0
```

## Step 7: To activate the venv:
```bash
module load gcc arrow/15.0.1
source ~/lean_env/bin/activate
```

## Read the Training Dataset:

```bash
python ./datasets_training/NuminaMath-LEAN-PF/getDataset.py
```

This'll create a data structure called `pairs`, that's a list of tuples (informal_theoremproof_pair, formal_theoremproof_pair). This code also prints the first two elements from `pairs` for convenience.

## Setup for running LEAN type-check:

NOTE: You need to do this only once.

```bash
./LEAN_interaction/setupFolders_beforeLEANtypecheck.sh
```

If you are on a slurm-based system, run the following commands to ease space constraint:

```bash
# Move the folder to your scratch directory
mv ../autoformalization-jtemb ~/scratch/
# Recreate a symlink at the old location pointing to the new one
ln -s ~/scratch/autoformalization-jtemb ../autoformalization-jtemb
```

After this, you can call the `check_repl(...)` function in `./LEAN_interaction/checkLEAN.py` for LEAN type-checking. See an example of how to use it, in `./datasets_training/NuminaMath-LEAN-PF/dataset_construction/createDataset_step2_leanTypeCheck.py`.

## Setting up interactive GPU shell (if in a slurm-based system)

```bash
srun --account=def-vganesh --gpus-per-node=h100:1 --cpus-per-task=8 --mem=128000 --time=0-01:15 --job-name=vllm --pty bash
```

## Citation
If you find the paper or this repository useful, please cite it with:

```bash
@inproceedings{jana2026proofbridge,
  title = {{ProofBridge: Auto-Formalization of Natural Language Proofs in Lean via Joint Embeddings}},
  author = {Jana, Prithwish and Kale, Kaan and Tanriverdi, Ahmet Ege and Song, Cruise and Vishwanath, Sriram and Ganesh, Vijay},
  booktitle = {Proceedings of the 14th International Conference on Learning Representations (ICLR)},
  year = {2026},
  note = {arXiv:2510.15681},
}
```
