import argparse
import os
import glob
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from tqdm import tqdm

import pandas as pd

# Define the target path
target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../LEAN_interaction'))

# 🗂️ First, list what's in that directory
print("Contents of target directory before adding to sys.path:")
if os.path.exists(target_dir):
    for item in os.listdir(target_dir):
        print("  -", item)
else:
    print("Directory does not exist:", target_dir)

# ➕ Then append it to sys.path
sys.path.append(target_dir)
print("\nAdded to sys.path:", target_dir)

from checkLEAN import *

# ====================== CONFIG ======================
FILE_PATTERN = "./datasets_training/NuminaMath-LEAN/step3Parts/dataset*.csv"
SLICE_LENGTH = 1000
SAVE_FREQ = 5
# ====================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Batch Lean build/REPL harness")
    ap.add_argument("--start_indx", type=int, default=0, help=f"in multiples of {SLICE_LENGTH}")
    return ap.parse_args()

def get_df_nlfl_theoremproof():
    global FILE_PATTERN
    # Read all matching files
    all_files = glob.glob(FILE_PATTERN)
    print ("Read", len(all_files), "files!")
    # Load and concat
    df_list = [pd.read_csv(f, dtype=str) for f in all_files]
    final_df = pd.concat(df_list, ignore_index=True)
    # Filter rows
    filtered_df = final_df[
        (final_df["has_fl_proof?"] == "yes") &
        (final_df["fl_proof_compiles?"] == "yes") &
        (final_df["has_nl_proof?"] != "no")
    ]
    filtered_df_nlfl = filtered_df[["uuid", "informal_statement","informal_proof",
                                    "formal_statement","formal_proof"]]
    # Add empty column
    filtered_df_nlfl["repl_formal_proof"] = ""

    print ("Count of nlfl_theoremproof:", len(filtered_df_nlfl))
    return filtered_df_nlfl

def saveData(data_list, output_file):
    # Convert back to DataFrame
    df_out = pd.DataFrame(data_list)
    # Save to CSV
    df_out.to_csv(output_file, index=False)
    # Count how many repl_formal_proof are empty vs non-empty
    empty_count = (df_out["repl_formal_proof"] == "").sum()
    non_empty_count = (df_out["repl_formal_proof"] != "").sum()
    print(f"\nrepl_formal_proof empty: {empty_count}")
    print(f"repl_formal_proof non-empty: {non_empty_count}")

def check_LEANcompilability(list_of_dicts, output_file, strt_indx_padded):
    #run_dir, project_dir, lean_file_path = bootstrap_project()
    run_dir = Path("../autoformalization-jtemb/tmpFolder/8d9a729d-dec6-452d-bb90-d63be139ee52")
    project_dir = run_dir / "TmpProjDir"
    lean_file_path = project_dir / "TmpProjDir" / f"Basic_{strt_indx_padded}.lean"
    for rowIndx, row in enumerate(tqdm(list_of_dicts, desc="Checking Lean proofs")):
        fl_proof = row["formal_proof"]
        lean_code = write_basic_lean("", fl_proof, lean_file_path)
        ok_repl, out_repl = check_repl(lean_file_path, project_dir)
        # Print with clear delimiters
        print("\n" + "="*50)
        print(f"Row: {rowIndx}")
        print(f"OK_REPL: {ok_repl}")
        print(f"OUT_REPL: \n{out_repl}")
        print("="*50 + "\n", flush = True)
        # Update the row
        if ok_repl:
            row["repl_formal_proof"] = out_repl
        if rowIndx % SAVE_FREQ == 0:
            saveData(list_of_dicts, output_file)
    saveData(list_of_dicts, output_file)

def main():
    args = parse_args()
    strt_indx = int(args.start_indx)
    strt_indx_padded = f"{strt_indx:06d}"

    # Load and slice
    df_nlfl_theorempf = get_df_nlfl_theoremproof()
    slicedList_nlfl_theorempf = df_nlfl_theorempf.to_dict(orient="records")[strt_indx : strt_indx + SLICE_LENGTH]

    print(f"Taking slice [{strt_indx}:{strt_indx + SLICE_LENGTH})")

    # Output file for this batch
    out_csv_path = Path("./joint_embedding") / "step1REPL_Parts" / f"step1REPL_Part-{strt_indx_padded}.csv"
    # Ensure directory exists
    out_csv_path.parent.mkdir(parents = True, exist_ok = True)
    print (f"Will save .csv to {out_csv_path}")

    check_LEANcompilability(slicedList_nlfl_theorempf, out_csv_path, strt_indx_padded)
    print ("Done!")

if __name__ == "__main__":
    main()