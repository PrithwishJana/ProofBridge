import json
import re
import pandas as pd
from tqdm import tqdm
import sys, os
import argparse

SAVE_FREQ = 50

# Define the target path
target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../LEAN_interaction'))

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

def saveData(data_list, output_file):
    # Convert back to DataFrame
    df_out = pd.DataFrame(data_list)
    # Save to CSV
    df_out.to_csv(output_file, index=False)
    # Count combinations of has_fl_proof? and fl_proof_compiles?
    counts = df_out.groupby(["has_fl_proof?", "fl_proof_compiles?"]).size().reset_index(name="count")
    print(counts)

def check_LEANcompilability(list_of_dicts, output_file, strt_indx_padded):
    #run_dir, project_dir, lean_file_path = bootstrap_project()
    run_dir = Path("../autoformalization-jtemb/tmpFolder/8d9a729d-dec6-452d-bb90-d63be139ee52")
    project_dir = run_dir / "TmpProjDir"
    lean_file_path = project_dir / "TmpProjDir" / f"Basic_{strt_indx_padded}.lean"
    for rowIndx, row in enumerate(tqdm(list_of_dicts, desc="Checking Lean proofs")):
        if row["has_fl_proof?"] == "yes":
            if row["fl_proof_compiles?"] in ["yes", "no"]:
                print("Skipping", f"Row {rowIndx}")
                continue
            fl_proof = row["formal_proof"]
            lean_code = write_basic_lean("", fl_proof, lean_file_path)
            ok_repl, out_repl = check_repl(lean_file_path, project_dir)
            # Print with clear delimiters
            print("\n" + "="*50)
            print(f"Row {rowIndx}")
            #print("FL_PROOF:")
            #print(fl_proof)
            print("-"*50)
            print("OK_REPL:")
            print(ok_repl)
            print(out_repl)
            #print("-"*50)
            #print("OUT_REPL:")
            #print(out_repl)
            print("="*50 + "\n", flush = True)

            # Update the row
            row["fl_proof_compiles?"] = "yes" if ok_repl else "no"
            row["fl_proof_compilation_REPLerror"] = out_repl if (not ok_repl) else ""
        if rowIndx % SAVE_FREQ == 0:
            saveData(list_of_dicts, output_file)
    saveData(list_of_dicts, output_file)

if __name__ == "__main__":

    #------------------parse command line arguments------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strt_indx", 
        type=int, 
        default=0,
        help="in multiples of 1000"
    )
    args = parser.parse_args()
    strt_indx = args.strt_indx
    strt_indx_padded = f"{strt_indx:06d}"

    # Get the "AI-MO/NuminaMath-LEAN" dataset
    output_file = f"./datasets_training/NuminaMath-LEAN-PF/step2Parts/dataset_step2Part-{strt_indx_padded}.csv"
    if os.path.exists(output_file):
        input_file = output_file
    else:
        input_file = "./datasets_training/NuminaMath-LEAN-PF/dataset_step1_dwnldAndPreprocess.csv"
    # Read CSV with pandas
    df = pd.read_csv(input_file, dtype=str)
    df = df.fillna("")
    # Convert to list of dicts (records)
    data_list = df.to_dict(orient="records")[strt_indx : strt_indx + 1000]
    check_LEANcompilability(data_list, output_file, strt_indx_padded)
    print ("Done!")
