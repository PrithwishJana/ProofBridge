import json
import sys
import os, shutil
import argparse

# Add the directory containing checkLEAN.py to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../LEAN_interaction')))
from checkLEAN import *

# Path to dataset file
dataset_jsonl_file = "./datasets_validation/minif2f/dataset.jsonl"

#------------------parse command line arguments------------------
parser = argparse.ArgumentParser(description="Evaluate Lean4 outputs on a dataset")
parser.add_argument(
    "--output_jsonl", 
    type=str, 
    default="./datasets_validation/minif2f/goldStandard_output.jsonl",
    help="Path to the output JSONL file to evaluate"
)
args = parser.parse_args()
output_jsonl_file = args.output_jsonl

#--------------store the dataset as a dict--------------
dataset_dict = {}
with open(dataset_jsonl_file, "r", encoding="utf-8") as infile:
    for line in infile:
        if not line.strip():
            continue
        obj = json.loads(line)
        key = f"{obj['name']}_{obj['split']}"
        dataset_dict[key] = obj

#--------------read the output and evaluate--------------
total_count = 0
num_compile_ok = 0
run_dir, project_dir, lean_file_path = bootstrap_project()
with open(output_jsonl_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        
        # Extract required fields
        item = {
            "key": f"{obj['name']}_{obj['split']}",
            "header": obj.get("header", ""),
            "formal_statement": obj["formal_statement"],
            "formal_proof": obj["formal_proof"]
        }
        item["informal_statement_proof"] = dataset_dict[item["key"]]["informal_statement"] + "\n" + dataset_dict[item["key"]]["informal_proof"]
        total_count += 1

        print("="*80)
        print("Instance Name:", item["key"])
        print ("")
        print("Informal Statement+Proof (Input):", item["informal_statement_proof"])
        print ("")
        lean_code = write_basic_lean(item["header"], item["formal_statement"] + "\n" + item["formal_proof"], lean_file_path)
        ok_compile, out_compile = check_compiles(project_dir)
        num_compile_ok += (ok_compile == True)
        print("LEAN4 Statement+Proof (Output):", lean_code)
        print ("")
        print ("LEAN4 check outcome:", ok_compile)
        print ("LEAN4 check error:", out_compile)
        print("="*80 + "\n")
shutil.rmtree(run_dir)

# Print metrics
print(f"Total count       : {total_count}")
percentage_compile_ok = (num_compile_ok / total_count * 100) if total_count > 0 else 0
print(f"%age compile OK : {percentage_compile_ok:.2f}%")