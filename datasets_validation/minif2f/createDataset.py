import json
from datasets import load_dataset
import re

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("AI-MO/minif2f_test")
oldDatasetPath = "./datasets_validation/minif2f_4-9/dataset.jsonl"   # your original jsonl file
output_file = "./datasets_validation/minif2f/dataset.jsonl" # new jsonl file

#--------------store the dataset as a dict--------------
dataset_dict = {}
with open(oldDatasetPath, "r", encoding="utf-8") as infile:
    for line in infile:
        if not line.strip():
            continue
        obj = json.loads(line)
        key = str(obj['name']).lower()
        dataset_dict[key] = obj

for k in dataset_dict:
    print (k)

with open(output_file, "w", encoding="utf-8") as outfile:
    for row in ds["train"]:
        name = str(row["name"]).lower()
        new_data = {
            "name": name,
            "id": dataset_dict[name]["id"],
            "split": dataset_dict[name]["split"],
            "informal_statement": row["informal_prefix"],
            "informal_proof": dataset_dict[name]["informal_proof"],
            "formal_statement": re.sub(r"/--.*?-/", "", row["formal_statement"], flags=re.DOTALL),
            "formal_proof": "sorry"
        }
        # Write back as jsonl
        outfile.write(json.dumps(new_data, ensure_ascii=True) + "\n")

print(f"Processed file written to {output_file}")
