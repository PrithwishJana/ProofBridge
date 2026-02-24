import json
from datasets import load_dataset
import re
import pandas as pd
from tqdm import tqdm

# Function to convert problem column to lowercase and remove spaces
def normalize_problem(problem: str) -> str:
    return problem.strip().replace(" ", "").lower()

def extractCommentsFromLEANcode(lean_code):
    # regex for line comments `-- ...`
    line_comments = re.findall(r'--.*', lean_code)
    # regex for block comments `/-- ... -/` (multiline, Lean doc-comments style)
    block_comments = re.findall(r'/--(.*?)-/', lean_code, flags=re.DOTALL)
    # Clean up whitespace in block comments
    block_comments = [c.strip() for c in block_comments]
    all_comments = line_comments + block_comments
    return "\n".join(all_comments)

def join_rawNLDescription(ds, problem_to_solution):
    #--------------store the dataset as a jsonl--------------
    newDataset = []
    for row in tqdm(ds["train"], desc="Processing dataset"):
        key = normalize_problem(row["problem"])
        if key in problem_to_solution:
            informal_soln_description = problem_to_solution[key]
            if (informal_soln_description is not None) and (len(informal_soln_description.strip()) > 0):
                has_informal_soln_description = "yes"
            else:
                has_informal_soln_description = "no"
        else:
            has_informal_soln_description = "no"
            informal_soln_description = "" 
        if row["ground_truth_type"] in ["complete", "with_sorry"]:
            #dataset mentions formal_ground_truth available only when author == human. For ground_truth_type:
            #complete: this means the formal proof is complete, 
            #with_sorry: this means the formal proof is written with some sorry in certain have statements.
            #statement: only the statement has been formalised. Proof is filled only with a sorry
            formal_proof = row["formal_ground_truth"]
            has_fl_proof = "yes"
        elif (row.get("formal_proof") or "").strip() != "":
            #dataset mentions formal_proof available only if the model manage to prove it, otherwise it is empty.
            formal_proof = row["formal_proof"]
            has_fl_proof = "yes"
        else:
            formal_proof = ""
            has_fl_proof = "no"
        new_data = {
            "uuid": row["uuid"],
            "question_type": row["question_type"],
            "has_fl_proof?": has_fl_proof,
            "fl_proof_compiles?": "",
            "has_nl_proof?": "no",
            "has_soln_description?": has_informal_soln_description,
            "soln_description": informal_soln_description,
            "informal_statement": row["problem"],
            "informal_proof": "",
            "formal_statement": row["formal_statement"],
            "formal_proof": formal_proof,
        }
        newDataset.append(new_data)
    return newDataset

if __name__ == "__main__":
    # Get the "AI-MO/NuminaMath-LEAN" dataset
    dataset_superset = load_dataset("AI-MO/NuminaMath-1.5")
    train_data = dataset_superset["train"]
    problem_to_solution = {
        normalize_problem(p): s
        for p, s, v in zip(train_data["problem"], train_data["solution"], train_data["solution_is_valid"])
        if str(v).lower() == "yes"
    }

    # Get the "AI-MO/NuminaMath-LEAN" dataset
    ds = load_dataset("AI-MO/NuminaMath-LEAN")

    newDataset = join_rawNLDescription(ds, problem_to_solution)
    df = pd.DataFrame(newDataset)
    output_file = "./datasets_training/NuminaMath-LEAN/dataset_step1.csv" # new csv file
    # save to CSV
    df.to_csv(output_file, index=False)
    print(f"Processed file written to {output_file}")

    # Count combinations of has_fl_proof? and has_nl_proof?
    counts = df.groupby(["has_fl_proof?", "has_nl_proof?"]).size().reset_index(name="count")
    print(counts)
