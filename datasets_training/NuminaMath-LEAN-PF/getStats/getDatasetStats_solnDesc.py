import pandas as pd
import glob

# Pattern to match your files
file_pattern = "./datasets_training/NuminaMath-LEAN/dataset_step3/dataset*.csv"

# Read all matching files
all_files = glob.glob(file_pattern)
print("Read", len(all_files), "files!")

# Load and concat
df_list = [pd.read_csv(f, dtype=str) for f in all_files]
final_df = pd.concat(df_list, ignore_index=True)

# Filter rows
filtered_df = final_df[
    (final_df["has_fl_proof?"] == "yes") &
    (final_df["fl_proof_compiles?"] == "yes") &
    (final_df["has_nl_proof?"] != "no") &
    (final_df["has_soln_description?"] != "no") &
    (final_df["soln_description"].str.len() > 0)
]

# Prepare pairs
pairs = [
    (
        f"<informal_statement>\n{row['informal_statement']}\n</informal_statement>\n\n"
        f"<informal_proof>\n{row['informal_proof']}\n</informal_proof>",
        row["formal_proof"],
        row["soln_description"]
    )
    for _, row in filtered_df.iterrows()
]

print("Count of informal:formal pairs:", len(pairs))

# Word counts for soln_description
word_lengths = filtered_df["soln_description"].str.split().str.len()

# Box plot stats
desc = word_lengths.describe(percentiles=[0.25, 0.5, 0.75])

print("Box plot values of soln_description word length:")
print(f"Min: {desc['min']}")
print(f"Q1:  {desc['25%']}")
print(f"Median: {desc['50%']}")
print(f"Q3:  {desc['75%']}")
print(f"Max: {desc['max']}")
