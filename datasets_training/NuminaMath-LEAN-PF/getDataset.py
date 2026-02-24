import pandas as pd
import glob

# Pattern to match your files
file_pattern = "./datasets_training/NuminaMath-LEAN/step3Parts/*.csv"

# Read all matching files
all_files = glob.glob(file_pattern)
print ("Read", len(all_files), "files!")

# Load and concat
df_list = [pd.read_csv(f, dtype=str) for f in all_files]
#len_df_list = [len(d) for d in df_list]
#print (len_df_list)
final_df = pd.concat(df_list, ignore_index=True)

# Filter rows
filtered_df = final_df[
    (final_df["has_fl_proof?"] == "yes") &
    (final_df["fl_proof_compiles?"] == "yes") &
    (final_df["has_nl_proof?"] != "no")
]

pairs = [
    (
        f"<informal_statement>\n{row['informal_statement']}\n</informal_statement>\n\n"
        f"<informal_proof>\n{row['informal_proof']}\n</informal_proof>",
        row["formal_proof"]
    )
    for _, row in filtered_df.iterrows()
]

print ("Count of informal:formal pairs:", len(pairs))

# Print two examples with delimiters
for i, (informal, formal) in enumerate(pairs[:2], start=1):
    print(f"\n================ Example {i} ================\n")
    print("----- Informal Theorem+Proof Pair -----")
    print(informal)
    print("\n----- Formal Theorem+Proof Pair -----")
    print(formal)
    print("\n============================================\n")