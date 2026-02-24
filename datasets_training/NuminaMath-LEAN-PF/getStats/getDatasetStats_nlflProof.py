import pandas as pd
import glob

# Pattern to match your files
file_pattern = "./datasets_training/NuminaMath-LEAN/step3Parts/*.csv"

# Read all matching files
all_files = glob.glob(file_pattern)
print("Read", len(all_files), "files!")

# Load and concat
df_list = [pd.read_csv(f, dtype=str) for f in all_files]
final_df = pd.concat(df_list, ignore_index=True)

# --- NEW SECTION: Counts for fl/nl proofs ---
count_has_fl = (final_df["has_fl_proof?"] == "yes").sum()
count_has_nl = final_df["has_nl_proof?"].str.startswith("yes", na=False).sum()
count_hasNOT_nl = final_df["has_nl_proof?"].str.startswith("no", na=False).sum()

print(f"Count of rows with formal (fl) proof: {count_has_fl}")
print(f"Count of rows with natural language (nl) proof: {count_has_nl}")
print(f"Count of rows NOT with natural language (nl) proof: {count_hasNOT_nl}")
