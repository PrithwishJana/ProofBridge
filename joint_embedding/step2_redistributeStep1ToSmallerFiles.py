import glob
import os
import pandas as pd
from pathlib import Path

FILE_PATTERN = "./joint_embedding/step1REPL_Parts/*.csv"

def restructureToSmallerFiles():
    global FILE_PATTERN

    # Read all matching files
    all_files = glob.glob(FILE_PATTERN)
    print("Read", len(all_files), "files!")

    if not all_files:
        print("No files found. Exiting.")
        return

    # Load and concat
    df_list = [pd.read_csv(f, dtype=str) for f in all_files]
    final_df = pd.concat(df_list, ignore_index=True)
    print("Total rows:", len(final_df))

    # Output directory
    output_dir = Path("./joint_embedding/step2REPL_Parts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split into chunks of 400 rows
    chunk_size = 300
    num_parts = (len(final_df) + chunk_size - 1) // chunk_size

    for i in range(num_parts):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(final_df))
        df_chunk = final_df.iloc[start_idx:end_idx]

        out_csv_path = output_dir / f"step2REPL_Part-{i*1000:06d}.csv"
        df_chunk.to_csv(out_csv_path, index=False)
        print(f"Saved {len(df_chunk)} rows to {out_csv_path}")

if __name__ == "__main__":
    restructureToSmallerFiles()