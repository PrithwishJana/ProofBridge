import argparse
import gc
import json
import ast
import os
import glob
from tqdm import tqdm
tqdm.pandas()
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, SentenceTransformerTrainingArguments
from datasets import Dataset as dsDatasets

#-------READ DATASET-------
def parse_event(data: str | dict[str, Any]) -> dict | None:
    """Parse a proof event (JSON or Python dict string) into a Python dict."""
    if isinstance(data, dict):
        obj = data
    else:
        if "[JSON parsed correctly]" in data:
            cleaned = data.split("]", 1)[-1].strip()
            obj = ast.literal_eval(cleaned)

    # Validate the parsed object
    if isinstance(obj, dict) and "tactics" in obj and isinstance(obj["tactics"], list):
        return obj
    return None

def load_csv_or_folder(path):
    if os.path.isfile(path) and path.endswith(".csv"):
        # Case 1: Single CSV file
        print(f"Loading single CSV file: {path}")
        df = pd.read_csv(path, dtype=str)
    elif os.path.isdir(path):
        # Case 2: Folder containing CSV files
        file_pattern = os.path.join(path, "*.csv")
        all_files = glob.glob(file_pattern)
        print(f"Found {len(all_files)} CSV files in folder {path}")
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in folder: {path}")
        df_list = [pd.read_csv(f, dtype=str) for f in all_files]
        df = pd.concat(df_list, ignore_index=True)
    else:
        raise ValueError(f"Invalid path: {path} (not a CSV file or directory)")

    # Drop rows that have any empty or NaN cells
    before_rows = len(df)
    df = df.dropna()  # removes NaN
    df = df[~(df.eq('').any(axis=1))]  # removes empty strings

    print(f"Filtered out {before_rows - len(df)} rows with empty fields")
    print(f"Final DataFrame: {len(df)} rows. \nColumn list: {df.columns}")
    return df


class NLFLDataset(Dataset):
    """Dataset for proofs in two modalities (NL and FL).

    Each sample contains:
    - Natural Language (NL) proof
    - Formal Language (FL) proof i.e., sequential proof states (concatenated or as sequence)
    - UUID for tracking
    """

    def __init__(
        self,
        csv_path: str,
        samples: List[Dict] = [{}],
        max_samples = None
    ):
        super().__init__()

        # Load CSV
        self.df = load_csv_or_folder(csv_path)
        if max_samples is not None:
            self.df = self.df.head(max_samples)

        # Construct modality_NL for each row
        self.df["modality_NL_raw"] = self.df.progress_apply(
            lambda row: (
                f"<informal_statement>\n{row['informal_statement']}\n</informal_statement>\n\n"
                f"<informal_proof>\n{row['informal_proof']}\n</informal_proof>"
            ),
            axis=1
        )
        # Construct modality_FL using parse_event
        self.df["modality_FL_raw"] = self.df["repl_formal_proof"].progress_apply(parse_event)

        # Process samples
        self.samples = self._process_samples()
        print(f"Loaded {len(self.samples)} NL-FL proof pairs")

    def _process_samples(self) -> List[Dict]:
        samples = []

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            # Get NL proof
            NL_item = str(row["modality_NL_raw"])

            # Get FL proof
            FL_dict = row["modality_FL_raw"]
            if not FL_dict or "tactics" not in FL_dict:
                continue

            tactics = FL_dict["tactics"]
            trace_lines = []
            for t in tactics:
                proof_state = t.get("proofState")
                state = t.get("goals")
                tactic = t.get("tactic")
                trace_lines.append(f"STATE {proof_state}:\n{state}")
                trace_lines.append(f"TACTIC {proof_state}:\n{tactic}")
            FL_item = "\n\n".join(trace_lines)

            samples.append(
                {
                    "modality_NL": NL_item,
                    "modality_FL": FL_item,
                    "uuid": row["uuid"],
                }
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]
#--------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on NL-FL proof pairs")
    args = parser.parse_args()

    dataset = NLFLDataset("./joint_embedding/step2REPL_Parts") #, max_samples = 1000
    #print(f"Sample data: {dataset[0]}")

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    model.max_seq_length = 1300
    train_dataset = dsDatasets.from_dict({
        "anchor": [s["modality_NL"] for s in dataset.samples],
        "positive": [s["modality_FL"] for s in dataset.samples],
    })
    loss = losses.MultipleNegativesRankingLoss(model)

    # Define training arguments specifically to save VRAM
    training_args = SentenceTransformerTrainingArguments(
        per_device_train_batch_size=4,     # Smallest possible micro-batch
        gradient_accumulation_steps=4,    # Effectively a batch size of 16
        fp16=True,                         # Use Mixed Precision
        num_train_epochs=3,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        loss=loss,
        args=training_args,
    )
    #trainer.gradient_accumulation_steps = 8  # simulate larger batch
    #trainer.use_amp = True  # mixed precision
    trainer.train()