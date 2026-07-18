from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
import pandas as pd
import glob, os
from datasets import Dataset

os.environ["WANDB_PROJECT"] = "jtEmb_AF"  # name your W&B project
os.environ["WANDB_API_KEY"] = "9f8010035104bdf821bbc6591ac73abe48b86165"

def load_training_dataset():
    # Pattern to match your files
    file_pattern = "./datasets_training/NuminaMath-LEAN/dataset_step3/dataset*.csv"

    # Read all matching files
    all_files = glob.glob(file_pattern)
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

    # Convert to HF dataset format
    hf_data = []
    for _, row in filtered_df.iterrows():
        input_NL_stmt = row['informal_statement']
        input_NL_pf = row['informal_proof']
        output_FL = row["formal_proof"]

        hf_data.append({
            "input_NL_stmt": input_NL_stmt,
            "input_NL_pf": input_NL_pf,
            "output_FL": output_FL,
        })

    print ("Size of training data:", len(hf_data))

    # Convert to a HuggingFace Dataset
    return Dataset.from_list(hf_data)

def preprocess_function(example):
    SYSTEM_PROMPT = '''
    You are an expert in formal mathematics and Lean 4 theorem proving (version v4.15.0).  
    Your task is to take an informal theorem in natural language as input and autoformalize it in Lean 4, including a proper theorem header.  
    Think step-by-step and ensure that the resulting formal theorem is compilable with Lean 4 (version 4.15.0).

    The input informal theorem in natural language will be provided as follows:
    - The informal statement enclosed in:
    <informal_statement>
    ...
    </informal_statement>

    - The informal proof enclosed in:
    <informal_proof>
    ...
    </informal_proof>

    You may also receive additional information. If present, these will be provided as follows:

    1. Semantically related formal theorems in Lean 4 (which may help you formalize the input theorem), retrieved via a highly intelligent semantic retrieval system from a large corpus of formal theorems, enclosed in:
    <related_formal_theorems>
    ...
    </related_formal_theorems>

    2. An existing attempt of formalization in Lean 4 that is erroneous, enclosed in:
    <formal_theorem_erroneous_attempt>
    ...
    </formal_theorem_erroneous_attempt>

    3. Lean compilation output for the erroneous formal theorem, enclosed in:
    <lean_compilation_error>
    ...
    </lean_compilation_error>   

    First, carefully think and reason step-by-step for the corresponding formal theorem, and output the complete formal theorem in Lean 4 with a header. 
    Importantly, **enclose the final formal theorem in Lean 4 (version 4.15.0) inside the following tags**:

    <formal_theorem>
    ```lean4
    (Provide the entire Lean 4 theorem with header here)
    ```
    </formal_theorem>
    '''
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<informal_statement>\n{example['input_NL_stmt']}\n</informal_statement>" + "\n\n" +\
                                        f"<informal_proof>\n{example['input_NL_pf']}\n</informal_proof>" + "\n\n" +\
                                        "<related_formal_theorems>\n</related_formal_theorems>" + "\n\n" +\
                                        "<formal_theorem_erroneous_attempt>\n</formal_theorem_erroneous_attempt>" + "\n\n" +\
                                        "<lean_compilation_error>\n</lean_compilation_error>"}
        ],
        "completion": [
            {"role": "assistant", "content":  "<formal_theorem>\n```lean4\n" + example['output_FL'] + "\n```\n</formal_theorem>"}
        ],
    }

if __name__ == "__main__":
    df_train = load_training_dataset()
    df_train_processed = df_train.map(preprocess_function, 
                                remove_columns=["input_NL_stmt", "input_NL_pf", "output_FL"]).select(range(200))
    print(df_train_processed)
    print("Number of examples:", len(df_train_processed))

    # Print a single example
    example = df_train_processed[0]
    print("\nPrompt:")
    for message in example["prompt"]:
        print(f"{message['role'].upper()}:\n{message['content']}\n")

    print("\nCompletion:")
    for message in example["completion"]:
        print(f"{message['role'].upper()}:\n{message['content']}\n")

    training_args = TrainingArguments(
        output_dir="./llm_SFT_miscFiles/",
        report_to="wandb",  # this tells the Trainer to log the metrics to W&B
        per_device_train_batch_size=8,
        bf16=True,
        gradient_accumulation_steps=4,
        num_train_epochs=20,
        # logging strategies 
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="epoch", # saving is done at the end of each epoch
    )

    trainer = SFTTrainer(
        model = "AI-MO/Kimina-Prover-RL-1.7B",
        train_dataset = df_train_processed,
        args=training_args
    )
    trainer.train()