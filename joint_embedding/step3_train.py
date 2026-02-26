# train_nl_proof_embeddings.py
from __future__ import annotations

import argparse
import gc
import json
import ast
import os
import glob
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForTextEncoding,
    AutoTokenizer,
)
import wandb


def get_gpu_memory_info():
    """Get current GPU memory usage information."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
    return "GPU not available"


def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_event(data: str | dict[str, Any]) -> dict | None:
    """Parse a proof event (JSON or Python dict string) into a Python dict."""
    if isinstance(data, dict):
        obj = data
    else:
        if "[JSON parsed correctly]" in data:
            cleaned = data.split("]", 1)[-1].strip()
            #print ("bb", cleaned)
            obj = ast.literal_eval(cleaned)
            #print ("cc", obj)

    # Validate the parsed object
    if isinstance(obj, dict) and "tactics" in obj and isinstance(obj["tactics"], list):
        #print ("ee")
        return obj
    #print ("ff")
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
        max_seq_len: int = 1000,
        max_samples: Optional[int] = None,
        concat_states: bool = True,  # Whether to concatenate states into one string
    ):
        super().__init__()

        # Load CSV
        self.df = load_csv_or_folder(csv_path)

        # Construct modality_NL for each row
        self.df["modality_NL_raw"] = self.df.apply(
            lambda row: (
                f"<informal_statement>\n{row['informal_statement']}\n</informal_statement>\n\n"
                f"<informal_proof>\n{row['informal_proof']}\n</informal_proof>"
            ),
            axis=1
        )
        # Construct modality_FL using parse_event
        self.df["modality_FL_raw"] = self.df["repl_formal_proof"].apply(parse_event)
        print ("xx", self.df["modality_FL_raw"][:2])

        if max_samples:
            self.df = self.df.head(max_samples).reset_index(drop=True)
        self.max_seq_len = max_seq_len
        self.concat_states = concat_states

        # Process samples
        self.samples = self._process_samples()
        print(f"Loaded {len(self.samples)} NL-FL proof pairs")

    def _process_samples(self) -> List[Dict]:
        samples = []

        for idx, row in self.df.iterrows():
            # Get NL proof
            NL_item = str(row["modality_NL_raw"])

            # Get FL proof
            FL_dict = row["modality_FL_raw"]
            if not FL_dict or "tactics" not in FL_dict:
                #print ("modality_FL_raw", row["modality_FL_raw"])
                continue
            # Extract FL proof states
            states = []
            for t in FL_dict["tactics"][: self.max_seq_len]:
                if "goals" in t and t["goals"]:
                    states.append(t["goals"])

            if not states:
                #print ("t", t)
                continue

            # Process proof states
            if self.concat_states:
                # Concatenate all states with separator
                FL_item = " [SEP] ".join(states)
            else:
                FL_item = states  # Keep as list

            #print ("Appended!")
            samples.append(
                {
                    "modality_NL": NL_item,
                    "modality_FL": FL_item,
                    "uuid": row["uuid"],
                    "modality_FL_numStates": len(states),
                }
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


@dataclass
class ContrastiveBatch:
    """Batch for contrastive learning with proper positive/negative pairs."""

    nl_ids: torch.Tensor  # (B, L_nl)
    nl_mask: torch.Tensor  # (B, L_nl)
    proof_ids: torch.Tensor  # (B, L_proof)
    proof_mask: torch.Tensor  # (B, L_proof)
    proof_lens: list[int]
    # In-batch negatives are implicit - each NL matches with its corresponding proof


class ContrastiveCollator:
    """Collator for NL-proof pairs for contrastive learning."""

    def __init__(
        self,
        nl_tokenizer,
        proof_tokenizer,
        nl_max_len: int = 128,
        proof_max_len: int = 512,
    ):
        self.nl_tokenizer = nl_tokenizer
        self.proof_tokenizer = proof_tokenizer
        self.nl_max_len = nl_max_len
        self.proof_max_len = proof_max_len

    def __call__(self, examples: List[Dict]) -> ContrastiveBatch:
        # Extract NL and proof texts
        nl_texts = [ex["modality_NL"] for ex in examples]
        all_proof_states = []
        proof_lens = [len(ex["modality_FL"]) for ex in examples]
        for ex in examples:
            all_proof_states.extend(ex["modality_FL"])

        # Tokenize NL statements
        nl_batch = self.nl_tokenizer(
            nl_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.nl_max_len,
        )

        # Tokenize proof sequences
        proof_states_tokenized = self.proof_tokenizer(
            all_proof_states,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.proof_max_len,
        )

        return ContrastiveBatch(
            nl_ids=nl_batch["input_ids"],
            nl_mask=nl_batch["attention_mask"],
            proof_ids=proof_states_tokenized["input_ids"],
            proof_mask=proof_states_tokenized["attention_mask"],
            proof_lens=proof_lens,
        )


# ----------------------------- Model Components ------------------------------


def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean pooling over sequence."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class ProjectionHead(nn.Module):
    """Projection head for mapping to shared embedding space."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualEncoder(nn.Module):
    """Dual encoder for NL statements and proof sequences.

    Args:
        nl_model_name: Pretrained model for NL encoding
        proof_model_name: Pretrained model for proof encoding
        projection_dim: Dimension of shared embedding space
        freeze_nl: Freeze NL encoder weights
        freeze_proof: Freeze proof encoder weights
    """

    def __init__(
        self,
        nl_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        proof_model_name: str = "kaiyuy/leandojo-lean4-retriever-byt5-small",
        projection_dim: int = 256,
        freeze_nl: bool = True,
        freeze_proof: bool = True,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
        nl_trainable_layers: Optional[int] = None,
        proof_trainable_layers: Optional[int] = None,
    ):
        super().__init__()

        # Load pretrained encoders
        self.nl_encoder = AutoModel.from_pretrained(nl_model_name)
        self.proof_encoder = AutoModelForTextEncoding.from_pretrained(proof_model_name)

        # Get hidden dimensions
        nl_hidden = self.nl_encoder.config.hidden_size
        proof_hidden = self.proof_encoder.config.hidden_size

        # Projection heads to shared space
        self.nl_proj = ProjectionHead(nl_hidden, projection_dim, dropout)
        self.proof_proj = ProjectionHead(proof_hidden, projection_dim, dropout)

        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            if hasattr(self.nl_encoder, "gradient_checkpointing_enable"):
                self.nl_encoder.gradient_checkpointing_enable()
            elif hasattr(self.nl_encoder, "enable_input_require_grads"):
                self.nl_encoder.enable_input_require_grads()

            if hasattr(self.proof_encoder, "gradient_checkpointing_enable"):
                self.proof_encoder.gradient_checkpointing_enable()
            elif hasattr(self.proof_encoder, "enable_input_require_grads"):
                self.proof_encoder.enable_input_require_grads()

        # Optionally freeze encoders or train only last N layers
        self._freeze_encoder_layers(
            self.nl_encoder, freeze_nl, nl_trainable_layers, "modality_NL"
        )
        self._freeze_encoder_layers(
            self.proof_encoder, freeze_proof, proof_trainable_layers, "modality_FL"
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing

        print(f"NL encoder: {nl_model_name} (hidden={nl_hidden})")
        print(f"Proof encoder: {proof_model_name} (hidden={proof_hidden})")
        print(f"Projection dim: {projection_dim}")
        print(f"Gradient checkpointing: {use_gradient_checkpointing}")

    def _freeze_encoder_layers(
        self,
        encoder,
        freeze_all: bool,
        trainable_layers: Optional[int],
        encoder_name: str,
    ):
        """Freeze encoder layers based on configuration.

        Args:
            encoder: The encoder model to freeze
            freeze_all: Whether to freeze all layers
            trainable_layers: Number of last layers to keep trainable (overrides freeze_all)
            encoder_name: Name for logging purposes
        """
        if trainable_layers is not None:
            # Train only the last N layers
            total_layers = self._get_encoder_layer_count(encoder)
            freeze_up_to = total_layers - trainable_layers

            print(
                f"{encoder_name} encoder: Training last {trainable_layers}/{total_layers} layers"
            )

            # Get the transformer layers
            layers = self._get_encoder_layers(encoder)

            # Freeze embedding layers and early transformer layers
            for name, param in encoder.named_parameters():
                if any(
                    layer_component in name
                    for layer_component in ["embeddings", "embed"]
                ):
                    param.requires_grad = False

            # Freeze/unfreeze transformer layers
            for i, layer in enumerate(layers):
                for param in layer.parameters():
                    param.requires_grad = i >= freeze_up_to

        elif freeze_all:
            # Freeze all encoder parameters
            print(f"{encoder_name} encoder: All layers frozen")
            for param in encoder.parameters():
                param.requires_grad = False
        else:
            # Keep all layers trainable
            print(f"{encoder_name} encoder: All layers trainable")

    def _get_encoder_layer_count(self, encoder) -> int:
        """Get the number of transformer layers in the encoder."""
        if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
            return len(encoder.encoder.layer)
        elif hasattr(encoder, "transformer") and hasattr(encoder.transformer, "layers"):
            return len(encoder.transformer.layers)
        elif hasattr(encoder, "layers"):
            return len(encoder.layers)
        else:
            # Fallback: count layers by name
            layer_names = set()
            for name, _ in encoder.named_parameters():
                if "layer." in name or "layers." in name:
                    parts = name.split(".")
                    for i, part in enumerate(parts):
                        if part in ["layer", "layers"] and i + 1 < len(parts):
                            if parts[i + 1].isdigit():
                                layer_names.add(int(parts[i + 1]))
            return len(layer_names) if layer_names else 12  # Default fallback

    def _get_encoder_layers(self, encoder):
        """Get the transformer layers from the encoder."""
        if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
            return encoder.encoder.layer
        elif hasattr(encoder, "transformer") and hasattr(encoder.transformer, "layers"):
            return encoder.transformer.layers
        elif hasattr(encoder, "layers"):
            return encoder.layers
        else:
            # Fallback: return empty list
            print("Warning: Could not find transformer layers in encoder")
            return []

    def encode_nl(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode natural language statements."""
        outputs = self.nl_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Pool the outputs
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = mean_pool(outputs.last_hidden_state, attention_mask)

        # Project to shared space
        projected = self.nl_proj(pooled)

        # L2 normalize
        return F.normalize(projected, p=2, dim=-1)

    def encode_proof(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode proof sequences."""
        outputs = self.proof_encoder(input_ids, attention_mask=attention_mask)

        # Pool the outputs
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = mean_pool(outputs.last_hidden_state, attention_mask)

        # Project to shared space
        projected = self.proof_proj(pooled)

        # L2 normalize
        return F.normalize(projected, p=2, dim=-1)

    def forward(self, batch: ContrastiveBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both NL and proof embeddings."""
        nl_emb = self.encode_nl(batch.nl_ids, batch.nl_mask)
        proof_emb = self.encode_proof(batch.proof_ids, batch.proof_mask)
        proof_lens = batch.proof_lens

        i = 0
        proof_emb_list = []
        for _len in proof_lens:
            proof_emb_list.append(
                torch.mean(proof_emb[i : i + _len], dim=0, keepdim=True)
            )
            i += _len
        proof_emb = torch.concat(proof_emb_list, dim=0)
        return nl_emb, proof_emb


class EmbeddingQueue:
    """Queue to store embeddings from previous batches for cross-batch negatives."""

    def __init__(self, max_size: int, embedding_dim: int, device: torch.device):
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        self.device = device

        # Initialize empty queues
        self.nl_queue = torch.zeros(
            0, embedding_dim, device=device, dtype=torch.float32
        )
        self.proof_queue = torch.zeros(
            0, embedding_dim, device=device, dtype=torch.float32
        )

    def add_embeddings(self, nl_emb: torch.Tensor, proof_emb: torch.Tensor):
        """Add new embeddings to the queues."""
        with torch.no_grad():
            # Detach embeddings to avoid gradient flow
            nl_emb = nl_emb.detach()
            proof_emb = proof_emb.detach()

            # Add to queues
            self.nl_queue = torch.cat([self.nl_queue, nl_emb], dim=0)
            self.proof_queue = torch.cat([self.proof_queue, proof_emb], dim=0)

            # Maintain max size (FIFO)
            if len(self.nl_queue) > self.max_size:
                excess = len(self.nl_queue) - self.max_size
                self.nl_queue = self.nl_queue[excess:]
                self.proof_queue = self.proof_queue[excess:]

    def get_all_negatives(self):
        """Get all stored embeddings for use as negatives."""
        return self.nl_queue, self.proof_queue

    def size(self):
        """Get current queue size."""
        return len(self.nl_queue)


class InfoNCELoss(nn.Module):
    """InfoNCE loss with in-batch negatives.

    For each NL-proof pair in the batch:
    - Positive: The paired proof
    - Negatives: All other proofs in the batch
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, nl_emb: torch.Tensor, proof_emb: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            nl_emb: (B, D) normalized NL embeddings
            proof_emb: (B, D) normalized proof embeddings
        """
        batch_size = nl_emb.shape[0]

        # Compute similarity matrix (B, B)
        # Each row i represents similarities between nl_i and all proofs
        similarity = torch.matmul(nl_emb, proof_emb.T) / self.temperature

        # Labels: diagonal elements are the positive pairs
        labels = torch.arange(batch_size, device=nl_emb.device)

        # Compute cross-entropy loss
        loss = self.cross_entropy(similarity, labels)

        return loss


class InfoNCEWithQueueLoss(nn.Module):
    """InfoNCE loss with cross-batch negatives from embedding queue."""

    def __init__(
        self,
        temperature: float = 0.07,
        queue_size: int = 4096,
        embedding_dim: int = 256,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
        self.queue_size = queue_size
        self.embedding_queue: EmbeddingQueue = EmbeddingQueue(
            max_size=self.queue_size, embedding_dim=embedding_dim, device=device
        )

    def forward(self, nl_emb: torch.Tensor, proof_emb: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss with cross-batch negatives.

        Args:
            nl_emb: (B, D) normalized NL embeddings
            proof_emb: (B, D) normalized proof embeddings
        """
        batch_size = nl_emb.shape[0]
        device = nl_emb.device

        # Get stored negatives from queue
        queue_nl, queue_proof = self.embedding_queue.get_all_negatives()

        if len(queue_proof) > 0:
            # Combine current batch with queue negatives
            all_proof_emb = torch.cat([proof_emb, queue_proof], dim=0)

            # Compute similarities: (B, B + queue_size)
            similarity = torch.matmul(nl_emb, all_proof_emb.T) / self.temperature

            # Labels remain the same - positives are still at diagonal positions
            labels = torch.arange(batch_size, device=device)
        else:
            # Fall back to in-batch negatives only
            similarity = torch.matmul(nl_emb, proof_emb.T) / self.temperature
            labels = torch.arange(batch_size, device=device)

        # Add current batch to queue for future use
        self.embedding_queue.add_embeddings(nl_emb, proof_emb)

        # Compute loss
        loss = self.cross_entropy(similarity, labels)
        return loss


class SymmetricInfoNCELoss(InfoNCELoss):
    """Symmetric InfoNCE loss (both NL->proof and proof->NL)."""

    def forward(self, nl_emb: torch.Tensor, proof_emb: torch.Tensor) -> torch.Tensor:
        # NL -> Proof loss
        loss_nl_to_proof = super().forward(nl_emb, proof_emb)

        # Proof -> NL loss
        loss_proof_to_nl = super().forward(proof_emb, nl_emb)

        # Average both directions
        return (loss_nl_to_proof + loss_proof_to_nl) / 2


class SymmetricInfoNCEWithQueueLoss(InfoNCEWithQueueLoss):
    """Symmetric InfoNCE loss with cross-batch negatives (both NL->proof and proof->NL)."""

    def forward(self, nl_emb: torch.Tensor, proof_emb: torch.Tensor) -> torch.Tensor:
        batch_size = nl_emb.shape[0]
        device = nl_emb.device

        # Get stored negatives from queue
        queue_nl, queue_proof = self.embedding_queue.get_all_negatives()

        if len(queue_proof) > 0:
            # NL -> Proof direction with queue negatives
            all_proof_emb = torch.cat([proof_emb, queue_proof], dim=0)
            similarity_nl_to_proof = (
                torch.matmul(nl_emb, all_proof_emb.T) / self.temperature
            )

            # Proof -> NL direction with queue negatives
            all_nl_emb = torch.cat([nl_emb, queue_nl], dim=0)
            similarity_proof_to_nl = (
                torch.matmul(proof_emb, all_nl_emb.T) / self.temperature
            )

            labels = torch.arange(batch_size, device=device)
        else:
            # Fall back to in-batch negatives only
            similarity_nl_to_proof = (
                torch.matmul(nl_emb, proof_emb.T) / self.temperature
            )
            similarity_proof_to_nl = (
                torch.matmul(proof_emb, nl_emb.T) / self.temperature
            )
            labels = torch.arange(batch_size, device=device)

        # Add current batch to queue for future use
        self.embedding_queue.add_embeddings(nl_emb, proof_emb)

        # Compute losses in both directions
        loss_nl_to_proof = self.cross_entropy(similarity_nl_to_proof, labels)
        loss_proof_to_nl = self.cross_entropy(similarity_proof_to_nl, labels)

        # Average both directions
        return (loss_nl_to_proof + loss_proof_to_nl) / 2


def train_epoch(
    model: DualEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_wandb: bool = False,
    epoch: int = 1,
) -> Dict[str, float]:
    """Train for one epoch with gradient accumulation support."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    accumulated_loss = 0.0

    if use_amp and device.type == "cuda":
        scaler = torch.GradScaler()
    else:
        scaler = None

    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        # Move to device
        batch = ContrastiveBatch(
            nl_ids=batch.nl_ids.to(device),
            nl_mask=batch.nl_mask.to(device),
            proof_ids=batch.proof_ids.to(device),
            proof_mask=batch.proof_mask.to(device),
            proof_lens=batch.proof_lens,
        )

        with torch.autocast(device_type="cuda", enabled=use_amp):
            # Get embeddings
            nl_emb, proof_emb = model(batch)

            # Compute loss with in-batch negatives
            loss = loss_fn(nl_emb, proof_emb)
            assert isinstance(loss, torch.Tensor)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass
        if use_amp and device.type == "cuda" and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulated_loss += loss.item()

        # Update weights every gradient_accumulation_steps
        if (step + 1) % gradient_accumulation_steps == 0:
            if use_amp and device.type == "cuda" and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
                optimizer.step()

            optimizer.zero_grad()

            # Update metrics (use accumulated loss for proper averaging)
            total_loss += accumulated_loss
            accumulated_loss = 0.0
            num_batches += 1

            # Memory cleanup every few steps to prevent buildup
            if (step + 1) % (gradient_accumulation_steps * 10) == 0:
                clear_memory()

            # Log metrics to wandb
            if use_wandb and (step + 1) % gradient_accumulation_steps == 0:
                current_loss = total_loss / num_batches if num_batches > 0 else 0.0
                wandb.log(
                    {
                        "train/loss": current_loss,
                        "train/step": (epoch - 1) * len(dataloader) + step + 1,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=(epoch - 1) * len(dataloader) + step + 1,
                )

            pbar.set_postfix(
                {
                    "loss": total_loss / num_batches if num_batches > 0 else 0.0,
                    "step": step + 1,
                    "acc_steps": gradient_accumulation_steps,
                }
            )

    # Handle remaining accumulated gradients
    if accumulated_loss > 0:
        if use_amp and device.type == "cuda" and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        optimizer.zero_grad()
        total_loss += accumulated_loss
        num_batches += 1

    return {"loss": total_loss / num_batches if num_batches > 0 else 0.0}


def calculate_mrr(ranks: torch.Tensor) -> float:
    """Calculate Mean Reciprocal Rank.

    MRR measures the average reciprocal rank of the first relevant result.
    Higher values indicate better performance.

    Args:
        ranks: Tensor of shape (N,) containing the rank of the first correct result
              for each query (1-indexed)

    Returns:
        Mean reciprocal rank as a float
    """
    reciprocal_ranks = 1.0 / ranks.float()
    return reciprocal_ranks.mean().item()


def calculate_map(similarities: torch.Tensor, k: int = 10) -> float:
    """Calculate Mean Average Precision.

    MAP evaluates the quality of ranked retrieval results by computing the mean
    of average precision scores across all queries.

    Args:
        similarities: Tensor of shape (N, N) where similarities[i,j] is the
                     similarity between query i and document j
        k: Maximum rank to consider for AP calculation

    Returns:
        Mean Average Precision as a float
    """
    n = similarities.size(0)
    ranked_indices = similarities.argsort(dim=1, descending=True)

    ap_scores = []
    for i in range(n):
        # For each query, the correct document is at index i
        correct_doc_rank = (ranked_indices[i] == i).nonzero(as_tuple=True)[0].item() + 1

        if correct_doc_rank <= k:
            # AP is 1/rank for single relevant document
            ap = 1.0 / correct_doc_rank
        else:
            ap = 0.0
        ap_scores.append(ap)

    return float(np.mean(ap_scores))


def calculate_ndcg_at_k(similarities: torch.Tensor, k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K.

    NDCG measures ranking quality with position-based discounting.
    Perfect ranking gets score 1.0.

    Args:
        similarities: Tensor of shape (N, N) where similarities[i,j] is the
                     similarity between query i and document j
        k: Rank cutoff for NDCG calculation

    Returns:
        NDCG@K score as a float
    """
    n = similarities.size(0)
    ranked_indices = similarities.argsort(dim=1, descending=True)

    ndcg_scores = []
    for i in range(n):
        # Find position of correct document (i) in ranked list
        correct_positions = (ranked_indices[i][:k] == i).nonzero(as_tuple=True)[0]

        if len(correct_positions) > 0:
            # Document found in top-k
            pos = correct_positions[0].item() + 1  # 1-indexed position
            dcg = 1.0 / np.log2(pos + 1)  # DCG for single relevant document
            idcg = 1.0 / np.log2(2)  # Ideal DCG (best possible)
            ndcg = dcg / idcg
        else:
            ndcg = 0.0

        ndcg_scores.append(ndcg)

    return float(np.mean(ndcg_scores))


def calculate_precision_at_k(similarities: torch.Tensor, k: int) -> float:
    """Calculate Precision@K.

    Precision@K measures the fraction of top-K retrieved documents that are relevant.

    Note:
        For our task with single relevant document per query, this is equivalent to Recall@K.

    Args:
        similarities: Tensor of shape (N, N) where similarities[i,j] is the
                     similarity between query i and document j
        k: Number of top documents to consider

    Returns:
        Precision@K as a float
    """
    n = similarities.size(0)
    ranked_indices = similarities.argsort(dim=1, descending=True)

    correct = (
        ranked_indices[:, :k]
        == torch.arange(n, device=similarities.device).unsqueeze(1)
    ).any(dim=1)
    return correct.float().mean().item()


def calculate_hit_rate_at_k(similarities: torch.Tensor, k: int) -> float:
    """Calculate Hit Rate@K.

    Hit Rate@K is the fraction of queries for which at least one relevant document
    appears in the top-K results.

    Note:
        For single relevant document per query, this is equivalent to Recall@K.

    Args:
        similarities: Tensor of shape (N, N) where similarities[i,j] is the
                     similarity between query i and document j
        k: Number of top documents to consider

    Returns:
        Hit Rate@K as a float
    """
    return calculate_precision_at_k(similarities, k)


def calculate_rank_statistics(similarities: torch.Tensor) -> tuple[float, float]:
    """Calculate average rank and median rank of correct documents.

    These metrics provide intuitive measures of retrieval performance.
    Lower ranks indicate better performance.

    Args:
        similarities: Tensor of shape (N, N) where similarities[i,j] is the
                     similarity between query i and document j

    Returns:
        Tuple of (average_rank, median_rank) as floats
    """
    n = similarities.size(0)
    ranked_indices = similarities.argsort(dim=1, descending=True)

    ranks = []
    for i in range(n):
        # Find rank of correct document (i) in sorted list
        rank = (ranked_indices[i] == i).nonzero(as_tuple=True)[
            0
        ].item() + 1  # 1-indexed
        ranks.append(rank)

    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    avg_rank = ranks_tensor.mean().item()
    median_rank = ranks_tensor.median().item()

    return avg_rank, median_rank


def calculate_similarity_statistics(
    nl_embs: torch.Tensor, proof_embs: torch.Tensor
) -> Dict[str, float]:
    """Calculate statistics of cosine similarity scores.

    Analyzes the distribution of similarity scores between positive pairs
    (matched NL-proof pairs) and negative pairs (mismatched pairs).

    Args:
        nl_embs: Tensor of shape (N, D) containing NL embeddings
        proof_embs: Tensor of shape (N, D) containing proof embeddings

    Returns:
        Dictionary containing similarity statistics
    """
    # Compute full similarity matrix
    similarities = torch.matmul(nl_embs, proof_embs.T)

    # Extract positive pairs (diagonal)
    positive_sims = torch.diag(similarities)

    # Extract negative pairs (off-diagonal)
    n = similarities.size(0)
    mask = torch.eye(n, dtype=torch.bool, device=similarities.device)
    negative_sims = similarities[~mask]

    return {
        "pos_sim_mean": positive_sims.mean().item(),
        "pos_sim_std": positive_sims.std().item(),
        "neg_sim_mean": negative_sims.mean().item(),
        "neg_sim_std": negative_sims.std().item(),
        "pos_sim_min": positive_sims.min().item(),
        "pos_sim_max": positive_sims.max().item(),
        "sim_gap": (positive_sims.mean() - negative_sims.mean()).item(),
    }


@torch.no_grad()
def evaluate(
    model: DualEncoder,
    dataloader: DataLoader,
    device: torch.device,
    ks: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Evaluate retrieval performance with comprehensive metrics.

    Computes bidirectional retrieval metrics including Recall@K, Precision@K,
    Hit Rate@K, NDCG@K, MRR, MAP, rank statistics, and similarity statistics.

    Args:
        model: The dual encoder model to evaluate
        dataloader: DataLoader containing evaluation data
        device: Device to run evaluation on
        ks: List of K values for top-K metrics

    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()

    all_nl_embs = []
    all_proof_embs = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = ContrastiveBatch(
            nl_ids=batch.nl_ids.to(device),
            nl_mask=batch.nl_mask.to(device),
            proof_ids=batch.proof_ids.to(device),
            proof_mask=batch.proof_mask.to(device),
            proof_lens=batch.proof_lens,
        )

        nl_emb, proof_emb = model(batch)

        all_nl_embs.append(nl_emb.cpu())
        all_proof_embs.append(proof_emb.cpu())

    # Concatenate all embeddings
    nl_embs = torch.cat(all_nl_embs, dim=0)
    proof_embs = torch.cat(all_proof_embs, dim=0)

    # Compute similarity matrices
    nl_to_proof_similarities = torch.matmul(nl_embs, proof_embs.T)
    proof_to_nl_similarities = nl_to_proof_similarities.T

    metrics = {}
    n = len(nl_embs)

    # NL -> Proof retrieval metrics
    nl_to_proof_ranks = nl_to_proof_similarities.argsort(dim=1, descending=True)

    # Calculate ranks of correct documents for MRR
    nl_to_proof_correct_ranks = []
    for i in range(n):
        rank = (nl_to_proof_ranks[i] == i).nonzero(as_tuple=True)[0].item() + 1
        nl_to_proof_correct_ranks.append(rank)
    nl_to_proof_correct_ranks = torch.tensor(nl_to_proof_correct_ranks)

    # Recall@K, Precision@K, Hit Rate@K, NDCG@K
    for k in ks:
        # Recall@K
        correct = (nl_to_proof_ranks[:, :k] == torch.arange(n).unsqueeze(1)).any(dim=1)
        recall = correct.float().mean().item()
        metrics[f"nl_to_proof_R@{k}"] = recall

        # NDCG@K
        metrics[f"nl_to_proof_NDCG@{k}"] = calculate_ndcg_at_k(
            nl_to_proof_similarities, k
        )

    # MRR
    metrics["nl_to_proof_MRR"] = calculate_mrr(nl_to_proof_correct_ranks)

    # MAP@10 (using largest k as default)
    map_k = max(ks) if ks else 10
    metrics[f"nl_to_proof_MAP@{map_k}"] = calculate_map(nl_to_proof_similarities, map_k)

    # Rank statistics
    avg_rank, median_rank = calculate_rank_statistics(nl_to_proof_similarities)
    metrics["nl_to_proof_avg_rank"] = avg_rank
    metrics["nl_to_proof_median_rank"] = median_rank

    # Proof -> NL retrieval metrics
    proof_to_nl_ranks = proof_to_nl_similarities.argsort(dim=1, descending=True)

    # Calculate ranks of correct documents for MRR
    proof_to_nl_correct_ranks = []
    for i in range(n):
        rank = (proof_to_nl_ranks[i] == i).nonzero(as_tuple=True)[0].item() + 1
        proof_to_nl_correct_ranks.append(rank)
    proof_to_nl_correct_ranks = torch.tensor(proof_to_nl_correct_ranks)

    # Recall@K, Precision@K, Hit Rate@K, NDCG@K
    for k in ks:
        # Recall@K
        correct = (proof_to_nl_ranks[:, :k] == torch.arange(n).unsqueeze(1)).any(dim=1)
        recall = correct.float().mean().item()
        metrics[f"proof_to_nl_R@{k}"] = recall

        # NDCG@K
        metrics[f"proof_to_nl_NDCG@{k}"] = calculate_ndcg_at_k(
            proof_to_nl_similarities, k
        )

    # MRR
    metrics["proof_to_nl_MRR"] = calculate_mrr(proof_to_nl_correct_ranks)

    # MAP@10
    metrics[f"proof_to_nl_MAP@{map_k}"] = calculate_map(proof_to_nl_similarities, map_k)

    # Rank statistics
    avg_rank, median_rank = calculate_rank_statistics(proof_to_nl_similarities)
    metrics["proof_to_nl_avg_rank"] = avg_rank
    metrics["proof_to_nl_median_rank"] = median_rank

    # Similarity statistics
    sim_stats = calculate_similarity_statistics(nl_embs, proof_embs)
    for key, value in sim_stats.items():
        metrics[f"similarity_{key}"] = value

    return metrics


def main():
    parser = argparse.ArgumentParser("NL-Proof Embedding Training")

    # Data arguments
    parser.add_argument(
        "--csv_path", type=str, required=True, 
        help="Path to CSV file or a folder containing multiple CSV files"
    )
    parser.add_argument(
        "--nl_column",
        type=str,
        default="LEAN Source",
        help="Column name for natural language statements",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=10,
        help="Maximum number of proof states to use",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Limit dataset size for debugging"
    )

    # Model arguments
    parser.add_argument(
        "--nl_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Pretrained model for NL encoding",
    )
    parser.add_argument(
        "--proof_model",
        type=str,
        default="kaiyuy/leandojo-lean4-retriever-byt5-small",
        help="Pretrained model for proof encoding",
    )
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=256,
        help="Dimension of shared embedding space",
    )
    parser.add_argument(
        "--freeze_nl", action="store_true", help="Freeze NL encoder weights"
    )
    parser.add_argument(
        "--freeze_proof", action="store_true", help="Freeze proof encoder weights"
    )
    parser.add_argument(
        "--nl_trainable_layers",
        type=int,
        default=None,
        help="Number of last layers to train in NL encoder (overrides --freeze_nl)",
    )
    parser.add_argument(
        "--proof_trainable_layers",
        type=int,
        default=None,
        help="Number of last layers to train in proof encoder (overrides --freeze_proof)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate in projection heads"
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay for AdamW"
    )
    parser.add_argument(
        "--nl_max_len", type=int, default=128, help="Max length for NL tokenization"
    )
    parser.add_argument(
        "--proof_max_len",
        type=int,
        default=512,
        help="Max length for proof tokenization",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.07, help="Temperature for InfoNCE loss"
    )
    parser.add_argument(
        "--symmetric_loss", action="store_true", help="Use symmetric InfoNCE loss"
    )
    parser.add_argument(
        "--use_amp", action="store_true", help="Use automatic mixed precision"
    )
    parser.add_argument(
        "--use_queue_loss",
        action="store_true",
        help="Use cross-batch negatives with embedding queue",
    )
    parser.add_argument(
        "--queue_size",
        type=int,
        default=4096,
        help="Size of embedding queue for cross-batch negatives",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )

    # Evaluation arguments
    parser.add_argument(
        "--val_split", type=float, default=0.05, help="Validation split ratio"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=1, help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--eval_ks",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="K values for recall@k",
    )

    # System arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )

    # Wandb arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="nl-proof-embeddings",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="Wandb run name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Wandb entity/team name"
    )
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb logging")

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
            reinit=True,
        )
        print(f"Initialized wandb project: {args.wandb_project}")
    else:
        print("Wandb logging disabled")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizers
    nl_tokenizer = AutoTokenizer.from_pretrained(args.nl_model)
    proof_tokenizer = AutoTokenizer.from_pretrained(args.proof_model)

    # Create model
    model = DualEncoder(
        nl_model_name=args.nl_model,
        proof_model_name=args.proof_model,
        projection_dim=args.projection_dim,
        freeze_nl=args.freeze_nl,
        freeze_proof=args.freeze_proof,
        dropout=args.dropout,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        nl_trainable_layers=args.nl_trainable_layers,
        proof_trainable_layers=args.proof_trainable_layers,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Log model info to wandb
    if args.use_wandb:
        wandb.log(
            {
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/nl_model": args.nl_model,
                "model/proof_model": args.proof_model,
                "model/projection_dim": args.projection_dim,
            },
            step=0,
        )

    # Create dataset
    dataset = NLFLDataset(
        csv_path=args.csv_path,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        concat_states=False,
    )

    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create dataloaders
    collator = ContrastiveCollator(
        nl_tokenizer=nl_tokenizer,
        proof_tokenizer=proof_tokenizer,
        nl_max_len=args.nl_max_len,
        proof_max_len=args.proof_max_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss function
    if args.use_queue_loss:
        if args.symmetric_loss:
            loss_fn = SymmetricInfoNCEWithQueueLoss(
                temperature=args.temperature,
                queue_size=args.queue_size,
                embedding_dim=args.projection_dim,
            )
        else:
            loss_fn = InfoNCEWithQueueLoss(
                temperature=args.temperature,
                queue_size=args.queue_size,
                embedding_dim=args.projection_dim,
            )
    else:
        if args.symmetric_loss:
            loss_fn = SymmetricInfoNCELoss(temperature=args.temperature)
        else:
            loss_fn = InfoNCELoss(temperature=args.temperature)

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_metric = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 50}")

        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            use_amp=args.use_amp,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            use_wandb=args.use_wandb,
            epoch=epoch,
        )

        scheduler.step()

        print(f"Train loss: {train_metrics['loss']:.4f}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")
        print(f"{get_gpu_memory_info()}")

        # Log epoch-level metrics to wandb
        if args.use_wandb:
            epoch_step = epoch * len(train_loader)
            wandb.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_metrics["loss"],
                    "train/learning_rate_epoch": scheduler.get_last_lr()[0],
                },
                step=epoch_step,
            )

        # Evaluate
        if epoch % args.eval_freq == 0:
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                ks=args.eval_ks,
            )

            print("\nValidation metrics:")
            for metric, value in val_metrics.items():
                print(f"  {metric}: {value:.4f}")

            # Log validation metrics to wandb
            if args.use_wandb:
                wandb_val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
                # Use the same step counter as training for consistency
                val_step = epoch * len(train_loader)
                print(
                    f"Logging {len(wandb_val_metrics)} validation metrics to wandb at step {val_step}"
                )
                wandb.log(wandb_val_metrics, step=val_step)

            # Save best model based on average recall
            avg_recall = np.mean([v for k, v in val_metrics.items() if "R@" in k])

            if avg_recall > best_metric:
                best_metric = avg_recall
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "metrics": val_metrics,
                    "args": vars(args),
                }

                save_path = os.path.join(args.save_dir, "best_model.pt")
                torch.save(checkpoint, save_path)
                print(f"\nSaved best model with avg recall: {best_metric:.4f}")

                # Log best metric to wandb
                if args.use_wandb:
                    val_step = epoch * len(train_loader)
                    wandb.log({"val/best_avg_recall": best_metric}, step=val_step)

    # Save final model
    final_checkpoint = {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
    }
    torch.save(final_checkpoint, os.path.join(args.save_dir, "final_model.pt"))

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best validation avg recall: {best_metric:.4f}")

    # Finish wandb logging
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
