"""
pipeline.py
-----------
Orchestrates the full feature engineering pipeline end-to-end.

Steps:
  1. Load all customers (data_loader)
  2. Build + save the event vocabulary (event_encoder)
  3. Encode + pad each customer's event sequence
  4. Build the tabular feature matrix (tabular_features)
  5. Build the labels (churned / not)
  6. Convert everything to PyTorch tensors
  7. Save tensors to data/processed/
"""

import torch
import os
from pathlib import Path
from src.logger import get_logger

from src.data.data_loader import load_all_customers
from src.features.event_encoder import (
    build_vocab, encode_sequence, pad_sequence, save_vocab
)
from src.features.tabular_features import build_feature_matrix

logger = get_logger(__name__)

# Absolute path to project root
# __file__ is src/features/pipeline.py → three parents up = project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


# ── Build labels ───────────────────────────────────────────────────
def build_labels(all_customers):
    """
    Build the target label for each customer.

    Label logic:
        active customer   -> 0 (safe, did not churn)
        inactive customer -> 1 (churned)

    Returns:
        A list of integers (0 or 1), one per customer.
    """
    logger.info("Building labels from customer active status...")

    labels = []
    for customer in all_customers:
        is_active = customer["profile"]["is_active"]
        label = 0 if is_active else 1
        labels.append(label)

    churned = sum(labels)
    logger.info(
        f"Labels built. Churned: {churned}, Active: {len(labels) - churned}, "
        f"Total: {len(labels)}"
    )
    return labels


# ── Run the full pipeline ──────────────────────────────────────────
def run_pipeline(max_length=100):
    """
    Run the complete feature engineering pipeline and save tensors to disk.

    Args:
        max_length: fixed length every event sequence is padded/truncated to.
    """
    logger.info("=" * 55)
    logger.info("PulseAI — Feature Pipeline Started")
    logger.info("=" * 55)

    # ── Step 1 — Load all customers ──
    all_customers = load_all_customers()
    total = len(all_customers)

    # ── Step 2 — Build and save the vocabulary ──
    vocab = build_vocab(all_customers)
    save_vocab(vocab)

    # ── Step 3 — Encode + pad every customer's event sequence ──
    logger.info(f"Encoding and padding sequences (max_length={max_length})...")
    sequences = []
    for customer in all_customers:
        encoded = encode_sequence(customer["events"], vocab)
        padded = pad_sequence(encoded, max_length)
        sequences.append(padded)

    # ── Step 4 — Build the tabular feature matrix ──
    tabular_matrix, feature_names = build_feature_matrix(all_customers)

    # ── Step 5 — Build the labels ──
    labels = build_labels(all_customers)

    # ── Step 6 — Convert everything to PyTorch tensors ──
    logger.info("Converting to PyTorch tensors...")
    sequences_tensor = torch.tensor(sequences, dtype=torch.long)
    tabular_tensor   = torch.tensor(tabular_matrix, dtype=torch.float)
    labels_tensor    = torch.tensor(labels, dtype=torch.float)

    logger.info(f"  sequences tensor shape: {tuple(sequences_tensor.shape)}")
    logger.info(f"  tabular tensor shape:   {tuple(tabular_tensor.shape)}")
    logger.info(f"  labels tensor shape:    {tuple(labels_tensor.shape)}")

    # ── Step 7 — Save the tensors to data/processed/ ──
    processed_dir = ROOT_DIR / "data" / "processed"
    os.makedirs(processed_dir, exist_ok=True)

    torch.save(sequences_tensor, processed_dir / "sequences.pt")
    torch.save(tabular_tensor,   processed_dir / "tabular.pt")
    torch.save(labels_tensor,    processed_dir / "labels.pt")

    logger.info(f"Tensors saved to: {processed_dir}")
    logger.info("=" * 55)
    logger.info("PulseAI — Feature Pipeline Complete")
    logger.info("=" * 55)

    return sequences_tensor, tabular_tensor, labels_tensor, feature_names


if __name__ == "__main__":
    sequences, tabular, labels, feature_names = run_pipeline(max_length=100)

    print("\n── Pipeline Summary ──")
    print(f"Sequences : {tuple(sequences.shape)}  (customers x max_length)")
    print(f"Tabular   : {tuple(tabular.shape)}  (customers x features)")
    print(f"Labels    : {tuple(labels.shape)}  (customers)")
    print(f"Features  : {feature_names}")
    print(f"Churned   : {int(labels.sum().item())} / {len(labels)}")