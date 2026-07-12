"""
data_split.py
-------------
Single source of truth for the train/val/test split.
Both train.py and evaluate.py import from here so the split
is guaranteed identical everywhere.
"""

import torch
from sklearn.model_selection import train_test_split


def get_splits(sequences, tabular, labels, seed=42):
    """
    Stratified split so the churn ratio is preserved in every set.
    Returns train, val, and test slices for all three tensors.
    """
    n = len(labels)
    indices = list(range(n))

    # First split: 80% train+val, 20% test — stratified on labels
    trainval_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels.tolist(), random_state=seed
    )

    # Second split: of the 80%, carve out 20% for validation
    trainval_labels = [labels[i].item() for i in trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.2, stratify=trainval_labels, random_state=seed
    )

    def pick(idx_list):
        idx = torch.tensor(idx_list)
        return sequences[idx], tabular[idx], labels[idx]

    return {
        "train": pick(train_idx),
        "val":   pick(val_idx),
        "test":  pick(test_idx),
    }