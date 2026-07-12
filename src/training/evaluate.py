import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from src.models.risk_model import RiskModel
from src.logger import get_logger
from src.data.data_split import get_splits

logger = get_logger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def load_tensors():
    processed = ROOT_DIR / "data" / "processed"
    sequences = torch.load(processed / "sequences.pt")
    tabular   = torch.load(processed / "tabular.pt")
    labels    = torch.load(processed / "labels.pt")
    return sequences, tabular, labels


def evaluate():
    sequences, tabular, labels = load_tensors()
    vocab_size = int(sequences.max().item()) + 1

    # ── Recreate the same test split as training ──
    torch.manual_seed(42)
    n = len(labels)
    sequences, tabular, labels = load_tensors()
    splits = get_splits(sequences, tabular, labels)
    test_seq, test_tab, test_labels = splits["test"]

    model=RiskModel(vocab_size=vocab_size)
    model.load_state_dict(torch.load(ROOT_DIR / "checkpoints" / "best_model.pt"))
    model.eval()

    # ── Get predictions ──
    with torch.no_grad():
        logits = model(test_seq, test_tab)
        probs = torch.sigmoid(logits)          # ← convert logits to 0–1
        preds = (probs >= 0.5).float()

    # Convert tensors to plain numbers for sklearn
    y_true = test_labels.int().tolist()
    y_pred = preds.int().tolist()
    y_prob = probs.tolist()

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)          # uses probabilities, not 0/1!
    cm   = confusion_matrix(y_true, y_pred)
    logger.info("----Evaluation Results (Test Set)---")
    logger.info(f"Accuracy : {acc:.2%}")
    logger.info(f"Precision: {prec:.2%}")
    logger.info(f"Recall   : {rec:.2%}")
    logger.info(f"F1 Score : {f1:.2%}")
    logger.info(f"AUC-ROC  : {auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
if __name__ == "__main__":
    evaluate()