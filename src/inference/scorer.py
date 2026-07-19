import torch
import json
from pathlib import Path

from src.models.risk_model import RiskModel
from src.logger import get_logger
from src.features.event_encoder import encode_sequence, pad_sequence, load_vocab
from src.features.tabular_features import compute_features
logger = get_logger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def load_model():
    vocab_path= ROOT_DIR / "data"/"processed"/ "vocab.json"
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    model=RiskModel(vocab_size=vocab_size)
    model.load_state_dict(torch.load(ROOT_DIR / "checkpoints" / "best_model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model

def label_from_score(prob):
    if prob >= 0.8:
        return "Critical Risk"
    elif prob >= 0.6:
        return "High Risk"
    elif prob >= 0.3:
        return "Medium Risk"
    else:
        return "Low Risk"

def score_customer(customer_id, model, vocab, all_customers):
    # find the customer in the already-loaded list
    customer = None
    for c in all_customers:
        if c["customer_id"] == customer_id:
            customer = c
            break
    if customer is None:
        logger.warning(f"Customer {customer_id} not found.")
        return None

    # no more load_vocab() here — it's passed in
    encoded  = encode_sequence(customer["events"], vocab)
    padded   = pad_sequence(encoded, 100)
    features = compute_features(customer)

    seq_tensor = torch.tensor([padded], dtype=torch.long)
    tab_tensor = torch.tensor([features], dtype=torch.float)

    with torch.no_grad():
        logit = model(seq_tensor, tab_tensor)
        prob = torch.sigmoid(logit).item()

    label = label_from_score(prob)
    return {"customer_id": customer_id, "risk_score": round(prob, 4), "risk_label": label}

def score_all_customers(model):
    # load data + vocab ONCE
    raw_path = ROOT_DIR / "data" / "raw"
    latest_file = sorted(raw_path.glob("customers_raw_*.json"))[-1]
    with open(latest_file) as f:
        all_customers = json.load(f)
    vocab = load_vocab()

    results = []
    for c in all_customers:
        result = score_customer(c["customer_id"], model, vocab, all_customers)
        if result is not None:
            results.append(result)

    results.sort(key=lambda r: r["risk_score"], reverse=True)
    return results

if __name__ == "__main__":
    model = load_model()
    all_scores = score_all_customers(model)
    print(f"Scored {len(all_scores)} customers.")
    print("Top 5 highest risk:")
    for r in all_scores[:5]:
        print(f"  {r['customer_id']}: {r['risk_score']} ({r['risk_label']})")