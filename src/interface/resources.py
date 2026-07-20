"""
Shared resources loaded once, imported by all tabs.
Prevents loading the model / data multiple times.
"""
import json
from pathlib import Path

from src.inference.scorer import load_model
from src.features.event_encoder import load_vocab

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

model = load_model()
vocab = load_vocab()

raw_path = ROOT_DIR / "data" / "raw"
latest_file = sorted(raw_path.glob("customers_raw_*.json"))[-1]
with open(latest_file) as f:
    all_customers = json.load(f)