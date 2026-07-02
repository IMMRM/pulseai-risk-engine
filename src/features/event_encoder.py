"""
Encodes raw event type strings into integer token sequences.
"""
import json
import os
from pathlib import Path
from src.logger import get_logger

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

logger = get_logger(__name__)

def build_vocab(all_customers):
    logger.info("Building event vocabulary...")

    # Step 1 — collect all unique event types
    event_types = set()

    for customer in all_customers:
        for event in customer["events"]:
            event_types.add(event["event_type"])

    # Step 2 — sort them for a consistent, repeatable order
    sorted_types = sorted(event_types)

    # Step 3 — build the vocab dict, reserving 0 for padding
    vocab = {"<PAD>": 0}
    for index, event_type in enumerate(sorted_types, start=1):
        vocab[event_type] = index

    logger.info(f"Vocabulary built. {len(vocab)} entries (including <PAD>).")
    return vocab

def encode_sequence(events, vocab):
    """
    Convert one customer's list of event dicts into a list of integers
    using the vocabulary.

    Args:
        events: list of event dicts, each containing an "event_type" key.
        vocab:  dict mapping event_type strings to integers.

    Returns:
        A list of integers — one per event, in order.
        Unknown event types map to 0.
    """
    encoded = []
    for event in events:
        event_type = event["event_type"]
        token = vocab.get(event_type, 0)   # 0 if the type isn't in vocab
        encoded.append(token)
    return encoded

def pad_sequence(sequence, max_length):
    """
    Force a sequence to be exactly max_length long.

    - Longer sequences: keep the LAST max_length items (most recent events).
    - Shorter sequences: pad with 0s at the end.

    Args:
        sequence:   list of integers (from encode_sequence).
        max_length: the fixed length every sequence must become.

    Returns:
        A list of integers exactly max_length long.
    """
    if len(sequence) > max_length:
        # Too long — keep only the most recent events
        return sequence[-max_length:]

    # Too short (or exactly right) — pad with zeros at the end
    padding_needed = max_length - len(sequence)
    return sequence + [0] * padding_needed

def save_vocab(vocab, filename="vocab.json"):
    """
    Save the vocabulary dict to data/processed/ as JSON.
    """
    processed_dir = ROOT_DIR / "data" / "processed"
    os.makedirs(processed_dir, exist_ok=True)

    filepath = processed_dir / filename
    with open(filepath, "w") as f:
        json.dump(vocab, f, indent=2)

    logger.info(f"Vocabulary saved to: {filepath}  ({len(vocab)} entries)")
    return filepath


def load_vocab(filename="vocab.json"):
    """
    Load the vocabulary dict back from data/processed/.
    """
    filepath = ROOT_DIR / "data" / "processed" / filename
    with open(filepath, "r") as f:
        vocab = json.load(f)

    logger.info(f"Vocabulary loaded from: {filepath}  ({len(vocab)} entries)")
    return vocab

if __name__ == "__main__":
    import json
    with open("data/raw/customers_raw_2026-07-02_09-16-13.json") as f:
        all_customers = json.load(f)

    vocab = build_vocab(all_customers)
    save_vocab(vocab)

    reloaded = load_vocab()
    print("Match:", vocab == reloaded)   # should print: Match: True