"""
Unified loader: pulls from both DBs and joins on customer_id.
"""
from src.config import (
    SUPABASE_DB_HOST, SUPABASE_DB_PORT,
    SUPABASE_DB_NAME, SUPABASE_DB_USER, SUPABASE_DB_PASS,
    MONGO_URI, MONGO_DB_NAME
)
import psycopg2
import psycopg2.extras
from pymongo import MongoClient
from src.data.supabase_connector import get_supabase_connection
from src.data.mongo_connector import get_database as get_mongo_database
from src.logger import get_logger
import json
import os
from datetime import datetime
from pathlib import Path

logger = get_logger(__name__)

# Absolute path to project root
# __file__ is src/data/data_loader.py
# .parent       → src/data/
# .parent.parent → src/
# .parent.parent.parent → project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


# ── Load customers data ────────────────────────────────────────────
def load_customers():
    logger.info("Phase 1 — Connecting to Supabase to load customers...")
    conn = get_supabase_connection()
    if conn is None:
        logger.error("Phase 1 — Supabase connection failed. Returning empty result.")
        return {}

    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("SELECT * FROM customers;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    logger.info(f"Phase 1 — Customers loaded successfully. Total: {len(rows)}")
    return list(rows)


# ── Load customer's transactions ───────────────────────────────────
def load_transactions(customer_id):
    conn = get_supabase_connection()
    if conn is None:
        logger.error(f"  Supabase connection failed while loading transactions for {customer_id}.")
        return {}

    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("SELECT * FROM transactions WHERE customer_id=%s;", (customer_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return list(rows)


# ── Load subscriptions ─────────────────────────────────────────────
def load_subscriptions(customer_id):
    conn = get_supabase_connection()
    if conn is None:
        logger.error(f"  Supabase connection failed while loading subscriptions for {customer_id}.")
        return {}

    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("SELECT * FROM subscriptions WHERE customer_id=%s;", (customer_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return list(rows)


# ── Load customer events from MongoDB ──────────────────────────────
def load_customer_events(customer_id):
    db = get_mongo_database()
    document = db["customer_events"].find_one({"customer_id": customer_id})
    if document is None:
        logger.warning(f"  No events document found for {customer_id}. Returning empty list.")
        return []
    return document["events"]


# ── Load support tickets from MongoDB ─────────────────────────────
def load_support_tickets(customer_id):
    db = get_mongo_database()
    document = db["support_tickets"].find({"customer_id": customer_id})
    return list(document)


# ── Load sessions from MongoDB ─────────────────────────────────────
def load_sessions(customer_id):
    db = get_mongo_database()
    document = db["behavioural_sessions"].find({"customer_id": customer_id})
    return list(document)


# ── Load notifications from MongoDB ───────────────────────────────
def load_notifications(customer_id):
    db = get_mongo_database()
    document = db["notifications_responses"].find_one({"customer_id": customer_id})
    if document is None:
        logger.warning(f"  No notifications document found for {customer_id}. Returning empty dict.")
        return {}
    return document


# ── Save raw data to data/raw/ as JSON ────────────────────────────
def save_raw_data(data: list, filename: str = None):
    """
    Saves the combined customer data to data/raw/ at the project root.

    Args:
        data:     The list of combined customer dicts from load_all_customers().
        filename: Optional custom filename. If not provided, a timestamped
                  name is generated automatically.
                  Example: customers_raw_2026-06-28_14-32-01.json
    """
    logger.info("Phase 4 — Saving raw data to disk...")

    # Always save to project root / data / raw
    raw_dir = ROOT_DIR / "data" / "raw"
    os.makedirs(raw_dir, exist_ok=True)

    # Generate a timestamped filename if none provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"customers_raw_{timestamp}.json"

    filepath = raw_dir / filename

    # MongoDB documents contain ObjectId fields (_id) which are not
    # JSON serialisable by default. default=str handles this cleanly.
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    size_kb = os.path.getsize(filepath) / 1024
    logger.info(f"Phase 4 — Saved {len(data)} records to: {filepath}  ({size_kb:.1f} KB)")

    return filepath


# ── Load all customers: combining all sources ──────────────────────
def load_all_customers():
    logger.info("=" * 55)
    logger.info("PulseAI — Data Loader Started")
    logger.info("=" * 55)

    # Phase 1 — Load customer list from Supabase
    customers = load_customers()
    total = len(customers)

    # Phase 2 — Loop and load all data per customer
    logger.info(f"Phase 2 — Loading full data for {total} customers...")
    combined_customer_data = []

    for index, customer in enumerate(customers, start=1):
        cid = customer["customer_id"]
        print(f"  [{index}/{total}] Loading data for {cid}...")

        combined_customer_data.append({
            "customer_id":   cid,
            "profile":       customer,
            "transactions":  load_transactions(cid),
            "subscriptions": load_subscriptions(cid),
            "events":        load_customer_events(cid),
            "tickets":       load_support_tickets(cid),
            "sessions":      load_sessions(cid),
            "notifications": load_notifications(cid),
        })

        percentage = (index / total) * 100
        print(f"  Loading customers... {index}/{total}  ({percentage:.0f}%)", end="\r")

    print(f"  Loading customers... {total}/{total}  (100%) — Done!          ")

    # Phase 3 — Done loading
    logger.info("=" * 55)
    logger.info(f"Phase 3 — All data loaded. Total customers: {total}")
    logger.info("=" * 55)

    # Phase 4 — Save to disk
    save_raw_data(combined_customer_data)

    return combined_customer_data


if __name__ == "__main__":
    data = load_all_customers()
    print(f"\nTotal customers loaded  : {len(data)}")
    print("Keys per customer       :", list(data[0].keys()))
    print("Events for first customer :", len(data[0]["events"]))
    print("Tickets for first customer:", len(data[0]["tickets"]))