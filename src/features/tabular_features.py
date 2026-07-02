"""
Computes tabular features: failure rate, tenure, open rate.
"""
from src.logger import get_logger

logger=get_logger(__name__)

def compute_features(customer):
    """
    Convert one customer's structured data into a fixed-order list of numbers.
    """
    # ── transactions ──
    transactions = customer["transactions"]
    total_txns = len(transactions)
    failed_txns = sum(1 for t in transactions if t["payment_status"] == "failed")
    failure_rate = failed_txns / total_txns if total_txns > 0 else 0.0

    # ── profile ──
    tenure_days = customer["profile"]["tenure_days"]
    plan_map = {"basic": 0, "standard": 1, "premium": 2, "enterprise": 3}
    plan_encoded = plan_map.get(customer["profile"]["plan_type"], 0)

    # ── tickets ──
    tickets = customer["tickets"]
    total_tickets = len(tickets)
    escalated_tickets = sum(1 for t in tickets if t.get("escalated") is True)

    # ── notifications ──
    notifications = customer["notifications"]
    if notifications and "engagement_summary" in notifications:
        open_rate = notifications["engagement_summary"].get("open_rate", 0.0)
    else:
        open_rate = 0.0

    # ── sessions ──
    sessions = customer["sessions"]
    total_sessions = len(sessions)
    visited_cancel = 0
    for session in sessions:
        if session.get("risk_signals", {}).get("visited_cancel_page") is True:
            visited_cancel = 1
            break

    # ── subscriptions ──
    ever_downgraded = 0
    for sub in customer["subscriptions"]:
        if sub.get("renewal_status") == "downgraded":
            ever_downgraded = 1
            break

    # ── assemble in FIXED order ──
    features = [
        failure_rate,       # 0
        tenure_days,        # 1
        plan_encoded,       # 2
        total_tickets,      # 3
        escalated_tickets,  # 4
        open_rate,          # 5
        visited_cancel,     # 6
        total_sessions,     # 7
        ever_downgraded,    # 8
    ]
    return features

def build_feature_matrix(all_customers):
    """
    Compute features for every customer.

    Returns:
        matrix:        list of feature rows (one per customer).
        feature_names: list of column names, matching the row order.
    """
    logger.info("Building tabular feature matrix...")

    # Must match the order in compute_features() exactly
    feature_names = [
        "failure_rate",       # 0
        "tenure_days",        # 1
        "plan_encoded",       # 2
        "total_tickets",      # 3
        "escalated_tickets",  # 4
        "open_rate",          # 5
        "visited_cancel",     # 6
        "total_sessions",     # 7
        "ever_downgraded",    # 8
    ]

    matrix = []
    for customer in all_customers:
        row = compute_features(customer)
        matrix.append(row)

    logger.info(
        f"Feature matrix built. {len(matrix)} customers x {len(feature_names)} features."
    )
    return matrix, feature_names

if __name__ == "__main__":
    import json
    with open("data/raw/customers_raw_2026-07-02_09-16-13.json") as f:
        all_customers = json.load(f)

    matrix, names = build_feature_matrix(all_customers)
    print("Matrix size:", len(matrix), "customers")
    print("Features per customer:", len(names))
    print("Names:", names)
    print("First row:", matrix[0])