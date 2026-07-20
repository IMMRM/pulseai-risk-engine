"""
Tab 2 — Top risk customers (ranked table).
"""
import gradio as gr

from src.inference.scorer import score_all_customers
from src.interface.resources import model


def get_top_risk():
    results = score_all_customers(model)          # ranked list of dicts
    rows = []
    for r in results:
        rows.append([r["customer_id"], r["risk_score"], r["risk_label"]])
    return rows


def build_top_risk_tab():
    refresh_button = gr.Button("Load Top Risk Customers", variant="primary")

    table = gr.Dataframe(
        headers=["Customer ID", "Risk Score", "Risk Label"],
        label="Customers ranked by risk (highest first)",
    )

    refresh_button.click(
        fn=get_top_risk,
        inputs=None,
        outputs=table,
    )