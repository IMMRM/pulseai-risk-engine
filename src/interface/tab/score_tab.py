"""
Tab 1 — Score a single customer.
"""
import gradio as gr

from src.inference.scorer import score_customer          # ← add this import
from src.interface.resources import model, vocab, all_customers
from src.logger import get_logger

logger = get_logger(__name__)


def score_one(customer_id):
    result = score_customer(customer_id, model, vocab, all_customers)
    if result is None:
        return "Customer not found", "Please check the ID"
    return str(result["risk_score"]), result["risk_label"]


def build_score_tab():
    customer_input = gr.Textbox(label="Customer ID", placeholder="CUST-B1A7E63F")
    score_button = gr.Button("Score", variant="primary")

    score_output = gr.Textbox(label="Risk Score")
    label_output = gr.Textbox(label="Risk Label")

    score_button.click(
        fn=score_one,
        inputs=customer_input,
        outputs=[score_output, label_output],
    )