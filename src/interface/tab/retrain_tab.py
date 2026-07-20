"""
Tab 4 — Retrain the model.
Triggers the training script and reports the result.
"""
import gradio as gr

from src.training.train import train
from src.logger import get_logger

logger = get_logger(__name__)


def run_retrain():
    # TODO 1: call train() to retrain the model
    #   wrap in try/except so a failure doesn't crash the app
    try:
        train()
        return "Retraining complete. New model saved to checkpoints/best_model.pt"
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        return f"Retraining failed: {e}"


def build_retrain_tab():
    gr.Markdown(
        "Retrain the risk model on the latest data. "
        "This may take a moment — the interface will be busy while it runs."
    )

    retrain_button = gr.Button("Retrain Model", variant="primary")
    status_output = gr.Textbox(label="Status", lines=3)

    retrain_button.click(
        fn=run_retrain,
        inputs=None,
        outputs=status_output,
    )