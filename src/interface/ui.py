"""
Builds the full Gradio interface by assembling all tabs.
"""
import gradio as gr
from src.interface.tab.retrain_tab import build_retrain_tab
from src.interface.tab.score_tab import build_score_tab
from src.interface.tab.top_risk_tab import build_top_risk_tab
from src.interface.tab.timeline_tab import build_timeline_tab

        

def build_app():
    with gr.Blocks(title="PulseAI") as app:
        gr.Markdown("# PulseAI — Customer Risk Prediction")

        with gr.Tab("Score a Customer"):
            build_score_tab()

        with gr.Tab("Top Risk Customers"):
            build_top_risk_tab()
        with gr.Tab("Event Timeline"):
            build_timeline_tab()
        with gr.Tab("Retrain Model"):
            build_retrain_tab()

    return app