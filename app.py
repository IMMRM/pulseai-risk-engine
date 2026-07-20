"""
PulseAI — Hugging Face Spaces entry point.
Thin launcher; all UI logic lives in src/interface/.
"""
from src.interface.ui import build_app

app = build_app()

if __name__ == "__main__":
    app.launch()