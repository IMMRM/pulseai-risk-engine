"""
config.py
---------
Loads environment variables from .env once.
Everything else in the project imports from here.

Usage:
    from src.config import MONGO_URI, SUPABASE_URL, ...
"""

import os
from dotenv import load_dotenv

# Load the .env file into os.environ
# This works from any directory — it finds .env by walking up from this file
load_dotenv(override=True)


# ── Supabase (PostgreSQL) ──────────────────────────────────────────
SUPABASE_URL      = os.getenv("SUPABASE_URL")
SUPABASE_KEY      = os.getenv("SUPABASE_KEY")
SUPABASE_DB_HOST  = os.getenv("SUPABASE_DB_HOST")
SUPABASE_DB_PORT  = int(os.getenv("SUPABASE_DB_PORT", 5432))
SUPABASE_DB_NAME  = os.getenv("SUPABASE_DB_NAME", "postgres")
SUPABASE_DB_USER  = os.getenv("SUPABASE_DB_USER", "postgres")
SUPABASE_DB_PASS  = os.getenv("SUPABASE_DB_PASSWORD")

# ── MongoDB Atlas ──────────────────────────────────────────────────
MONGO_URI         = os.getenv("MONGO_URI")
MONGO_DB_NAME     = os.getenv("MONGO_DB_NAME", "pulseai")

# ── API ────────────────────────────────────────────────────────────
API_KEY           = os.getenv("API_KEY")
API_HOST          = os.getenv("API_HOST", "0.0.0.0")
API_PORT          = int(os.getenv("API_PORT", 8000))

# ── Model ──────────────────────────────────────────────────────────
MODEL_VERSION     = os.getenv("MODEL_VERSION", "1.0.0")
CHECKPOINT_DIR    = os.getenv("CHECKPOINT_DIR", "checkpoints")
LOG_DIR           = os.getenv("LOG_DIR", "logs")