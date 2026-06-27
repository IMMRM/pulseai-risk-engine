"""
Supabase / PostgreSQL connection and query helpers.
"""
import psycopg2
from psycopg2 import sql
from src.config import SUPABASE_DB_HOST, SUPABASE_DB_PORT, SUPABASE_DB_NAME, SUPABASE_DB_USER, SUPABASE_DB_PASS



# Function to establish a connection to the Supabase PostgreSQL database
def get_supabase_connection():
    try:
        conn = psycopg2.connect(
            host=SUPABASE_DB_HOST,
            port=SUPABASE_DB_PORT,
            dbname=SUPABASE_DB_NAME,
            user=SUPABASE_DB_USER,
            password=SUPABASE_DB_PASS
        )
        return conn
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")
        return None
    
# Fetch all customers
def fetch_all_customers():
    conn = get_supabase_connection()
    if conn is None:
        return []

    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM customers;")
            customers = cursor.fetchall()
            return customers
    except Exception as e:
        print(f"Error fetching customers: {e}")
        return []
    finally:
        conn.close()

if __name__ == "__main__":
    rows = fetch_all_customers()
    print(f"Connected! Fetched {len(rows)} rows.")
    for row in rows:
        print(row)