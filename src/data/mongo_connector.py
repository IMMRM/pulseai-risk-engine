"""
MongoDB Atlas connection and collection helpers.
"""
from pymongo import MongoClient
from src.config import MONGO_URI, MONGO_DB_NAME


def get_database():
    """Connect to MongoDB Atlas and return the pulseai database."""
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]

def fetch_customer_events(customer_id):
    """Fetch all events for a given customer from the events collection."""
    db = get_database()
    events_collection = db["customer_events"]
    return list(events_collection.find({"customer_id": customer_id}))

if __name__ == "__main__":
    db = get_database()
    collections = db.list_collection_names()
    print(f"Connected! Collections found: {collections}")
    # fetch all events for a specific customer (example customer_id: "customer_123")
    customer_id = "CUST-B1A7E63F"
    events = fetch_customer_events(customer_id)
    print(f"Events for customer {customer_id}: {events}")