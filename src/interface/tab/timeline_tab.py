"""
Tab 3 — Event timeline for a customer.
"""
import gradio as gr

from src.interface.resources import all_customers


def get_timeline(customer_id):
    
    customer = None
    for c in all_customers:
        if c["customer_id"] == customer_id:
            customer = c
            break

    if customer is None:
        return [["Not found", "Check the customer ID"]]

    
    rows = []
    for event in customer["events"]:
        rows.append([event["timestamp"], event["event_type"]])

    return rows


def build_timeline_tab():
    customer_input = gr.Textbox(label="Customer ID", placeholder="CUST-B1A7E63F")
    load_button = gr.Button("Load Timeline", variant="primary")

    timeline_table = gr.Dataframe(
        headers=["Timestamp", "Event"],
        label="Customer event journey (chronological)",
    )

    load_button.click(
        fn=get_timeline,
        inputs=customer_input,
        outputs=timeline_table,
    )