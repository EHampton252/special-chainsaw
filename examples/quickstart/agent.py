"""A toy support agent used by the quickstart suite. Replace it with your own function or endpoint."""

import re

ORDERS = {"1001": "shipped", "1002": "processing"}


def respond(user_input: str, history: list[dict]) -> dict:
    text = user_input.lower()
    # Remember the order number from earlier turns so follow-up questions work.
    earlier = " ".join(m["content"] for m in history if m["role"] == "user")
    order = re.search(r"\b(\d{4})\b", user_input) or re.search(r"\b(\d{4})\b", earlier)

    if "return" in text or "refund" in text:
        subject = f"order {order.group(1)}" if order else "any item"
        return {"output": f"You can return {subject} within 30 days of delivery for a full refund."}
    if "order" in text and order:
        order_id = order.group(1)
        status = ORDERS.get(order_id, "not found")
        return {
            "output": f"Order {order_id} is {status}.",
            "tool_calls": [{"name": "lookup_order", "args": {"order_id": order_id}}],
        }
    return {"output": "I can help with order status and returns."}
