"""Tool functions that query the CRM database and perform escalations.
These are wrapped as LangChain Tools in app/agent.py
"""
from app.db import get_connection
import uuid
from datetime import datetime


def customer_lookup(email: str) -> str:
    """Return basic info about a customer by email."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id,name,email,plan,join_date,status FROM customers WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return f"No customer found with email {email}"
    return f"id:{row['id']} | name:{row['name']} | email:{row['email']} | plan:{row['plan']} | status:{row['status']} | join_date:{row['join_date']}"


def order_history(customer_id: int) -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id,product,amount,date,status FROM orders WHERE customer_id = ? ORDER BY date DESC LIMIT 5",
        (customer_id,)
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return f"No orders found for customer_id {customer_id}"
    lines = []
    for r in rows:
        lines.append(f"order_id:{r['id']} | product:{r['product']} | amount:{r['amount']} | date:{r['date']} | status:{r['status']}")
    return "\n".join(lines)


def ticket_status(customer_id: int) -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id,issue_type,priority,resolved FROM tickets WHERE customer_id = ? AND resolved = 0", (customer_id,))
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return f"No open tickets for customer_id {customer_id}"
    lines = []
    for r in rows:
        lines.append(f"ticket_id:{r['id']} | issue_type:{r['issue_type']} | priority:{r['priority']}")
    return "\n".join(lines)


def escalate_to_human(reason: str, priority: str = "high") -> str:
    conn = get_connection()
    cur = conn.cursor()
    ticket_number = f"ESC-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO escalations (ticket_number,reason,priority,created_at) VALUES (?,?,?,?)",
                (ticket_number, reason, priority, now))
    conn.commit()
    conn.close()
    return ticket_number


def order_lookup(order_id: int) -> str:
    """Return order details and customer_id for an order_id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, customer_id, product, amount, date, status FROM orders WHERE id = ?", (order_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return f"No order found with id {order_id}"
    return f"order_id:{row['id']} | customer_id:{row['customer_id']} | product:{row['product']} | amount:{row['amount']} | date:{row['date']} | status:{row['status']}"
