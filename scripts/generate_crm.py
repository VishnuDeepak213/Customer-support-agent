"""Generate a mock CRM SQLite database with Faker.
Creates tables: customers, orders, tickets, escalations, conversations, security_logs
"""
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

DB_PATH = Path(__file__).resolve().parents[1] / "crm.db"

def create_schema(conn):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            plan TEXT,
            join_date TEXT,
            status TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER,
            product TEXT,
            amount REAL,
            date TEXT,
            status TEXT,
            FOREIGN KEY(customer_id) REFERENCES customers(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER,
            issue_type TEXT,
            priority TEXT,
            resolved INTEGER,
            FOREIGN KEY(customer_id) REFERENCES customers(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS escalations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_number TEXT,
            reason TEXT,
            priority TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_query TEXT,
            agent_response TEXT,
            tool_calls TEXT,
            timestamp TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS security_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            input_text TEXT,
            reason TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()

def populate(conn, n_customers=500):
    fake = Faker()
    plans = ["free", "starter", "pro", "enterprise"]
    statuses = ["active", "past_due", "cancelled"]
    products = ["Widget A", "Widget B", "Service X", "Subscription Pro", "Addon" ]
    issue_types = ["billing", "login", "bug", "feature_request", "other"]

    cur = conn.cursor()

    customers = []
    for _ in range(n_customers):
        name = fake.name()
        email = fake.unique.email()
        plan = random.choices(plans, weights=[0.5,0.25,0.18,0.07])[0]
        join_date = fake.date_between(start_date='-3y', end_date='today').isoformat()
        status = random.choices(statuses, weights=[0.85,0.1,0.05])[0]
        cur.execute("INSERT INTO customers (name,email,plan,join_date,status) VALUES (?,?,?,?,?)",
                    (name,email,plan,join_date,status))
        customers.append(cur.lastrowid)

    # orders
    for cid in customers:
        n = random.choices([0,1,2,3,4,5], weights=[0.1,0.3,0.25,0.15,0.1,0.1])[0]
        for _ in range(n):
            product = random.choice(products)
            amount = round(random.uniform(5,500),2)
            date = (datetime.utcnow() - timedelta(days=random.randint(0,1000))).date().isoformat()
            status = random.choice(["completed","refunded","processing"])
            cur.execute("INSERT INTO orders (customer_id,product,amount,date,status) VALUES (?,?,?,?,?)",
                        (cid, product, amount, date, status))

    # tickets
    for cid in customers:
        n = random.choices([0,1,2], weights=[0.6,0.3,0.1])[0]
        for _ in range(n):
            issue_type = random.choice(issue_types)
            priority = random.choices(["low","medium","high"], weights=[0.6,0.3,0.1])[0]
            resolved = random.choices([0,1], weights=[0.3,0.7])[0]
            cur.execute("INSERT INTO tickets (customer_id,issue_type,priority,resolved) VALUES (?,?,?,?)",
                        (cid,issue_type,priority,resolved))

    conn.commit()

def sample_print(conn, limit=5):
    cur = conn.cursor()
    print("Sample customers:")
    for row in cur.execute("SELECT id,name,email,plan,status,join_date FROM customers LIMIT ?", (limit,)):
        print(row)
    print('\nSample orders:')
    for row in cur.execute("SELECT id,customer_id,product,amount,date,status FROM orders LIMIT ?", (limit,)):
        print(row)
    print('\nSample tickets:')
    for row in cur.execute("SELECT id,customer_id,issue_type,priority,resolved FROM tickets LIMIT ?", (limit,)):
        print(row)

if __name__ == '__main__':
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    create_schema(conn)
    populate(conn, n_customers=500)
    sample_print(conn)
    conn.close()
