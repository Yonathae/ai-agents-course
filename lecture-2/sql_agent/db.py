"""A tiny SQLite database for the running example.

Two tables:
    customers(customer_id, name)
    orders(order_id, customer_id, amount, order_date, status)

status is stored as a one-letter code: S = shipped, R = returned, C = cancelled.
The codes are visible only in the data, not in the schema.

The data is fixed, so every run of the lecture sees the same answers.
"""
import random
import sqlite3
from pathlib import Path

CUSTOMERS = [
    (1, "Alice Cohen"),
    (2, "Ben Levi"),
    (3, "Carmen Diaz"),
    (4, "David Mizrahi"),
    (5, "Eva Novak"),
    (6, "Farid Haddad"),
    (7, "Grace Kim"),
    (8, "Hiro Tanaka"),
]

# (order_id, customer_id, amount, order_date, status)
ORDERS = [
    (101, 1, 120.00, "2025-01-14", "S"),
    (102, 2, 75.50, "2025-02-03", "S"),
    (103, 3, 210.00, "2025-02-20", "S"),
    (104, 1, 45.00, "2025-03-11", "R"),
    (105, 4, 330.00, "2025-04-02", "S"),
    (106, 5, 60.00, "2025-05-17", "S"),
    (107, 2, 99.90, "2025-06-09", "C"),
    (108, 6, 150.00, "2025-07-21", "S"),
    (109, 3, 80.00, "2025-08-30", "S"),
    (110, 4, 25.00, "2025-09-12", "R"),
    (111, 5, 410.00, "2025-10-05", "S"),
    (112, 1, 55.00, "2025-11-23", "S"),
    (113, 6, 95.00, "2025-12-31", "S"),
    (114, 2, 130.00, "2026-01-01", "S"),
    (115, 1, 70.00, "2026-01-19", "S"),
    (116, 7, 260.00, "2026-02-14", "S"),
    (117, 3, 40.00, "2026-03-03", "S"),
    (118, 2, 88.00, "2026-03-28", "R"),
    (119, 7, 115.00, "2026-04-16", "S"),
    (120, 1, 300.00, "2026-05-09", "S"),
    (121, 3, 65.00, "2026-06-22", "C"),
    (122, 7, 50.00, "2026-07-07", "R"),
    (123, 2, 175.00, "2026-08-18", "S"),
    (124, 1, 20.00, "2026-09-02", "S"),
]
# Ordered in 2025 but not 2026: David Mizrahi, Eva Novak, Farid Haddad.
# Grace Kim ordered only in 2026; Hiro Tanaka never ordered.
# Returned: 104, 110, 118, 122; cancelled: 107, 121; all others shipped.

SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL
);
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    amount      REAL NOT NULL,
    order_date  TEXT NOT NULL,  -- ISO format YYYY-MM-DD
    status      TEXT NOT NULL
);
"""


def create_database(path="shop.db", customers=CUSTOMERS, orders=ORDERS):
    """Write the database to `path` (overwriting it) and return the path."""
    path = Path(path)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO customers VALUES (?, ?)", customers)
    con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders)
    con.commit()
    con.close()
    return path


def random_database(path, seed, n_orders=30):
    """Same schema, random orders: used to expose queries that are only
    accidentally correct on the fixed database."""
    rng = random.Random(seed)
    orders = []
    for k in range(n_orders):
        year = rng.choice([2025, 2026])
        date = f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        status = rng.choices("SRC", weights=[75, 15, 10])[0]
        orders.append((200 + k, rng.randint(1, len(CUSTOMERS)),
                       round(rng.uniform(10, 400), 2), date, status))
    return create_database(path, CUSTOMERS, orders)
