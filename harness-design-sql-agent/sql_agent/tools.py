"""The four tools the harness exposes to the model.

Every tool returns a string observation. Errors are returned as observations
too (prefixed with "ERROR:"), so the harness can decide whether to feed them
back to the model (recovery) or stop.
"""
import json
import sqlite3
import time

MAX_ROWS = 50        # result-size limit
TIMEOUT_S = 2.0      # per-query time limit


def connect_readonly(db_path):
    """Read-only connection: writes fail at the SQLite level."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def run_query(db_path, query, max_rows=MAX_ROWS, timeout_s=TIMEOUT_S):
    """Execute one SQL query under time and size limits.
    Returns (columns, rows, truncated)."""
    con = connect_readonly(db_path)
    deadline = time.perf_counter() + timeout_s
    # SQLite calls this every 1000 VM steps; a non-zero return aborts the query.
    con.set_progress_handler(lambda: int(time.perf_counter() > deadline), 1000)
    try:
        cur = con.execute(query)
        rows = cur.fetchmany(max_rows + 1)
        cols = [d[0] for d in cur.description] if cur.description else []
    finally:
        con.close()
    return cols, rows[:max_rows], len(rows) > max_rows


# ---- tool implementations ------------------------------------------------

def list_tables(db_path):
    con = connect_readonly(db_path)
    names = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    con.close()
    return ", ".join(names)


def describe_table(db_path, table_name):
    con = connect_readonly(db_path)
    cols = con.execute("SELECT name, type FROM pragma_table_info(?)",
                       (table_name,)).fetchall()
    con.close()
    if not cols:
        return f"ERROR: no table named {table_name!r}"
    return ", ".join(f"{n} {t}" for n, t in cols)


def execute_sql(db_path, query):
    try:
        cols, rows, truncated = run_query(db_path, query)
    except sqlite3.Error as e:
        return f"ERROR: {type(e).__name__}: {e}"
    out = [" | ".join(cols)] + [" | ".join(map(str, r)) for r in rows]
    if truncated:
        out.append(f"... (truncated to {MAX_ROWS} rows)")
    return "\n".join(out) if rows else " | ".join(cols) + "\n(no rows)"


TOOLS = {
    "list_tables": list_tables,
    "describe_table": describe_table,
    "execute_sql": execute_sql,
    "submit_answer": execute_sql,   # same execution; the loop ends on success
}


def call_tool(db_path, name, arguments_json):
    """Dispatch a model tool call. Malformed calls become error observations."""
    if name not in TOOLS:
        return f"ERROR: unknown tool {name!r}"
    try:
        args = json.loads(arguments_json or "{}")
        return TOOLS[name](db_path, **args)
    except (json.JSONDecodeError, TypeError) as e:
        return f"ERROR: bad arguments for {name}: {e}"


# ---- JSON schemas sent to the model ---------------------------------------

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_tables",
        "description": "List the tables in the database.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "describe_table",
        "description": "List the columns and types of one table.",
        "parameters": {"type": "object",
                       "properties": {"table_name": {"type": "string"}},
                       "required": ["table_name"]},
    }},
    {"type": "function", "function": {
        "name": "execute_sql",
        "description": "Run a read-only SQLite query and return the rows "
                       f"(at most {MAX_ROWS}).",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Submit the final query. Its result is your answer, so it must "
                       "return exactly what the question asks for: only the requested "
                       "columns (e.g. names, not IDs) and only the requested rows.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]},
    }},
]


def schema_text(db_path):
    """The full schema as text: what a single-shot harness puts in the prompt."""
    return "\n".join(f"{t}({describe_table(db_path, t)})"
                     for t in list_tables(db_path).split(", "))
