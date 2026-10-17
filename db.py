import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.environ.get("DB_PATH", "variance.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  entity TEXT, period TEXT, currency TEXT,
  status TEXT NOT NULL,              -- ok | ai_failed
  confidence INTEGER,                -- final 0-100
  net_profit_impact REAL,
  input_json TEXT, logic_json TEXT, ai_json TEXT, flags_json TEXT,
  provider TEXT, model TEXT,
  tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0,
  latency_ms INTEGER DEFAULT 0, cost_usd REAL DEFAULT 0,
  error TEXT,
  feedback INTEGER                   -- 1 helpful, -1 not helpful, NULL none
);
"""


def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init():
    with conn() as c:
        c.executescript(SCHEMA)


def save_report(**k):
    with conn() as c:
        cur = c.execute(
            """INSERT INTO reports (created_at, entity, period, currency, status, confidence,
               net_profit_impact, input_json, logic_json, ai_json, flags_json, provider, model,
               tokens_in, tokens_out, latency_ms, cost_usd, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), k["entity"], k["period"],
             k["currency"], k["status"], k.get("confidence"), k.get("net_profit_impact"),
             json.dumps(k["input"]), json.dumps(k["logic"], default=str),
             json.dumps(k.get("ai")) if k.get("ai") else None, json.dumps(k.get("flags", [])),
             k.get("provider"), k.get("model"), k.get("tokens_in", 0), k.get("tokens_out", 0),
             k.get("latency_ms", 0), k.get("cost_usd", 0), k.get("error")))
        return cur.lastrowid


def get_report(rid):
    with conn() as c:
        r = c.execute("SELECT * FROM reports WHERE id=?", (rid,)).fetchone()
    if not r:
        return None
    d = dict(r)
    for f in ("input_json", "logic_json", "ai_json", "flags_json"):
        d[f[:-5]] = json.loads(d[f]) if d[f] else None
    return d


def list_reports(limit=200):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT id, created_at, entity, period, currency, status, confidence, net_profit_impact, "
            "feedback, flags_json FROM reports ORDER BY id DESC LIMIT ?", (limit,))]


def set_feedback(rid, value):
    with conn() as c:
        c.execute("UPDATE reports SET feedback=? WHERE id=?", (value, rid))


def delete_report(rid):
    with conn() as c:
        c.execute("DELETE FROM reports WHERE id=?", (rid,))


def stats():
    with conn() as c:
        one = c.execute("""SELECT COUNT(*) n,
            SUM(status='ok') ok, SUM(status='ai_failed') failed,
            AVG(CASE WHEN status='ok' THEN confidence END) avg_conf,
            AVG(latency_ms) avg_latency, SUM(cost_usd) cost,
            SUM(tokens_in) tin, SUM(tokens_out) tout,
            SUM(feedback=1) up, SUM(feedback=-1) down,
            SUM(status='ok' AND confidence<50) low_conf
            FROM reports""").fetchone()
        ai = [r["ai_json"] for r in c.execute("SELECT ai_json FROM reports WHERE ai_json IS NOT NULL")]
        flags = [r["flags_json"] for r in c.execute("SELECT flags_json FROM reports WHERE flags_json IS NOT NULL")]
        daily = [dict(r) for r in c.execute(
            "SELECT substr(created_at,1,10) day, COUNT(*) n FROM reports GROUP BY day ORDER BY day DESC LIMIT 14")]
    classes, flag_counts = {}, {}
    for a in ai:
        for d in json.loads(a).get("drivers", []):
            classes[d.get("classification", "unclear")] = classes.get(d.get("classification", "unclear"), 0) + 1
    for f in flags:
        for x in json.loads(f):
            flag_counts[x] = flag_counts.get(x, 0) + 1
    return dict(one), classes, flag_counts, daily
