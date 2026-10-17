"""Variance Explanation Assistant - Flask app.

Pipeline (matches the brief):
 1 structured input form  -> templates/index.html
 2 input validation       -> logic.validate
 3 logic layer (pre-AI)   -> logic.analyse
 4 prompt construction    -> llm.build_prompt
 5 LLM API call           -> llm.call_llm
 6 output parsing         -> llm.parse
 7 confidence/flag check  -> llm.check
 8 database storage       -> db.save_report
 9 structured UI output   -> templates/report.html
10 admin dashboard        -> /admin
"""
import os
from functools import wraps

from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   session, url_for)

import db
import llm
import logic

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")
db.init()


@app.template_filter("money")
def money(v, d=0):
    return "-" if v is None else f"{v:,.{d}f}"


@app.template_filter("signed")
def signed(v):
    return "-" if v is None else f"{v:+,.0f}"


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    try:
        clean = logic.validate(payload)                      # stage 2
    except logic.ValidationError as e:
        return jsonify(ok=False, errors=e.errors), 400
    analysis = logic.analyse(clean, {k: float(v) for k, v in (payload.get("settings") or {}).items()
                                     if k in ("pct_threshold", "min_share")})  # stage 3
    base = dict(entity=clean["entity"], period=clean["period"], currency=clean["currency"],
                input=clean, logic=analysis, net_profit_impact=analysis["totals"]["profit_impact"])
    if not analysis["drivers"]:                              # nothing material: skip the AI, save cost
        rid = db.save_report(**base, status="ok", flags=analysis["flags"], confidence=95,
                             ai=dict(summary="No category crossed the materiality thresholds, so there is "
                                             "nothing significant to explain.", drivers=[],
                                     questions_for_analyst=[]))
        return jsonify(ok=True, id=rid)
    system, user, facts = llm.build_prompt(clean, analysis)  # stage 4
    try:
        raw = llm.call_llm(system, user)                     # stage 5
        parsed = llm.parse(raw["text"])                      # stage 6
        ai, flags, conf = llm.check(parsed, analysis, facts)  # stage 7
        rid = db.save_report(**base, status="ok", ai=ai, flags=flags, confidence=conf,
                             **{k: raw[k] for k in ("provider", "model", "tokens_in", "tokens_out",
                                                    "latency_ms", "cost_usd")})  # stage 8
    except llm.LLMError as e:
        flags = analysis["flags"] + ["ai_unavailable"]
        rid = db.save_report(**base, status="ai_failed", flags=flags, error=str(e),
                             ai=dict(summary=llm.fallback_summary(clean, analysis), drivers=[],
                                     questions_for_analyst=[]))
    return jsonify(ok=True, id=rid)


@app.get("/report/<int:rid>")
def report(rid):
    r = db.get_report(rid)
    if not r:
        abort(404)
    return render_template("report.html", r=r)


@app.post("/report/<int:rid>/feedback")
def feedback(rid):
    v = 1 if request.form.get("v") == "up" else -1
    db.set_feedback(rid, v)
    return redirect(url_for("report", rid=rid))


@app.get("/history")
def history():
    return render_template("history.html", reports=db.list_reports())


# ---------- Stage 10: admin ----------
def admin_only(f):
    @wraps(f)
    def w(*a, **k):
        if not session.get("admin"):
            return redirect(url_for("login"))
        return f(*a, **k)
    return w


@app.route("/admin/login", methods=["GET", "POST"])
def login():
    err = None
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect(url_for("admin"))
        err = "Wrong password."
    return render_template("login.html", err=err)


@app.get("/admin")
@admin_only
def admin():
    s, classes, flag_counts, daily = db.stats()
    return render_template("admin.html", s=s, classes=classes, flags=flag_counts, daily=daily,
                           reports=db.list_reports(50))


@app.post("/admin/delete/<int:rid>")
@admin_only
def admin_delete(rid):
    db.delete_report(rid)
    return redirect(url_for("admin"))


@app.get("/admin/logout")
def logout():
    session.pop("admin", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=bool(os.environ.get("DEBUG")))
