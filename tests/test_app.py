import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["DB_PATH"] = tempfile.mktemp(suffix=".db")
import app as appmod, llm, logic

SAMPLE = json.load(open(os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample.json")))

def test_validation_rejects_bad_input():
    try:
        logic.validate({"period": "", "rows": [{"category": "A", "kind": "revenue", "forecast": "x", "actual": 1}]})
        assert False
    except logic.ValidationError as e:
        assert any("Period" in x for x in e.errors) and any("not a number" in x for x in e.errors)

def test_pvm_identity_and_hints():
    a = logic.analyse(logic.validate(SAMPLE))
    by = {r["category"]: r for r in a["rows"]}
    assert abs(by["Product sales"]["pvm"]["volume_effect"] + by["Product sales"]["pvm"]["price_effect"] - by["Product sales"]["variance"]) < 1e-6
    assert by["Product sales"]["hint"] == "volume"
    assert by["Cloud hosting"]["hint"] == "price"
    assert by["Enterprise contracts"]["hint"] == "timing"
    assert by["Legal and compliance"]["hint"] == "anomaly"
    assert not by["Salaries"]["material"]          # 0.7% variance: must not be flagged

def test_guardrails_downgrade_and_flag_invented_numbers():
    clean = logic.validate(SAMPLE); a = logic.analyse(clean)
    _, _, facts = llm.build_prompt(clean, a)
    fake = {"summary": "Revenue missed by 9,999,999.", "questions_for_analyst": [], "drivers": [
        {"category": "Marketing spend", "classification": "price", "confidence": 95,
         "explanation": "Agency rates rose 40%.", "evidence_used": []}]}
    ai, flags, conf = llm.check(fake, a, facts)
    d = ai["drivers"][0]
    assert d["classification"] == "unclear" and d["confidence"] <= 55
    assert "ungrounded_number" in flags and "ungrounded_number_in_summary" in flags and conf <= 45

def test_end_to_end_with_mock_llm(monkeypatch=None):
    def fake_call(system, user):
        facts = json.loads(user.split("\n", 1)[1])
        drivers = [{"category": d["category"], "classification": d["rule_hint"], "confidence": 80,
                    "explanation": f"{d['category']} moved {d['figures']['variance']} ({d['figures']['variance_pct']}).",
                    "evidence_used": d["rule_evidence"], "missing_info": ""} for d in facts["drivers"]]
        return dict(text=json.dumps({"summary": "ok", "drivers": drivers, "questions_for_analyst": ["q?"]}),
                    provider="mock", model="mock", tokens_in=10, tokens_out=10, latency_ms=5, cost_usd=0)
    orig = llm.call_llm; llm.call_llm = fake_call
    try:
        c = appmod.app.test_client()
        r = c.post("/api/analyze", json=SAMPLE); assert r.get_json()["ok"]
        rid = r.get_json()["id"]
        assert c.get(f"/report/{rid}").status_code == 200
        assert c.get("/history").status_code == 200
        assert c.get("/admin").status_code == 302
        c.post("/admin/login", data={"password": "admin"})
        assert b"Admin dashboard" in c.get("/admin").data
    finally:
        llm.call_llm = orig

def test_ai_failure_still_saves_report():
    os.environ.pop("OPENAI_API_KEY", None); os.environ.pop("ANTHROPIC_API_KEY", None)
    c = appmod.app.test_client()
    rid = c.post("/api/analyze", json=SAMPLE).get_json()["id"]
    page = c.get(f"/report/{rid}").data.decode()
    assert "AI service was unavailable" in page and "Product sales" in page

def test_bad_request_returns_400():
    r = appmod.app.test_client().post("/api/analyze", json={"rows": []})
    assert r.status_code == 400 and r.get_json()["errors"]
