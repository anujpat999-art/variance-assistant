"""Prompt construction, LLM call, output parsing and grounding checks."""
import json
import os
import re
import time
import urllib.error
import urllib.request

from logic import CLASSES

PRICES_PER_MTOK = {  # rough USD per 1M tokens (input, output), for the cost dashboard only
    "gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60),
    "claude-haiku-5-5": (1.0, 5.0),
}

SYSTEM = """You are a finance analyst's assistant that explains forecast-vs-actual variances.
Strict rules:
1. Use ONLY the facts in the JSON the user sends. Never invent causes, events, customers, or numbers.
2. Every driver must be classified as one of: volume, price, timing, anomaly, unclear.
   - Use volume or price only if price/volume figures are provided for that line.
   - Use timing only if the analyst note says so.
   - If the data does not support a cause, classify as "unclear" and say what information is missing.
3. Quote numbers exactly as given in the "figures" strings. Do not do new arithmetic.
4. "rule_hint" is a suggestion from a deterministic rules engine. You may disagree, but explain why.
5. Write for a CFO: short, plain, defensible. Mention direction (favorable/unfavorable).
6. confidence is 0-100: how well the supplied data supports your explanation, not how fluent it sounds.
Return ONLY valid JSON, no markdown, with this shape:
{"summary": "2-4 sentences",
 "drivers": [{"category": "...", "classification": "volume|price|timing|anomaly|unclear",
              "explanation": "1-2 sentences", "evidence_used": ["..."], "confidence": 0-100,
              "missing_info": "what would confirm the cause, or empty string"}],
 "questions_for_analyst": ["up to 3 questions"],
 "overall_confidence": 0-100}"""


def _fmt(x, d=0):
    return f"{x:,.{d}f}"


def build_facts(clean, analysis):
    cur = clean["currency"]
    drivers = []
    for r in analysis["drivers"]:
        fig = {
            "forecast": _fmt(r["forecast"]), "actual": _fmt(r["actual"]),
            "variance": f"{r['variance']:+,.0f}",
            "variance_pct": "n/a (zero forecast)" if r["pct"] is None else f"{r['pct']:+.1f}%",
            "share_of_total_variance": f"{r['share']:.1f}%",
        }
        if r["pvm"]:
            fig["volume_effect"] = f"{r['pvm']['volume_effect']:+,.0f}"
            fig["price_effect"] = f"{r['pvm']['price_effect']:+,.0f}"
            fig["units"] = f"{_fmt(r['forecast_units'])} -> {_fmt(r['actual_units'])}"
            fig["unit_price"] = f"{_fmt(r['forecast_price'], 2)} -> {_fmt(r['actual_price'], 2)}"
        drivers.append({
            "category": r["category"], "type": r["kind"],
            "direction": "favorable" if r["favorable"] else "unfavorable",
            "figures": fig, "rule_hint": r["hint"], "rule_evidence": r["evidence"],
            "analyst_note": r["note"] or None,
        })
    t = analysis["totals"]
    return {
        "entity": clean["entity"], "period": clean["period"], "currency": cur,
        "net_profit_impact": f"{t['profit_impact']:+,.0f}",
        "top_drivers_cover_pct_of_total_variance": f"{analysis['coverage']:.0f}%",
        "drivers": drivers,
    }


def build_prompt(clean, analysis):
    facts = build_facts(clean, analysis)
    return SYSTEM, "Explain this variance.\n" + json.dumps(facts, indent=1), facts


def _post(url, headers, body, timeout=45):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class LLMError(Exception):
    pass


def provider_info():
    p = os.environ.get("LLM_PROVIDER") or ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai")
    if p == "anthropic":
        return p, os.environ.get("LLM_MODEL", "claude-haiku-5-5"), os.environ.get("ANTHROPIC_API_KEY")
    return "openai", os.environ.get("LLM_MODEL", "gpt-4o-mini"), os.environ.get("OPENAI_API_KEY")


def call_llm(system, user):
    provider, model, key = provider_info()
    if not key:
        raise LLMError("No API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
    last = None
    for attempt in range(2):  # one retry on transient failures
        t0 = time.time()
        try:
            if provider == "anthropic":
                d = _post("https://api.anthropic.com/v1/messages",
                          {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          {"model": model, "max_tokens": 1500, "temperature": 0.2, "system": system,
                           "messages": [{"role": "user", "content": user}]})
                text = "".join(b.get("text", "") for b in d["content"])
                tin, tout = d["usage"]["input_tokens"], d["usage"]["output_tokens"]
            else:
                d = _post("https://api.openai.com/v1/chat/completions",
                          {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          {"model": model, "temperature": 0.2, "response_format": {"type": "json_object"},
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
                text = d["choices"][0]["message"]["content"]
                tin, tout = d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]
            pin, pout = PRICES_PER_MTOK.get(model, (0, 0))
            return dict(text=text, provider=provider, model=model, tokens_in=tin, tokens_out=tout,
                        latency_ms=int((time.time() - t0) * 1000),
                        cost_usd=(tin * pin + tout * pout) / 1e6)
        except urllib.error.HTTPError as e:
            last = f"{provider} API error {e.code}: {e.read()[:200].decode(errors='ignore')}"
            if e.code not in (429, 500, 502, 503, 529):
                break
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            last = f"{provider} call failed: {e}"
        time.sleep(1.5)
    raise LLMError(last or "Unknown LLM error")


# ---------- Stage 6: output parsing ----------
def parse(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        raise LLMError("Model did not return JSON.")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise LLMError(f"Model returned malformed JSON: {e}")


_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*\.?\d*%?")


def _allowed_numbers(facts):
    out = set()
    for d in facts["drivers"]:
        for v in d["figures"].values():
            for m in _NUM.findall(v):
                out.add(m.replace(",", "").replace("+", "").rstrip("%").lstrip("-"))
    for k in ("net_profit_impact", "top_drivers_cover_pct_of_total_variance"):
        for m in _NUM.findall(str(facts[k])):
            out.add(m.replace(",", "").replace("+", "").rstrip("%").lstrip("-"))
    return out


def _grounded(text, allowed):
    """Return numbers in text that are not in the supplied facts (ignoring small ints)."""
    bad = []
    for m in _NUM.findall(text or ""):
        raw = m.replace(",", "").replace("+", "").lstrip("-")
        pct = raw.endswith("%")
        raw = raw.rstrip("%")
        try:
            val = float(raw)
        except ValueError:
            continue
        if not pct and val < 100:
            continue
        ok = any(abs(val - float(a)) <= max(0.051, abs(val) * 0.005) for a in allowed if a.replace(".", "", 1).isdigit())
        if not ok:
            bad.append(m)
    return bad


# ---------- Stage 7: confidence / flag check ----------
def check(ai, analysis, facts):
    """Validate the model's answer against the logic layer. Returns (clean_ai, flags, confidence)."""
    flags = list(analysis["flags"])
    drivers_by_name = {r["category"].lower(): r for r in analysis["drivers"]}
    allowed = _allowed_numbers(facts)
    out_drivers, conf_parts = [], []
    for d in ai.get("drivers", []):
        name = str(d.get("category", "")).strip()
        src = drivers_by_name.get(name.lower())
        if not src:
            flags.append(f"model_mentioned_unknown_category:{name}")
            continue
        cls = str(d.get("classification", "unclear")).lower()
        if cls not in CLASSES:
            cls = "unclear"
        note = []
        if cls in ("volume", "price") and not src["pvm"]:
            note.append("Downgraded to unclear: no price/volume data was provided for this line.")
            cls = "unclear"
        if cls == "timing" and src["hint"] != "timing":
            note.append("Downgraded to unclear: no timing evidence in the analyst note.")
            cls = "unclear"
        try:
            c = max(0, min(100, int(d.get("confidence", 50))))
        except (TypeError, ValueError):
            c = 50
        bad = _grounded(str(d.get("explanation", "")), allowed)
        if bad:
            note.append(f"Contains figures not in your data: {', '.join(bad[:3])}.")
            flags.append("ungrounded_number")
            c = min(c, 40)
        c = round(0.6 * c + 0.4 * src["evidence_score"])  # blend model and evidence-based confidence
        if cls == "unclear":
            c = min(c, 55)
        conf_parts.append((abs(src["variance"]), c))
        out_drivers.append(dict(
            category=src["category"], classification=cls, explanation=str(d.get("explanation", "")),
            evidence_used=[str(x) for x in d.get("evidence_used", [])][:5], confidence=c,
            missing_info=str(d.get("missing_info", "")), guardrail_notes=note))
    got = {d["category"].lower() for d in out_drivers}
    for name, src in drivers_by_name.items():
        if name not in got:
            flags.append(f"model_skipped_driver:{src['category']}")
    summary = str(ai.get("summary", ""))
    bad = _grounded(summary, allowed)
    if bad:
        flags.append("ungrounded_number_in_summary")
    w = sum(x for x, _ in conf_parts) or 1
    conf = round(sum(x * c for x, c in conf_parts) / w) if conf_parts else 0
    if bad:
        conf = min(conf, 45)
    if analysis["coverage"] < 60:
        conf = min(conf, 60)
    qs = [str(q) for q in ai.get("questions_for_analyst", [])][:3]
    return dict(summary=summary, drivers=out_drivers, questions_for_analyst=qs), sorted(set(flags)), conf


def fallback_summary(clean, analysis):
    """Used when the AI call fails: report still shows the deterministic findings."""
    t = analysis["totals"]
    cur = clean["currency"]
    word = "favorable" if t["profit_impact"] >= 0 else "unfavorable"
    top = ", ".join(f"{r['category']} ({r['variance']:+,.0f})" for r in analysis["drivers"][:3]) or "none material"
    return (f"Net profit impact is {t['profit_impact']:+,.0f} {cur} ({word}). "
            f"Largest drivers: {top}. AI explanation unavailable; showing calculated figures only.")
