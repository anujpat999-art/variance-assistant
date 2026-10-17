"""Logic layer: runs BEFORE any AI call.

Validates input, computes variances, ranks contributors, and pre-classifies
drivers with deterministic rules. The LLM only gets these computed facts, so it
cannot invent numbers, and has to say "unclear" when the data doesn't support a cause.
"""
import math
import re
import statistics

CLASSES = ("volume", "price", "timing", "anomaly", "unclear")
MAX_ROWS = 40

TIMING_WORDS = ("delay", "delayed", "pushed", "slipped", "slip", "deferred", "defer",
                "phasing", "timing", "postponed", "moved to", "shifted", "next month",
                "next quarter", "prepaid", "early", "brought forward", "invoice came late")
ANOMALY_WORDS = ("one-time", "one-off", "one off", "write-off", "write off", "reversal",
                 "error", "settlement", "penalty", "refund", "duplicate", "reclass")

DEFAULT_SETTINGS = {
    "pct_threshold": 5.0,   # a line is "material" if |variance %| >= this ...
    "min_share": 5.0,       # ... AND it explains at least this % of total absolute variance
    "big_share": 25.0,      # ... or it alone explains this much, regardless of %
    "top_n": 5,
}


class ValidationError(Exception):
    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = errors


def _num(v):
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, str):
        v = v.replace(",", "").strip()
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"'{v}' is not a number")
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"'{v}' is not a valid number")
    return f


# ---------- Stage 2: input validation ----------
def validate(payload):
    errors, rows = [], []
    entity = (payload.get("entity") or "").strip()
    period = (payload.get("period") or "").strip()
    currency = (payload.get("currency") or "USD").strip().upper()[:5]
    if not period:
        errors.append("Period is required (for example 'Sep 2026').")
    if len(entity) > 80 or len(period) > 40:
        errors.append("Entity or period is too long.")
    raw = payload.get("rows") or []
    if not raw:
        errors.append("Add at least one category.")
    if len(raw) > MAX_ROWS:
        errors.append(f"Too many categories (max {MAX_ROWS}).")
    seen = set()
    for i, r in enumerate(raw[:MAX_ROWS], 1):
        label = f"Row {i}"
        name = (r.get("category") or "").strip()
        if not name:
            errors.append(f"{label}: category name is missing.")
            continue
        label = f"'{name}'"
        if name.lower() in seen:
            errors.append(f"{label}: duplicate category.")
            continue
        seen.add(name.lower())
        kind = (r.get("kind") or "expense").strip().lower()
        if kind not in ("revenue", "expense"):
            errors.append(f"{label}: type must be revenue or expense.")
            continue
        try:
            fc, ac = _num(r.get("forecast")), _num(r.get("actual"))
            fu, au = _num(r.get("forecast_units")), _num(r.get("actual_units"))
            fp, ap = _num(r.get("forecast_price")), _num(r.get("actual_price"))
        except ValueError as e:
            errors.append(f"{label}: {e}.")
            continue
        if fc is None or ac is None:
            errors.append(f"{label}: forecast and actual are both required.")
            continue
        if fc < 0 or ac < 0:
            errors.append(f"{label}: enter amounts as positive numbers and pick revenue or expense.")
            continue
        note = (r.get("note") or "").strip()
        if len(note) > 500:
            errors.append(f"{label}: note is longer than 500 characters.")
            continue
        has_pvm = None not in (fu, au, fp, ap)
        if any(x is not None for x in (fu, au, fp, ap)) and not has_pvm:
            errors.append(f"{label}: give all four of forecast/actual units and price, or none.")
            continue
        rows.append(dict(category=name, kind=kind, forecast=fc, actual=ac, note=note,
                         forecast_units=fu, actual_units=au, forecast_price=fp, actual_price=ap,
                         has_pvm=has_pvm))
    if errors:
        raise ValidationError(errors)
    return dict(entity=entity or "Unnamed entity", period=period, currency=currency, rows=rows)


# ---------- Stage 3: logic layer ----------
def _pvm(r):
    """Price-volume split. Exact identity: AU*AP - FU*FP = (AU-FU)*FP + (AP-FP)*AU."""
    total = r["actual_units"] * r["actual_price"] - r["forecast_units"] * r["forecast_price"]
    vol = (r["actual_units"] - r["forecast_units"]) * r["forecast_price"]
    price = (r["actual_price"] - r["forecast_price"]) * r["actual_units"]
    implied_fc = r["forecast_units"] * r["forecast_price"]
    consistent = r["forecast"] == 0 or abs(implied_fc - r["forecast"]) / r["forecast"] <= 0.01
    return dict(volume_effect=vol, price_effect=price, total=total, consistent=consistent)


def analyse(clean, settings=None):
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    rows = []
    for r in clean["rows"]:
        var = r["actual"] - r["forecast"]
        pct = (var / r["forecast"] * 100) if r["forecast"] else None
        favorable = var >= 0 if r["kind"] == "revenue" else var <= 0
        rows.append({**r, "variance": var, "pct": pct, "favorable": favorable,
                     "profit_impact": var if r["kind"] == "revenue" else -var})

    total_abs = sum(abs(r["variance"]) for r in rows) or 1.0
    for r in rows:
        r["share"] = abs(r["variance"]) / total_abs * 100

    # robust outlier check on |pct| (median absolute deviation)
    pcts = [abs(r["pct"]) for r in rows if r["pct"] is not None]
    med = statistics.median(pcts) if pcts else 0
    mad = statistics.median([abs(p - med) for p in pcts]) if pcts else 0

    for r in rows:
        r["material"] = bool(
            r["variance"] != 0 and (
                ((r["pct"] is None or abs(r["pct"]) >= s["pct_threshold"]) and r["share"] >= s["min_share"])
                or r["share"] >= s["big_share"]))
        r["hint"], r["evidence"], r["pvm"] = "unclear", [], None
        note = r["note"].lower()
        if r["has_pvm"]:
            r["pvm"] = _pvm(r)
        # rule 1: price/volume decomposition
        if r["pvm"] and r["variance"] != 0:
            v, p = r["pvm"]["volume_effect"], r["pvm"]["price_effect"]
            denom = abs(v) + abs(p) or 1
            r["evidence"].append(f"volume effect {v:,.0f}, price effect {p:,.0f}")
            if not r["pvm"]["consistent"]:
                r["evidence"].append("units x price does not match the forecast amount; split is approximate")
            if abs(v) / denom >= 0.65:
                r["hint"] = "volume"
            elif abs(p) / denom >= 0.65:
                r["hint"] = "price"
            else:
                r["hint"] = "volume"  # mixed; the LLM must mention both effects
                r["evidence"].append("volume and price both contribute")
        # rule 2: analyst note keywords
        t_hit = [w for w in TIMING_WORDS if w in note]
        a_hit = [w for w in ANOMALY_WORDS if w in note]
        if t_hit and r["hint"] == "unclear":
            r["hint"] = "timing"
            r["evidence"].append(f"note mentions timing ('{t_hit[0]}')")
        if a_hit and r["hint"] == "unclear":
            r["hint"] = "anomaly"
            r["evidence"].append(f"note mentions a one-off item ('{a_hit[0]}')")
        # rule 3: statistical outlier vs the other lines
        if r["pct"] is not None and len(pcts) >= 4 and mad > 0:
            z = 0.6745 * (abs(r["pct"]) - med) / mad
            if z > 3.5:
                r["evidence"].append(f"variance % is a statistical outlier (robust z={z:.1f})")
                if r["hint"] == "unclear":
                    r["hint"] = "anomaly"
        if not r["evidence"] and note:
            r["evidence"].append("analyst note provided but no clear cause keyword")

    material = sorted([r for r in rows if r["material"]], key=lambda r: -abs(r["variance"]))
    drivers = material[: s["top_n"]]
    coverage = sum(abs(r["variance"]) for r in drivers) / total_abs * 100

    for r in drivers:
        score = 35
        if r["pvm"]:
            score += 25
        if r["note"]:
            score += 15
        if r["hint"] != "unclear":
            score += 15
        if r["pct"] is None:
            score -= 15
        if r["hint"] == "unclear" and not r["note"]:
            score -= 10
        r["evidence_score"] = max(5, min(95, score))

    rev_var = sum(r["variance"] for r in rows if r["kind"] == "revenue")
    exp_var = sum(r["variance"] for r in rows if r["kind"] == "expense")
    tot_fc_rev = sum(r["forecast"] for r in rows if r["kind"] == "revenue")
    flags = []
    if not drivers:
        flags.append("no_material_variance")
    if coverage < 60 and drivers:
        flags.append("top_drivers_explain_less_than_60pct")
    if any(r["pct"] is None for r in drivers):
        flags.append("zero_forecast_in_driver")
    if drivers and all(r["hint"] == "unclear" for r in drivers):
        flags.append("no_driver_has_supporting_evidence")

    return dict(
        settings=s, rows=rows, drivers=drivers, coverage=coverage, flags=flags,
        totals=dict(revenue_variance=rev_var, expense_variance=exp_var,
                    profit_impact=rev_var - exp_var, total_abs=total_abs,
                    revenue_forecast=tot_fc_rev),
    )
