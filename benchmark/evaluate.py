"""Error metrics for API predictions against tape measurements, and comparisons between algorithm versions."""
import math
import os
from collections import Counter, defaultdict

import numpy as np

DEFAULT_MIN_GROUP_SIZE = 5
FEW_PEOPLE = 30  # below this, results are flagged as indicative only

# (table, column) pairs whose values split the error report into groups.
GROUP_COLUMNS = (
    ("people", "sex"), ("people", "age_band"), ("people", "skin_tone"),
    ("samples", "clothing"), ("samples", "footwear"), ("samples", "background"),
    ("samples", "lighting"), ("samples", "height_source"), ("samples", "device"),
)


def _prediction(record, api_name):
    """(value, None) for a numeric prediction, else (None, why it is missing)."""
    if record is None:
        return None, "not predicted"
    if record["http_status"] != 200:
        return None, f"rejected: {record.get('error_code') or record['http_status']}"
    value = record["response"]["measurements"].get(api_name)
    if value is None:
        return None, record["response"]["measurement_status"].get(api_name, {}).get("status", "missing")
    return float(value), None


def _selected_samples(dataset, splits):
    return [s for s in dataset.usable_samples() if dataset.people[s["person_id"]]["split"] in splits]


def _paired(dataset, samples, predictions, comparison):
    """Rows (sample, predicted, reference) with both values, and a Counter of why the others are missing."""
    references = dataset.reference_values()
    rows, missing = [], Counter()
    for sample in samples:
        reference = references.get(sample["session_id"], {}).get(comparison["reference"])
        if reference is None:
            continue
        value, reason = _prediction(predictions.get(sample["sample_id"]), comparison["api_name"])
        if reason:
            missing[reason] += 1
        else:
            rows.append((sample, value, reference))
    return rows, missing


def error_summary(errors, tolerance=None):
    """Error statistics in cm; error = API - tape, so a positive bias means the API reads high."""
    e = np.asarray(errors, dtype=float)
    if e.size == 0:
        return {"mae": None, "median_ae": None, "p90_ae": None, "max_ae": None, "bias": None, "within_tolerance": None}
    a = np.abs(e)
    return {
        "mae": float(a.mean()),
        "median_ae": float(np.median(a)),
        "p90_ae": float(np.percentile(a, 90)),
        "max_ae": float(a.max()),
        "bias": float(e.mean()),
        "within_tolerance": float(np.mean(a <= tolerance)) if tolerance is not None else None,
    }


def _groups(dataset, rows, min_group_size):
    result = {}
    for table, column in GROUP_COLUMNS:
        by_value = defaultdict(list)
        for sample, value, reference in rows:
            source = dataset.people[sample["person_id"]] if table == "people" else sample
            if source[column] is not None:
                by_value[str(source[column])].append(value - reference)
        if not by_value:
            continue
        shown = [
            {"value": v, "n": len(errs), "mae": float(np.mean(np.abs(errs))), "bias": float(np.mean(errs))}
            for v, errs in sorted(by_value.items()) if len(errs) >= min_group_size
        ]
        result[column] = {"groups": shown, "suppressed_groups": sum(1 for errs in by_value.values() if len(errs) < min_group_size)}
    return result


def technical_error(pairs):
    """Technical error of measurement for repeat pairs: sqrt(sum of squared differences / 2n)."""
    if not pairs:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in pairs) / (2 * len(pairs)))


def evaluate(dataset, predictions, version, splits=("test",), tolerances=None, min_group_size=DEFAULT_MIN_GROUP_SIZE,
             tolerance_source=None):
    """The error report for one algorithm version on the chosen splits, as a dict."""
    tolerances = tolerances or {}
    samples = _selected_samples(dataset, splits)
    records = [predictions.get(s["sample_id"]) for s in samples]
    revisions = sorted({r["code_revision"] for r in records if r and r.get("code_revision")})

    measurements = []
    for comparison in dataset.definitions["comparisons"]:
        rows, missing = _paired(dataset, samples, predictions, comparison)
        tolerance = tolerances.get(comparison["api_name"])
        errors = [value - reference for _, value, reference in rows]
        largest = sorted(rows, key=lambda row: abs(row[1] - row[2]), reverse=True)[:5]
        measurements.append({
            **comparison,
            "n_people": len({sample["person_id"] for sample, _, _ in rows}),
            "n_with_reference": len(rows) + sum(missing.values()),
            "n_predicted": len(rows),
            "missing": dict(sorted(missing.items())),
            "tolerance_cm": tolerance,
            **error_summary(errors, tolerance),
            "largest_errors": [
                {"sample_id": s["sample_id"], "predicted": v, "reference": r, "error": v - r} for s, v, r in largest
            ],
            "groups": _groups(dataset, rows, min_group_size),
        })

    compared = {c["api_name"] for c in dataset.definitions["comparisons"]}
    returned = set()
    for record in records:
        if record and record["http_status"] == 200:
            returned.update(record["response"]["measurements"])
    pairs = dataset.repeat_pairs()

    return {
        "algorithm_version": version,
        "code_revisions": revisions,
        "dataset": os.path.basename(dataset.path),
        "definitions_version": dataset.definitions["version"],
        "splits": list(splits),
        "tolerance_source": tolerance_source,
        "n_people": len({s["person_id"] for s in samples}),
        "n_samples": len(samples),
        "requests": {
            "accepted": sum(1 for r in records if r and r["http_status"] == 200),
            "rejected": dict(sorted(Counter(r.get("error_code") or str(r["http_status"])
                                            for r in records if r and r["http_status"] != 200).items())),
            "not_predicted": sum(1 for r in records if r is None),
        },
        "measurements": measurements,
        "repeatability": [
            {"measurement": m["name"], "n_pairs": len(pairs.get(m["name"], [])),
             "tem_cm": technical_error(pairs.get(m["name"], []))}
            for m in dataset.definitions["reference_measurements"] if pairs.get(m["name"])
        ],
        "not_compared": sorted(returned - compared),
    }


def _person_bootstrap_ci(differences_by_person, n_boot, seed):
    """95% interval for the mean difference, resampling whole people so their photos stay together."""
    groups = [np.asarray(v, dtype=float) for v in differences_by_person.values()]
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
             for _ in range(n_boot)]
    low, high = np.percentile(means, [2.5, 97.5])
    return [float(low), float(high)]


def compare(dataset, base, candidate, base_version, candidate_version, splits=("test",), n_boot=2000, seed=0):
    """Paired comparison of two versions on the samples both predicted; negative change = candidate better."""
    samples = _selected_samples(dataset, splits)
    rows = []
    for comparison in dataset.definitions["comparisons"]:
        base_rows, _ = _paired(dataset, samples, base, comparison)
        cand_rows, _ = _paired(dataset, samples, candidate, comparison)
        cand_by_id = {s["sample_id"]: v for s, v, _ in cand_rows}
        by_person = defaultdict(list)
        base_abs, cand_abs = [], []
        for sample, value, reference in base_rows:
            if sample["sample_id"] in cand_by_id:
                b, c = abs(value - reference), abs(cand_by_id[sample["sample_id"]] - reference)
                base_abs.append(b)
                cand_abs.append(c)
                by_person[sample["person_id"]].append(c - b)
        diffs = [c - b for b, c in zip(base_abs, cand_abs)]
        rows.append({
            "api_name": comparison["api_name"],
            "reference": comparison["reference"],
            "base_predicted": len(base_rows),
            "candidate_predicted": len(cand_rows),
            "n_both": len(diffs),
            "n_people": len(by_person),
            "base_mae": float(np.mean(base_abs)) if diffs else None,
            "candidate_mae": float(np.mean(cand_abs)) if diffs else None,
            "mae_change": float(np.mean(diffs)) if diffs else None,
            "mae_change_ci95": _person_bootstrap_ci(by_person, n_boot, seed),
            "better": sum(1 for d in diffs if d < -0.05),
            "worse": sum(1 for d in diffs if d > 0.05),
        })
    return {
        "base_version": base_version,
        "candidate_version": candidate_version,
        "dataset": os.path.basename(dataset.path),
        "splits": list(splits),
        "measurements": rows,
    }


def _cm(value):
    return "–" if value is None else f"{value:.1f}"


def report_markdown(report):
    """The error report as Markdown."""
    requests = report["requests"]
    rejected = ", ".join(f"{code} {n}" for code, n in report["requests"]["rejected"].items()) or "none"
    lines = [
        f"# Benchmark report: algorithm {report['algorithm_version']}",
        "",
        f"- Dataset `{report['dataset']}`, definitions version {report['definitions_version']}, "
        f"splits: {', '.join(report['splits'])}",
        f"- {report['n_people']} people, {report['n_samples']} photo sets; code revision "
        f"{', '.join(report['code_revisions']) or 'unknown'}",
        f"- Requests: {requests['accepted']} accepted; rejected: {rejected}; not predicted: {requests['not_predicted']}",
    ]
    if report.get("tolerance_source"):
        lines.append(f"- Tolerances: {report['tolerance_source']}")
    lines += [
        "",
        "Error = API − tape, in cm. A positive bias means the API reads high.",
    ]
    if any(0 < m["n_people"] < FEW_PEOPLE for m in report["measurements"]):
        lines += ["", f"> Fewer than {FEW_PEOPLE} people per measurement: treat these numbers as indicative only."]

    lines += [
        "", "## Errors by measurement", "",
        "| API output | Tape measurement | Definitions | People | Predicted / with tape | MAE | Median | P90 | Max | Bias | Within tolerance |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in report["measurements"]:
        within = "–" if m["within_tolerance"] is None else f"{100 * m['within_tolerance']:.0f}% (±{m['tolerance_cm']:g})"
        lines.append(
            f"| `{m['api_name']}` | {m['reference']} | {m['definitions']} | {m['n_people']} | "
            f"{m['n_predicted']} / {m['n_with_reference']} | {_cm(m['mae'])} | {_cm(m['median_ae'])} | "
            f"{_cm(m['p90_ae'])} | {_cm(m['max_ae'])} | {_cm(m['bias'])} | {within} |"
        )

    missing = [(m["api_name"], m["missing"]) for m in report["measurements"] if m["missing"]]
    if missing:
        lines += ["", "## Missing predictions", "", "| API output | Why missing (count) |", "|---|---|"]
        lines += [f"| `{name}` | {', '.join(f'{k} {v}' for k, v in reasons.items())} |" for name, reasons in missing]

    lines += ["", "## Tape-measure repeatability", ""]
    if report["repeatability"]:
        lines += ["Technical error of measurement between the first two repeats; no method can be judged more "
                  "precisely than this.", "", "| Measurement | Pairs | TEM (cm) |", "|---|---|---|"]
        lines += [f"| {r['measurement']} | {r['n_pairs']} | {_cm(r['tem_cm'])} |" for r in report["repeatability"]]
    else:
        lines.append("No session has two repeats yet.")

    lines += ["", "## Largest errors", ""]
    for m in report["measurements"]:
        if m["largest_errors"]:
            cells = ", ".join(f"{e['sample_id']} ({e['error']:+.1f})" for e in m["largest_errors"])
            lines.append(f"- `{m['api_name']}`: {cells}")

    group_lines = []
    for m in report["measurements"]:
        for column, info in m["groups"].items():
            if info["groups"]:
                cells = "; ".join(f"{g['value']}: n {g['n']}, MAE {g['mae']:.1f}, bias {g['bias']:+.1f}" for g in info["groups"])
                hidden = f" ({info['suppressed_groups']} smaller groups not shown)" if info["suppressed_groups"] else ""
                group_lines.append(f"- `{m['api_name']}` by {column}: {cells}{hidden}")
    if group_lines:
        lines += ["", "## By group", "", "Only groups with enough photo sets are shown.", ""] + group_lines

    if report["not_compared"]:
        lines += ["", "## Not compared", "",
                  "No tape measurement is defined for: " + ", ".join(f"`{n}`" for n in report["not_compared"]) + "."]
    return "\n".join(lines) + "\n"


def comparison_markdown(result):
    """A version comparison as Markdown."""
    lines = [
        f"# Comparison: {result['candidate_version']} vs {result['base_version']}",
        "",
        f"Dataset `{result['dataset']}`, splits: {', '.join(result['splits'])}. MAE change = candidate − base "
        "on photo sets both versions predicted, in cm; negative is better. The 95% interval resamples whole people.",
        "",
        "| API output | Predicted (base / candidate) | Both | People | Base MAE | Candidate MAE | Change | 95% interval | Better / worse |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in result["measurements"]:
        ci = "–" if m["mae_change_ci95"] is None else f"{m['mae_change_ci95'][0]:+.1f} to {m['mae_change_ci95'][1]:+.1f}"
        change = "–" if m["mae_change"] is None else f"{m['mae_change']:+.1f}"
        lines.append(
            f"| `{m['api_name']}` | {m['base_predicted']} / {m['candidate_predicted']} | {m['n_both']} | "
            f"{m['n_people']} | {_cm(m['base_mae'])} | {_cm(m['candidate_mae'])} | {change} | {ci} | "
            f"{m['better']} / {m['worse']} |"
        )
    return "\n".join(lines) + "\n"
