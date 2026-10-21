"""What each garment needs: tape measurements with their tolerance, ease and source (merchant form or usual practice)."""
import json
import math
import os

from .definitions import load_definitions

GARMENTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "garments.json")
SOURCES = ("merchant", "standard")
PRIORITIES = ("must", "nice")


def load_garments(definitions=None, path=GARMENTS_PATH):
    definitions = definitions or load_definitions()
    with open(path, encoding="utf-8") as f:
        garments = json.load(f)
    known = {m["name"] for m in definitions["reference_measurements"]}
    names = [g["name"] for g in garments["garments"]]
    if len(names) != len(set(names)):
        raise ValueError("garments repeat a garment name")
    for garment in garments["garments"]:
        references = [m["reference"] for m in garment["measurements"]]
        if len(references) != len(set(references)):
            raise ValueError(f"{garment['name']} lists a measurement twice")
        for m in garment["measurements"]:
            where = f"{garment['name']}: {m['reference']}"
            if m["reference"] not in known:
                raise ValueError(f"{where} is not a reference measurement")
            if m["source"] not in SOURCES:
                raise ValueError(f"{where}: 'source' must be one of {', '.join(SOURCES)}")
            if m["priority"] not in PRIORITIES:
                raise ValueError(f"{where}: 'priority' must be one of {', '.join(PRIORITIES)}")
            for key in ("tolerance_cm", "ease_cm"):
                value = m[key]
                if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                    raise ValueError(f"{where}: '{key}' must be a number of cm, 0 or more, or null")
    return garments


def garment_tolerances(garments, definitions, name):
    """Tolerances for `evaluate`, keyed by API output name, from the garment's measurements that have one."""
    garment = next((g for g in garments["garments"] if g["name"] == name), None)
    if garment is None:
        raise ValueError(f"unknown garment {name!r}; known: {', '.join(g['name'] for g in garments['garments'])}")
    by_reference = {m["reference"]: m["tolerance_cm"] for m in garment["measurements"] if m["tolerance_cm"] is not None}
    return {c["api_name"]: by_reference[c["reference"]] for c in definitions["comparisons"] if c["reference"] in by_reference}
