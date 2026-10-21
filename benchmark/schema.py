"""Columns of the dataset tables, how each cell is checked, and the generated data dictionary."""
import datetime
import math
import re
from collections import namedtuple

from .definitions import reference_column

Field = namedtuple("Field", "name required kind description choices low high", defaults=(None, None, None))

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
NOTES = "Free text. No names or other identifying details."

PEOPLE = (
    Field("person_id", True, "id", "Pseudonymous participant code, e.g. P0001. Never a name, phone number or email."),
    Field("consent_status", True, "choice",
          "Current consent. People who withdraw are excluded and their photos must be deleted.",
          ("granted", "withdrawn")),
    Field("consent_version", True, "text", "Version of the consent form the person signed."),
    Field("consent_date", True, "date", "Date consent was given."),
    Field("consent_scope", True, "choice", "What the person's data may be used for.",
          ("evaluation", "evaluation_and_training")),
    Field("delete_by", True, "date", "Date by which the person's photos must be deleted."),
    Field("split", False, "choice", "Assigned once by `python -m benchmark splits`; never edit it by hand.",
          ("train", "calibration", "test")),
    Field("sex", False, "choice", "Optional; used only to compare errors between groups.",
          ("female", "male", "other", "undisclosed")),
    Field("age_band", False, "choice", "Optional; used only to compare errors between groups. Adults only.",
          ("18-24", "25-34", "35-44", "45-54", "55-64", "65+")),
    Field("skin_tone", False, "integer",
          "Optional Monk Skin Tone scale value, 1 (lightest) to 10 (darkest), to check errors across skin tones.",
          None, 1, 10),
    Field("notes", False, "text", NOTES),
)

SAMPLES = (
    Field("sample_id", True, "id", "Code for one set of photos, e.g. S0001."),
    Field("person_id", True, "id", "The person in the photos; must exist in people.csv."),
    Field("session_id", True, "id", "The visit. Photos and tape measurements from the same visit share it."),
    Field("capture_date", True, "date", "Date the photos were taken."),
    Field("front_image", True, "path", "Front photo, relative to the dataset's images/ folder."),
    Field("side_image", False, "path", "Side photo, relative to the dataset's images/ folder."),
    Field("height_cm_input", True, "number", "Height sent to the API with these photos, in cm.", None, 50, 250),
    Field("height_source", True, "choice", "Where height_cm_input came from.", ("self_reported", "measured")),
    Field("device", False, "text", "Phone or camera model."),
    Field("camera_distance_m", False, "number", "Distance from the camera to the person, in metres.", None, 0.5, 10),
    Field("camera_height_cm", False, "number", "Height of the camera lens above the floor, in cm.", None, 20, 250),
    Field("clothing", True, "choice", "What the person wore in the photos.", ("fitted", "loose", "traditional_garment")),
    Field("footwear", True, "choice", "What was on the person's feet.", ("barefoot", "socks", "shoes")),
    Field("background", True, "choice", "Background behind the person.", ("plain", "cluttered")),
    Field("lighting", True, "choice", "Lighting on the person.", ("even", "uneven", "dim")),
    Field("capture_protocol_version", True, "text", "Version of the photo protocol followed."),
    Field("quality_flags", False, "flags", "Problems seen in the photos.",
          ("blur", "cropped", "occluded", "wrong_pose", "heavy_shadow", "other")),
    Field("notes", False, "text", NOTES),
)

REFERENCE_KEYS = (
    Field("session_id", True, "id", "The visit; must match a session_id in samples.csv."),
    Field("repeat", True, "integer", "1 for the first full round of measurements, 2 for the second, and so on.",
          None, 1, 9),
    Field("measurer_id", True, "id", "Pseudonymous code of the person who took the measurements."),
    Field("measuring_protocol_version", True, "text", "Version of the tape-measuring protocol followed."),
    Field("measured_date", True, "date", "Date of the measurements."),
)


def reference_fields(definitions):
    columns = tuple(
        Field(reference_column(m["name"]), False, "number", f"{m['description']} Blank if not taken.",
              None, *m["plausible_cm"])
        for m in definitions["reference_measurements"]
    )
    return REFERENCE_KEYS + columns + (Field("notes", False, "text", NOTES),)


def table_fields(definitions):
    return {"people": PEOPLE, "samples": SAMPLES, "references": reference_fields(definitions)}


def parse_value(field, raw):
    """Parse one cell. Returns (value, problem); a blank cell gives value None."""
    text = (raw or "").strip()
    if not text:
        return None, ("is required" if field.required else None)
    if field.kind == "id":
        if ID_PATTERN.match(text):
            return text, None
        return None, "may only use letters, digits, - and _ (at most 40)"
    if field.kind in ("text", "path"):
        return text, None
    if field.kind == "date":
        try:
            return datetime.date.fromisoformat(text), None
        except ValueError:
            return None, "must be a date written YYYY-MM-DD"
    if field.kind == "choice":
        if text in field.choices:
            return text, None
        return None, f"must be one of: {', '.join(field.choices)}"
    if field.kind == "flags":
        flags = [flag.strip() for flag in text.split(";") if flag.strip()]
        unknown = [flag for flag in flags if flag not in field.choices]
        if unknown:
            return None, f"has unknown flags {', '.join(unknown)}; allowed: {', '.join(field.choices)}"
        return flags, None

    try:
        value = float(text) if field.kind == "number" else int(text)
    except ValueError:
        return None, "must be a number" if field.kind == "number" else "must be a whole number"
    if not math.isfinite(value) or not field.low <= value <= field.high:
        return None, f"must be between {field.low:g} and {field.high:g}"
    return value, None


def _format(field):
    if field.kind == "choice":
        return "one of: " + ", ".join(f"`{c}`" for c in field.choices)
    if field.kind == "flags":
        return "any of, separated by `;`: " + ", ".join(f"`{c}`" for c in field.choices)
    if field.kind in ("number", "integer"):
        return f"{'number' if field.kind == 'number' else 'whole number'}, {field.low:g} to {field.high:g}"
    return {"id": "code", "text": "text", "path": "file path", "date": "date, YYYY-MM-DD"}[field.kind]


def _garment_table(garment):
    if not garment["measurements"]:
        return ["No body measurements."]
    lines = ["| Measurement | Priority | Source | Tolerance | Ease | Why |", "|---|---|---|---|---|---|"]
    for m in garment["measurements"]:
        tolerance = "to confirm" if m["tolerance_cm"] is None else f"±{m['tolerance_cm']:g} cm"
        ease = "–" if m["ease_cm"] is None else f"{m['ease_cm']:g} cm"
        lines.append(f"| {m['reference']} | {m['priority']} | {m['source']} | {tolerance} | {ease} | {m.get('why', '')} |")
    return lines


def data_dictionary(definitions, garments):
    """The data dictionary as Markdown (docs/data-dictionary.md is generated from this)."""
    lines = [
        "# Data dictionary",
        "",
        "Generated by `python -m benchmark dictionary` from `benchmark/schema.py`, "
        f"`benchmark/definitions.json` (definitions version {definitions['version']}) and "
        f"`benchmark/garments.json` (garments version {garments['version']}); do not edit by hand.",
        "",
        "All lengths are in centimetres. Codes are pseudonymous: never put names, phone numbers or emails in any table.",
    ]
    for table, fields in table_fields(definitions).items():
        lines += ["", f"## {table}.csv", "", "| Column | Required | Format | Description |", "|---|---|---|---|"]
        lines += [
            f"| `{f.name}` | {'yes' if f.required else 'no'} | {_format(f)} | {f.description} |" for f in fields
        ]

    lines += ["", "## Reference measurements", "", definitions["note"], "",
              "| Measurement | Column | How to take it | Closest ISO 8559-1 | Repeats may differ by |",
              "|---|---|---|---|---|"]
    lines += [
        f"| {m['name']} | `{reference_column(m['name'])}` | {m['description']} | {m.get('iso_8559_1') or '–'} | "
        f"{m['repeat_tolerance_cm']:g} cm |"
        for m in definitions["reference_measurements"]
    ]

    lines += ["", "## Measurements by garment", "", garments["note"]]
    for garment in garments["garments"]:
        lines += ["", f"### {garment['label']}", "",
                  f"Worn by: {garment['worn_by']}. Fit: {garment['fit'].replace('_', '-')}.", ""]
        lines += _garment_table(garment)

    lines += ["", "## API outputs compared with tape measurements", "",
              "| API output | Compared with | Definitions | Note |", "|---|---|---|---|"]
    lines += [
        f"| `{c['api_name']}` | {c['reference']} | {c['definitions']} | {c['note']} |"
        for c in definitions["comparisons"]
    ]
    return "\n".join(lines) + "\n"
