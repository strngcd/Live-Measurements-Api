"""Reference (tape) measurement definitions and which API output is compared with each."""
import json
import os

DEFINITIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "definitions.json")


def load_definitions(path=DEFINITIONS_PATH):
    with open(path, encoding="utf-8") as f:
        definitions = json.load(f)
    names = [m["name"] for m in definitions["reference_measurements"]]
    if len(names) != len(set(names)):
        raise ValueError("definitions repeat a reference measurement name")
    for comparison in definitions["comparisons"]:
        if comparison["reference"] not in names:
            raise ValueError(f"{comparison['api_name']} is compared with unknown measurement {comparison['reference']}")
        if comparison["definitions"] not in ("same", "differ"):
            raise ValueError(f"{comparison['api_name']}: 'definitions' must be 'same' or 'differ'")
    return definitions


def reference_column(name):
    """Column holding a reference measurement in references.csv; the unit is part of the name."""
    return f"{name}_cm"
