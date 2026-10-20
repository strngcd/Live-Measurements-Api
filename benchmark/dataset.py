"""A dataset folder: people.csv, samples.csv, references.csv and the images/ folder."""
import csv
import datetime
import hashlib
import os
import statistics
from collections import Counter, defaultdict, namedtuple

from .definitions import load_definitions, reference_column
from .schema import parse_value, table_fields

TABLE_FILES = {"people": "people.csv", "samples": "samples.csv", "references": "references.csv"}
SPLITS = ("train", "calibration", "test")
DEFAULT_SPLIT_FRACTIONS = {"train": 0.5, "calibration": 0.2, "test": 0.3}
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRIVACY_NOTE = """\
This folder holds personal data: body photos and body measurements.

- Keep it on an encrypted disk, outside the code repository.
- Only the people named in the data plan may open it.
- Do not upload it, sync it to a cloud service or send photos to any external service.
- Delete a person's photos when they withdraw consent or their delete_by date passes;
  `python -m benchmark validate` reports photos that should already be gone.

See docs/data-collection-protocol.md in the code repository.
"""

Issue = namedtuple("Issue", "level where message")


def create_dataset(path, definitions=None):
    """Create an empty dataset folder: header-only tables, images/ and a privacy note."""
    definitions = definitions or load_definitions()
    if os.path.exists(path) and os.listdir(path):
        raise FileExistsError(f"{path} already exists and is not empty")
    os.makedirs(os.path.join(path, "images"), exist_ok=True)
    for table, fields in table_fields(definitions).items():
        with open(os.path.join(path, TABLE_FILES[table]), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([field.name for field in fields])
    with open(os.path.join(path, "README.txt"), "w", encoding="utf-8") as f:
        f.write(PRIVACY_NOTE)


class Dataset:
    """A parsed dataset folder. Cells hold parsed values (None when blank or invalid); problems go to `issues`."""

    def __init__(self, path, definitions=None, today=None):
        self.path = os.path.abspath(path)
        self.definitions = definitions or load_definitions()
        self.today = today or datetime.date.today()
        self.issues = []
        self.fields = table_fields(self.definitions)

        if os.path.commonpath([self.path, REPO_DIR]) == REPO_DIR:
            self.warning(self.path, "the dataset is inside the code repository; keep personal data outside it")
        tables = {name: self._read_table(name) for name in TABLE_FILES}
        self.people = self._index(tables["people"], "person_id")
        self.samples = self._index(tables["samples"], "sample_id")
        self.references = tables["references"]

        self._check_consent()
        self._check_links()
        self._check_images()
        self._check_repeats()

    @property
    def errors(self):
        return [issue for issue in self.issues if issue.level == "error"]

    def error(self, where, message):
        self.issues.append(Issue("error", where, message))

    def warning(self, where, message):
        self.issues.append(Issue("warning", where, message))

    def _read_table(self, name):
        filename = TABLE_FILES[name]
        path = os.path.join(self.path, filename)
        if not os.path.exists(path):
            self.error(filename, "file is missing")
            return []
        fields = self.fields[name]
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            known = {field.name for field in fields}
            for column in header:
                if column not in known:
                    self.error(filename, f"unknown column {column!r}")
            for field in fields:
                if field.required and field.name not in header:
                    self.error(filename, f"required column {field.name!r} is missing")

            rows = []
            for line, raw in enumerate(reader, start=2):
                row = {"_where": f"{filename} line {line}"}
                if None in raw:
                    self.error(row["_where"], "has more cells than the header")
                for field in fields:
                    value, problem = parse_value(field, raw.get(field.name))
                    if problem and field.name in header:
                        self.error(row["_where"], f"{field.name} {problem}")
                    row[field.name] = value
                rows.append(row)
        return rows

    def _index(self, rows, key):
        index = {}
        for row in rows:
            if row[key] is None:
                continue
            if row[key] in index:
                self.error(row["_where"], f"{key} {row[key]} appears twice")
                continue
            index[row[key]] = row
        return index

    def must_delete_photos(self, person):
        """True once a person's photos may no longer be kept."""
        return person["consent_status"] == "withdrawn" or (
            person["delete_by"] is not None and person["delete_by"] < self.today
        )

    def is_usable(self, person):
        return (
            person is not None
            and person["consent_status"] == "granted"
            and person["delete_by"] is not None
            and not self.must_delete_photos(person)
        )

    def _check_consent(self):
        for person in self.people.values():
            consent_date, delete_by = person["consent_date"], person["delete_by"]
            if consent_date and consent_date > self.today:
                self.error(person["_where"], "consent_date is in the future")
            if consent_date and delete_by and delete_by < consent_date:
                self.error(person["_where"], "delete_by is before consent_date")
            if self.is_usable(person) and person["split"] is None:
                self.warning(person["_where"], "no split yet; run `python -m benchmark splits`")

    def _check_links(self):
        self.session_person = {}
        for sample in self.samples.values():
            person = self.people.get(sample["person_id"])
            if sample["person_id"] and person is None:
                self.error(sample["_where"], f"person_id {sample['person_id']} is not in people.csv")
            if person and person["consent_date"] and sample["capture_date"] and sample["capture_date"] < person["consent_date"]:
                self.error(sample["_where"], "photos were taken before consent was given")
            if sample["session_id"]:
                owner = self.session_person.setdefault(sample["session_id"], sample["person_id"])
                if owner != sample["person_id"]:
                    self.error(sample["_where"], f"session {sample['session_id']} already belongs to person {owner}")

        seen = set()
        for ref in self.references:
            session_id = ref["session_id"]
            if session_id and session_id not in self.session_person:
                self.error(ref["_where"], f"session_id {session_id} has no photos in samples.csv")
            if session_id and ref["repeat"] is not None:
                if (session_id, ref["repeat"]) in seen:
                    self.error(ref["_where"], f"repeat {ref['repeat']} of session {session_id} appears twice")
                seen.add((session_id, ref["repeat"]))

        measured = {ref["session_id"] for ref in self.references}
        for session_id in self.session_person:
            if session_id not in measured:
                self.warning("samples.csv", f"session {session_id} has no tape measurements yet")

    def image_path(self, relative):
        """Absolute path of an image, or None if it would point outside images/."""
        images = os.path.realpath(os.path.join(self.path, "images"))
        full = os.path.realpath(os.path.join(images, relative))
        return full if os.path.commonpath([full, images]) == images else None

    def _check_images(self):
        for sample in self.samples.values():
            person = self.people.get(sample["person_id"])
            for column in ("front_image", "side_image"):
                if sample[column] is None:
                    continue
                full = self.image_path(sample[column])
                if full is None:
                    self.error(sample["_where"], f"{column} must stay inside the images/ folder")
                    sample[column] = None
                elif person and self.must_delete_photos(person):
                    if os.path.exists(full):
                        self.error(sample["_where"], f"{column} must be deleted: consent withdrawn or delete_by passed")
                elif not os.path.isfile(full):
                    self.error(sample["_where"], f"{column} {sample[column]} does not exist")

    def _check_repeats(self):
        for session_id, values in self._repeat_values().items():
            for m in self.definitions["reference_measurements"]:
                taken = values.get(m["name"], [])
                if len(taken) == 2 and abs(taken[0] - taken[1]) > m["repeat_tolerance_cm"]:
                    self.warning(
                        f"references.csv session {session_id}",
                        f"{m['name']} repeats differ by {abs(taken[0] - taken[1]):.1f} cm "
                        f"(more than {m['repeat_tolerance_cm']:g}); take a third measurement",
                    )

    def _repeat_values(self):
        """{session_id: {measurement: [values in repeat order]}}"""
        values = defaultdict(lambda: defaultdict(list))
        for ref in sorted(self.references, key=lambda r: (r["session_id"] or "", r["repeat"] or 0)):
            for m in self.definitions["reference_measurements"]:
                value = ref[reference_column(m["name"])]
                if ref["session_id"] and value is not None:
                    values[ref["session_id"]][m["name"]].append(value)
        return values

    def usable_samples(self):
        """Samples whose person has given consent that is still in force, sorted by sample_id."""
        return [s for _, s in sorted(self.samples.items()) if self.is_usable(self.people.get(s["person_id"]))]

    def reference_values(self):
        """{session_id: {measurement: value}}: mean of one or two repeats, median of three or more."""
        result = {}
        for session_id, values in self._repeat_values().items():
            result[session_id] = {
                name: statistics.median(taken) if len(taken) >= 3 else statistics.fmean(taken)
                for name, taken in values.items()
            }
        return result

    def repeat_pairs(self):
        """{measurement: [(first, second), ...]} from the first two repeats of usable sessions."""
        usable = {s["session_id"] for s in self.usable_samples()}
        pairs = defaultdict(list)
        for session_id, values in self._repeat_values().items():
            if session_id in usable:
                for name, taken in values.items():
                    if len(taken) >= 2:
                        pairs[name].append((taken[0], taken[1]))
        return pairs


def assign_splits(path, fractions=None):
    """Give every person without a split one, never changing existing splits.

    New people are taken in an order fixed by a hash of their id and each goes to
    the split furthest below its target share, so the shares stay close to
    `fractions` and nobody moves between runs. Returns {person_id: split} for the
    people assigned now.
    """
    fractions = fractions or DEFAULT_SPLIT_FRACTIONS
    if set(fractions) != set(SPLITS) or min(fractions.values()) < 0 or abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError(f"fractions must cover {', '.join(SPLITS)} and add up to 1")

    filename = os.path.join(path, TABLE_FILES["people"])
    with open(filename, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    if "split" not in header:
        header.append("split")

    current = [(row.get("split") or "").strip() for row in rows]
    counts = Counter({split: 0 for split in SPLITS})
    counts.update(split for split in current if split in SPLITS)
    new_people = sorted(
        {(row.get("person_id") or "").strip() for row, split in zip(rows, current) if not split} - {""},
        key=lambda person_id: hashlib.sha256(person_id.encode("utf-8")).hexdigest(),
    )

    assigned = {}
    for person_id in new_people:
        total = sum(counts.values()) + 1
        split = max(SPLITS, key=lambda s: fractions[s] * total - counts[s])
        counts[split] += 1
        assigned[person_id] = split
    for row in rows:
        person_id = (row.get("person_id") or "").strip()
        if person_id in assigned:
            row["split"] = assigned[person_id]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return assigned
