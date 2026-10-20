"""Tests for the dataset and benchmark kit.

Every person, measurement and prediction here is synthetic, made up for the
tests; none describes a real person.
"""
import contextlib
import csv
import datetime
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from benchmark import dataset as dataset_module
from benchmark.__main__ import main
from benchmark.dataset import TABLE_FILES, Dataset, assign_splits, create_dataset
from benchmark.definitions import load_definitions
from benchmark.evaluate import compare, evaluate, report_markdown, technical_error
from benchmark.predict import load_predictions, predict, write_predictions
from benchmark.schema import data_dictionary, table_fields
from measurement.estimator import MEASUREMENT_NAMES

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = datetime.date(2026, 10, 20)


def person(person_id, **values):
    row = {"person_id": person_id, "consent_status": "granted", "consent_version": "v1",
           "consent_date": "2026-09-01", "consent_scope": "evaluation", "delete_by": "2027-09-01", "split": "test"}
    row.update(values)
    return row


def sample(sample_id, person_id, **values):
    row = {"sample_id": sample_id, "person_id": person_id, "session_id": f"V-{sample_id}",
           "capture_date": "2026-09-02", "front_image": f"{sample_id}_front.jpg", "height_cm_input": "170",
           "height_source": "self_reported", "clothing": "fitted", "footwear": "barefoot", "background": "plain",
           "lighting": "even", "capture_protocol_version": "v1"}
    row.update(values)
    return row


def reference(session_id, repeat=1, **measurements):
    row = {"session_id": session_id, "repeat": str(repeat), "measurer_id": "M1",
           "measuring_protocol_version": "v1", "measured_date": "2026-09-02"}
    row.update({f"{name}_cm": str(value) for name, value in measurements.items()})
    return row


def prediction(sample_id, status=200, code=None, **values):
    if status != 200:
        return {"sample_id": sample_id, "http_status": status, "error_code": code, "code_revision": "test",
                "response": {"error": "synthetic", "code": code}}
    measurements = {name: None for name in MEASUREMENT_NAMES}
    measurements.update(values)
    statuses = {name: {"status": "approximate" if value is not None else "unsupported"}
                for name, value in measurements.items()}
    return {"sample_id": sample_id, "http_status": 200, "error_code": None, "code_revision": "test",
            "response": {"measurements": measurements, "measurement_status": statuses}}


class DatasetCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.root = os.path.join(self.tmp, "dataset")
        self.definitions = load_definitions()

    def write(self, people=(), samples=(), references=(), images=True):
        create_dataset(self.root, self.definitions)
        rows = {"people": people, "samples": samples, "references": references}
        for table, fields in table_fields(self.definitions).items():
            with open(os.path.join(self.root, TABLE_FILES[table]), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[field.name for field in fields])
                writer.writeheader()
                writer.writerows(rows[table])
        if images:
            for row in samples:
                with open(os.path.join(self.root, "images", row["front_image"]), "wb") as f:
                    f.write(b"synthetic")
        return self.load()

    def load(self):
        return Dataset(self.root, self.definitions, today=TODAY)

    def messages(self, dataset, level="error"):
        return [f"{i.where}: {i.message}" for i in dataset.issues if i.level == level]


class ValidateTest(DatasetCase):
    def test_valid_dataset_has_no_errors(self):
        dataset = self.write([person("P1")], [sample("S1", "P1")], [reference("V-S1", shoulder_width=40)])
        self.assertEqual(self.messages(dataset), [])
        self.assertEqual([s["sample_id"] for s in dataset.usable_samples()], ["S1"])

    def test_bad_cells_name_the_line_and_column(self):
        dataset = self.write(
            [person("P1", consent_date="01/09/2026", consent_scope="anything")],
            [sample("S1", "P1")],
            [reference("V-S1", waist_circumference=850)],  # 85.0 typed as 850
        )
        errors = self.messages(dataset)
        self.assertIn("people.csv line 2: consent_date must be a date written YYYY-MM-DD", errors)
        self.assertTrue(any(e.startswith("people.csv line 2: consent_scope must be one of") for e in errors))
        self.assertIn("references.csv line 2: waist_circumference_cm must be between 45 and 180", errors)

    def test_unknown_column_is_rejected(self):
        self.write([person("P1")], [sample("S1", "P1")])
        path = os.path.join(self.root, "references.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("session_id,repeat,measurer_id,measuring_protocol_version,measured_date,waist_cm\n")
        self.assertIn("references.csv: unknown column 'waist_cm'", self.messages(self.load()))

    def test_links_between_tables_are_checked(self):
        dataset = self.write(
            [person("P1")],
            [sample("S1", "P9"), sample("S2", "P1", session_id="V-S1")],
            [reference("V-404")],
        )
        errors = self.messages(dataset)
        self.assertIn("samples.csv line 2: person_id P9 is not in people.csv", errors)
        self.assertIn("samples.csv line 3: session V-S1 already belongs to person P9", errors)
        self.assertIn("references.csv line 2: session_id V-404 has no photos in samples.csv", errors)

    def test_duplicate_ids_are_rejected(self):
        dataset = self.write([person("P1"), person("P1")], [sample("S1", "P1")])
        self.assertIn("people.csv line 3: person_id P1 appears twice", self.messages(dataset))

    def test_image_paths_cannot_leave_the_images_folder(self):
        dataset = self.write([person("P1")], [sample("S1", "P1", front_image="../people.csv")], images=False)
        self.assertIn("samples.csv line 2: front_image must stay inside the images/ folder", self.messages(dataset))

    def test_missing_image_is_reported(self):
        dataset = self.write([person("P1")], [sample("S1", "P1")], images=False)
        self.assertIn("samples.csv line 2: front_image S1_front.jpg does not exist", self.messages(dataset))

    def test_photos_of_withdrawn_people_must_be_deleted(self):
        dataset = self.write([person("P1", consent_status="withdrawn")], [sample("S1", "P1")])
        self.assertIn("samples.csv line 2: front_image must be deleted: consent withdrawn or delete_by passed",
                      self.messages(dataset))
        os.remove(os.path.join(self.root, "images", "S1_front.jpg"))
        dataset = self.load()
        self.assertEqual(self.messages(dataset), [])
        self.assertEqual(dataset.usable_samples(), [])

    def test_people_past_delete_by_are_excluded(self):
        dataset = self.write([person("P1", delete_by="2026-10-01")], [sample("S1", "P1")], images=False)
        self.assertEqual(self.messages(dataset), [])
        self.assertEqual(dataset.usable_samples(), [])

    def test_photos_before_consent_are_rejected(self):
        dataset = self.write([person("P1")], [sample("S1", "P1", capture_date="2026-08-31")])
        self.assertIn("samples.csv line 2: photos were taken before consent was given", self.messages(dataset))

    def test_disagreeing_repeats_ask_for_a_third(self):
        dataset = self.write([person("P1")], [sample("S1", "P1")],
                             [reference("V-S1", 1, hip_circumference=100), reference("V-S1", 2, hip_circumference=102)])
        self.assertTrue(any("hip_circumference repeats differ by 2.0 cm" in w
                            for w in self.messages(dataset, "warning")))

    def test_reference_value_is_mean_of_two_and_median_of_three(self):
        dataset = self.write(
            [person("P1")], [sample("S1", "P1"), sample("S2", "P1")],
            [reference("V-S1", 1, hip_circumference=100), reference("V-S1", 2, hip_circumference=101),
             reference("V-S2", 1, hip_circumference=100), reference("V-S2", 2, hip_circumference=103),
             reference("V-S2", 3, hip_circumference=101)],
        )
        values = dataset.reference_values()
        self.assertEqual(values["V-S1"]["hip_circumference"], 100.5)
        self.assertEqual(values["V-S2"]["hip_circumference"], 101)

    def test_dataset_inside_the_repository_is_flagged(self):
        with mock.patch.object(dataset_module, "REPO_DIR", self.tmp):
            dataset = self.write([person("P1")], [sample("S1", "P1")])
        self.assertTrue(any("inside the code repository" in w for w in self.messages(dataset, "warning")))

    def test_init_refuses_a_folder_that_has_files(self):
        self.write()
        with self.assertRaises(FileExistsError):
            create_dataset(self.root)


class SplitsTest(DatasetCase):
    def splits(self):
        return {pid: p["split"] for pid, p in self.load().people.items()}

    def test_new_people_get_target_shares_and_existing_splits_never_move(self):
        self.write([person(f"P{i}", split="") for i in range(10)])
        assigned = assign_splits(self.root)
        self.assertEqual(sorted(assigned.values()).count("train"), 5)
        self.assertEqual(sorted(assigned.values()).count("calibration"), 2)
        self.assertEqual(sorted(assigned.values()).count("test"), 3)

        before = self.splits()
        with open(os.path.join(self.root, "people.csv"), "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[field.name for field in table_fields(self.definitions)["people"]])
            writer.writerows(person(f"Q{i}", split="") for i in range(5))
        assign_splits(self.root)
        after = self.splits()
        self.assertEqual({pid: after[pid] for pid in before}, before)
        self.assertTrue(all(after[f"Q{i}"] for i in range(5)))

    def test_assignment_depends_only_on_the_people(self):
        self.write([person(f"P{i}", split="") for i in range(8)])
        first = assign_splits(self.root)
        shutil.rmtree(self.root)
        self.write([person(f"P{i}", split="") for i in reversed(range(8))])
        self.assertEqual(assign_splits(self.root), first)

    def test_fractions_must_add_up(self):
        self.write([person("P1", split="")])
        with self.assertRaises(ValueError):
            assign_splits(self.root, {"train": 0.5, "calibration": 0.5, "test": 0.5})


class EvaluateTest(DatasetCase):
    def setUp(self):
        super().setUp()
        self.dataset = self.write(
            [person("P1", sex="female"), person("P2", sex="female"), person("P3", sex="male"),
             person("P4", split="train")],
            [sample("S1", "P1"), sample("S2", "P2"), sample("S3", "P3"), sample("S4", "P3"), sample("S5", "P4")],
            [reference("V-S1", 1, shoulder_width=40, waist_circumference=80),
             reference("V-S1", 2, shoulder_width=41, waist_circumference=80),
             reference("V-S2", 1, shoulder_width=38, waist_circumference=70),
             reference("V-S3", 1, shoulder_width=45, waist_circumference=90),
             reference("V-S4", 1, shoulder_width=45, waist_circumference=90),
             reference("V-S5", 1, shoulder_width=50, waist_circumference=95)],
        )
        self.predictions = {
            "S1": prediction("S1", shoulder_width=41.5),  # tape mean 40.5 -> +1.0
            "S2": prediction("S2", shoulder_width=36.0),  # -2.0
            "S3": prediction("S3", shoulder_width=48.0),  # +3.0
            "S4": prediction("S4", status=400, code="INVALID_POSE"),
            "S5": prediction("S5", shoulder_width=10.0),  # train split: must not be scored
        }

    def measurement(self, report, name):
        return next(m for m in report["measurements"] if m["api_name"] == name)

    def test_error_metrics_on_the_test_split(self):
        report = evaluate(self.dataset, self.predictions, "0.2.0", tolerances={"shoulder_width": 2.0})
        shoulder = self.measurement(report, "shoulder_width")
        self.assertEqual((shoulder["n_predicted"], shoulder["n_with_reference"], shoulder["n_people"]), (3, 4, 3))
        self.assertAlmostEqual(shoulder["mae"], 2.0)
        self.assertAlmostEqual(shoulder["median_ae"], 2.0)
        self.assertAlmostEqual(shoulder["max_ae"], 3.0)
        self.assertAlmostEqual(shoulder["bias"], 2 / 3)
        self.assertAlmostEqual(shoulder["within_tolerance"], 2 / 3)
        self.assertEqual(shoulder["largest_errors"][0]["sample_id"], "S3")
        self.assertEqual(report["requests"], {"accepted": 3, "rejected": {"INVALID_POSE": 1}, "not_predicted": 0})

    def test_missing_predictions_are_counted_by_reason(self):
        report = evaluate(self.dataset, self.predictions, "0.2.0")
        self.assertEqual(self.measurement(report, "waist")["missing"], {"rejected: INVALID_POSE": 1, "unsupported": 3})
        self.assertEqual(self.measurement(report, "shoulder_width")["missing"], {"rejected: INVALID_POSE": 1})
        self.assertIsNone(self.measurement(report, "waist")["mae"])

    def test_small_groups_are_suppressed(self):
        report = evaluate(self.dataset, self.predictions, "0.2.0", min_group_size=2)
        by_sex = self.measurement(report, "shoulder_width")["groups"]["sex"]
        self.assertEqual([g["value"] for g in by_sex["groups"]], ["female"])
        self.assertEqual(by_sex["suppressed_groups"], 1)

    def test_repeatability_uses_the_first_two_repeats(self):
        report = evaluate(self.dataset, self.predictions, "0.2.0")
        shoulder = next(r for r in report["repeatability"] if r["measurement"] == "shoulder_width")
        self.assertEqual(shoulder["n_pairs"], 1)
        self.assertAlmostEqual(shoulder["tem_cm"], technical_error([(40, 41)]))
        self.assertAlmostEqual(technical_error([(100, 101), (90, 90)]), 0.5)

    def test_markdown_report_flags_small_samples(self):
        markdown = report_markdown(evaluate(self.dataset, self.predictions, "0.2.0"))
        self.assertIn("indicative only", markdown)
        self.assertIn("| `shoulder_width` | shoulder_width | differ | 3 | 3 / 4 | 2.0 |", markdown)
        self.assertIn("`chest_width`", markdown)  # returned by the API but has no tape definition

    def test_compare_pairs_samples_and_reports_a_person_level_interval(self):
        candidate = {sid: dict(record) for sid, record in self.predictions.items()}
        candidate["S1"] = prediction("S1", shoulder_width=40.5)  # error 0 instead of +1
        candidate["S2"] = prediction("S2", shoulder_width=37.0)  # error -1 instead of -2
        candidate["S3"] = prediction("S3", shoulder_width=47.0)  # error +2 instead of +3
        result = compare(self.dataset, self.predictions, candidate, "0.2.0", "0.3.0")
        shoulder = next(m for m in result["measurements"] if m["api_name"] == "shoulder_width")
        self.assertEqual((shoulder["n_both"], shoulder["better"], shoulder["worse"]), (3, 3, 0))
        self.assertAlmostEqual(shoulder["mae_change"], -1.0)
        self.assertEqual(shoulder["mae_change_ci95"], [-1.0, -1.0])


class PredictTest(DatasetCase):
    """Runs the real API on the bundled sample photos; the tape values are synthetic."""

    def test_predict_records_accepted_and_rejected_samples(self):
        dataset = self.write([person("P1")], [sample("S1", "P1", side_image="S1_side.jpg"), sample("S2", "P1")],
                             [reference("V-S1", shoulder_width=40)], images=False)
        shutil.copy(os.path.join(REPO, "front.jpeg"), os.path.join(self.root, "images", "S1_front.jpg"))
        shutil.copy(os.path.join(REPO, "left_side.jpeg"), os.path.join(self.root, "images", "S1_side.jpg"))
        with open(os.path.join(self.root, "images", "S2_front.jpg"), "wb") as f:
            f.write(b"not an image")
        dataset = self.load()
        self.assertEqual(self.messages(dataset), [])

        write_predictions(self.root, predict(dataset), "0.2.0")
        records = load_predictions(self.root, "0.2.0")
        self.assertEqual(records["S1"]["http_status"], 200)
        self.assertAlmostEqual(records["S1"]["response"]["measurements"]["shoulder_width"], 42.68, delta=0.05)
        self.assertEqual((records["S2"]["http_status"], records["S2"]["error_code"]), (400, "INVALID_IMAGE"))

        shoulder = next(m for m in evaluate(dataset, records, "0.2.0")["measurements"] if m["api_name"] == "shoulder_width")
        self.assertAlmostEqual(shoulder["mae"], 2.68, delta=0.05)


class DocsAndCliTest(DatasetCase):
    def run_cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(list(args))
        return code, out.getvalue()

    def test_data_dictionary_doc_is_up_to_date(self):
        with open(os.path.join(REPO, "docs", "data-dictionary.md"), encoding="utf-8") as f:
            self.assertEqual(f.read(), data_dictionary(self.definitions),
                             "regenerate it: python -m benchmark dictionary > docs/data-dictionary.md")

    def test_comparisons_name_real_api_outputs(self):
        for comparison in self.definitions["comparisons"]:
            self.assertIn(comparison["api_name"], MEASUREMENT_NAMES)

    def test_definitions_reject_unknown_references(self):
        broken = dict(self.definitions, comparisons=[dict(self.definitions["comparisons"][0], reference="elbow")])
        path = os.path.join(self.tmp, "broken.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(broken, f)
        with self.assertRaises(ValueError):
            load_definitions(path)

    def test_init_then_validate_an_empty_dataset(self):
        self.assertEqual(self.run_cli("init", self.root)[0], 0)
        self.assertEqual(sorted(os.listdir(self.root)), ["README.txt", "images", "people.csv", "references.csv", "samples.csv"])
        code, out = self.run_cli("validate", self.root)
        self.assertEqual(code, 0)
        self.assertIn("0 error(s)", out)


if __name__ == "__main__":
    unittest.main()
