# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Experimental photo-based body measurement API for Habeshanfit, a marketplace for traditional Habesha clothing. A Flask service estimates measurements from a front photo plus the person's height. A separate benchmark kit scores those estimates against tape measurements of consented people. No measurement has been validated yet.

## Commands

Use the virtualenv at `.venv/` (Python 3.12), e.g. `.venv/bin/python`.

```bash
pip install -r requirements.txt   # the file is UTF-16 with a BOM; keep that encoding when editing it
python app.py                     # API on http://0.0.0.0:8001, POST /upload_images
python -m unittest -v             # full suite from the repo root; no network needed
python -m unittest tests.test_benchmark.EvaluateTest.test_evaluate_with_a_garments_tolerances   # one test
python -m benchmark --help        # init | validate | splits | predict | evaluate | compare | dictionary
python -m benchmark dictionary > docs/data-dictionary.md   # regenerate after editing benchmark/*.json or schema.py
```

There is no linter or formatter configured.

## Architecture

### Request path

The request path is `app.py` → `measurement/inputs.py` → `measurement/landmarks.py` → `measurement/estimator.py`.

- **Input errors:** `inputs.py` raises `InputError(code, message, field)`, and `app.py` turns it into a JSON 400. Every error body has `error` and `code`.
- **Pose detection:** `detect_pose` builds a new static-mode MediaPipe Holistic graph for every photo. Never share or reuse a detector across photos or requests. The original prototype did, and leaked tracking state made identical requests return different values.
- **Estimates:** `estimate_measurements` always returns every name in `MEASUREMENT_NAMES` (14, in the original order). Each name gets a status:
  - `approximate`: a number, computed from `APPROXIMATE_MEASUREMENTS`.
  - `unavailable`: null, because the landmarks it needs were unreliable in this photo.
  - `unsupported`: null, with the reason taken from `UNSUPPORTED_REASONS`.
  - Never return a fabricated number for a measurement the method can't support.
- **Scale:** pixels become cm by treating the nose-to-lowest-ankle span as the person's height. This is a known overestimate, reported in the `CALIBRATION_BIAS` warning.
- **Unused components:** `measurement/depth.py` (MiDaS, loaded through `torch.hub`) and the width/ellipse functions in `geometry.py` are kept and tested, but the API does not call them. The side photo is validated but not used.
- **Response compatibility:** response keys from the original API are kept. `debug_info.focal_length` is a constant.
- **Versioning:** bump `ALGORITHM_VERSION` in `measurement/__init__.py` whenever a returned value, status or validation rule changes. Benchmark predictions are stored per version.

### Benchmark kit (`benchmark/`, guide in `docs/benchmark.md`)

- **Dataset location:** a dataset is a folder of CSV tables (`people`, `samples`, `references`) plus `images/`. It holds personal data, so it lives outside the repository. `/data/` is gitignored, and `Dataset` warns when a dataset is inside the repo.
- **`schema.py`:** defines the columns of the `people` and `samples` tables.
- **`definitions.json`:** the `references` columns (`<name>_cm`) are generated from its `reference_measurements`. Its `comparisons` map each API output to a tape measurement, marked `same` or `differ`.
- **`garments.json`:** lists the tape measurements each garment needs, with priority, tolerance, ease and source:
  - `merchant`: from the merchants' requirements forms.
  - `standard`: from usual tailoring practice, still to be confirmed.
  - `evaluate --garment <name>` scores against one garment's tolerances.
- **Editing either JSON file:** bump its `version` and regenerate `docs/data-dictionary.md`. A test fails if the doc is stale.
- **`predict`:** runs the real API in-process through Flask's test client. It records the git revision, with a `-dirty` suffix when there are uncommitted changes.
- **Splits:** splits are assigned per person, once, and never move. Tune on train, choose on calibration, report on test.

## Tests

- **Synthetic landmarks:** `tests/helpers.make_landmarks` builds 33 MediaPipe-style landmarks for an upright front pose. `moved` overrides positions, and `hidden` lowers visibility.
- **Integration test:** `tests/test_integration.py` runs real MediaPipe on `front.jpeg` and `left_side.jpeg`. `UNCONTAMINATED_BASELINE` pins the current arithmetic, not correct sizes. Update it deliberately when the method changes.
- **Depth test:** `tests/test_depth.py` uses a fake model and is skipped when torch is missing.
- **Benchmark tests:** all people and measurements in `tests/test_benchmark.py` are synthetic. Never add real photos or tape values to the repository.
