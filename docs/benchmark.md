# Benchmark kit

The kit scores the API against tape measurements of consented people, one measurement at a time, on people the method was never tuned on. It needs no network and stores everything in a dataset folder that you keep outside this repository.

## Workflow

1. Create an empty dataset folder on an encrypted disk, outside the repository:

   ```bash
   python -m benchmark init /secure/habeshanfit-pilot
   ```

2. Collect photos and tape measurements as described in [data-collection-protocol.md](data-collection-protocol.md). Fill in `people.csv`, `samples.csv` and `references.csv` (columns in [data-dictionary.md](data-dictionary.md)) and put the photos in `images/`.
3. Check the folder after every session and fix every error:

   ```bash
   python -m benchmark validate /secure/habeshanfit-pilot
   ```

4. Assign new people to a split. Each person is assigned once, by person rather than by photo, with default target shares of 50% train, 20% calibration and 30% test; existing assignments never change.

   ```bash
   python -m benchmark splits /secure/habeshanfit-pilot
   ```

5. Run the current code on every sample with valid consent. This writes `predictions/<algorithm version>.jsonl`:

   ```bash
   python -m benchmark predict /secure/habeshanfit-pilot
   ```

6. Score one version, optionally against a garment's tolerances (see [Tolerances](#tolerances)). This writes `reports/<version>-test.md` and `.json`:

   ```bash
   python -m benchmark evaluate /secure/habeshanfit-pilot --version 0.2.0 --garment habesha_kemis
   ```

7. After changing the method: bump `ALGORITHM_VERSION` in `measurement/__init__.py`, commit, run `predict` again, then compare the two versions on the same people:

   ```bash
   python -m benchmark compare /secure/habeshanfit-pilot --base 0.2.0 --candidate 0.3.0
   ```

## Rules

- **Tune on train, choose on calibration, report on test.** Nothing may be adjusted after looking at test results; if that happens, the test set is spent and new test people are needed.
- **Compare like with like.** Compare versions only on the same split of the same dataset. Claim an improvement only when the 95% interval of the MAE change is entirely below zero, and say how many people it rests on.
- **Commit before predicting.** Each prediction records the code revision; uncommitted changes are marked `-dirty`, and the `predict` command warns about them.
- **Keep truth and predictions apart.** Tape measurements live only in `references.csv`; predictions live only in `predictions/` and are never edited by hand.
- **Change definitions deliberately.** Editing `benchmark/definitions.json` or `benchmark/garments.json` means bumping that file's `version` and regenerating the data dictionary with `python -m benchmark dictionary > docs/data-dictionary.md`.

## What the report shows

| Section | Contents |
|---|---|
| Errors by measurement | For each API output with a tape definition: people, photo sets predicted out of those with tape values, and error statistics in cm: mean absolute error (MAE), median, 90th percentile, maximum, and bias (positive means the API reads high), plus the share within tolerance when a garment or tolerances file is given. |
| Missing predictions | Why the rest have no number: the request was rejected (with its error code), or the measurement is `unsupported` or `unavailable`. |
| Tape-measure repeatability | Technical error of measurement between the first two tape repeats. No method can be judged more precisely than this. |
| Largest errors | The five photo sets with the largest errors per measurement, to inspect by hand. |
| By group | Errors split by sex, age band, skin tone, clothing, footwear, background, lighting, height source and device. Groups smaller than 5 photo sets (`--min-group-size`) are hidden. |
| Not compared | API outputs with no tape definition, such as the width estimates. |

With fewer than 30 people for a measurement, the report marks its numbers as indicative only.

## Tolerances

Score against one garment's tolerances, as the merchants gave them in `benchmark/garments.json`. The report is then named after the garment, e.g. `reports/0.2.0-test-habesha_kemis.md`:

```bash
python -m benchmark evaluate /secure/habeshanfit-pilot --version 0.2.0 --garment habesha_kemis
```

Garments are `habesha_kemis`, `mens_shirt`, `mens_trousers` and `netela_gabi`. Measurements whose tolerance is still "to confirm" are scored without one. To try other values, pass `--tolerances` with a JSON object mapping API output names to the largest acceptable error in cm:

```json
{"shoulder_width": 1.0, "arm_length": 1.5, "waist": 2.0}
```

## Definitions

`benchmark/definitions.json` defines each tape measurement and which API output is compared with it. Its `definitions` field says whether the API intends the same measurement (`same`) or something different (`differ`). For example, `trouser_length` runs from the hip joint to the ankle, while the tape's `trouser_waist_to_ankle_length` starts at the top of the hip bone, so a constant gap is expected until the API changes.

`benchmark/garments.json` lists the tape measurements each garment needs. Each one has a priority (`must` or `nice`), a tolerance and ease in cm, and a source. The source is `merchant` when a merchant ticked it on the requirements form, and `standard` when it was added from usual tailoring practice and still needs a merchant to confirm it. Both files are summarised in [data-dictionary.md](data-dictionary.md).
