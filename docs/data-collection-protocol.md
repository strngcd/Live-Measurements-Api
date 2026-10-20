# Data collection protocol (draft, version 1)

How to collect consented photos and tape measurements for the benchmark. Record `v1` in `capture_protocol_version` and `measuring_protocol_version`. This is a draft: review it with the merchants and a lawyer before the first participant, and bump the version whenever it changes.

## Before anyone is photographed

- [ ] A legal review of the consent form and data handling. Ethiopia adopted a personal data protection law in 2024, and diaspora participants may fall under the EU's GDPR or US state laws.
- [ ] A named data coordinator, and a written list of who may open the data.
- [ ] An encrypted disk, with the dataset created on it by `python -m benchmark init <folder>`, outside the code repository.
- [ ] Measurers trained on this protocol, with at least two practice sessions each, and a pseudonymous code for each measurer.
- [ ] A separate, locked key file that links participant codes to names and signed forms, open only to the coordinator. Names never go into the dataset.

## Consent

Use a form in the participant's own language. It must explain:

1. The purpose: testing a body-measurement system for Habeshanfit and, only if the person agrees separately, improving it (`consent_scope`).
2. What is collected: front and side photos (the face is visible, because the method uses it), tape measurements, height and, optionally, sex, age band and skin tone.
3. Where the data is stored and who can see it, and that it is never shared or sent to outside services.
4. When the photos will be deleted (`delete_by`).
5. That the person can withdraw at any time without giving a reason, and that their photos will then be deleted.

Record `consent_version`, `consent_date`, `consent_scope` and `delete_by` in `people.csv`. The pilot is for adults (18+) only.

## Participants

- 30–50 adults who resemble Habeshanfit customers: women and men, and a spread of heights, body shapes and skin tones.
- Each person has one visit (`session_id`). A second visit on another day, if possible, shows how stable the results are over time.

## Photos

**Setup**
- A plain wall that contrasts with the clothing, and even light from the front with no strong light behind the person.
- The phone on a tripod in portrait orientation, using the main (1×) camera: no zoom, portrait mode, beauty mode or filters.
- Location tagging turned off in the camera app.
- The lens at roughly the person's waist height, 2.5–3 m away. Measure and record `camera_distance_m` and `camera_height_cm`.
- The whole body in the frame, with space above the head and below the feet.

**Person**
- Fitted clothing (fitted trousers or leggings and a fitted top), hair tied back so the outline of the head shows, and barefoot. Record `clothing` and `footwear` if anything differs.
- **Front photo:** facing the camera, feet about hip-width apart, arms straight and held slightly away from the body, looking straight ahead.
- **Side photo:** turned 90° so the left side faces the camera, feet together, arms relaxed at the sides, looking straight ahead.
- Take the pair, have the person step out of the frame, then take a second pair. Give it a new `sample_id` with the same `session_id`; comparing the two shows how much results change between photos.

**After**
- Name the files by sample (`S0001_front.jpg`, `S0001_side.jpg`) and copy them into `images/`.
- Ask the person's height **before** measuring it, and enter it as `height_cm_input` with `height_source` `self_reported`, as customers will in the app. The measured height goes into `references.csv` as `stature_cm`.
- Fill in `background`, `lighting`, `device` and any `quality_flags`.

## Tape measurements

**Equipment:** a non-stretch tailor's tape marked in millimetres, a stadiometer or a wall tape with a set square for height, and a mirror or second person to check that the tape is level.

**Person:** the same fitted clothing as in the photos, barefoot, standing straight with feet together and arms relaxed, breathing normally.

**Procedure**
1. Take every measurement in the Reference measurements table of [data-dictionary.md](data-dictionary.md), in that order. Keep the tape snug without pressing in, and horizontal for circumferences.
2. Read to the nearest 0.1 cm, and height to the nearest 0.5 cm.
3. Complete a full round (`repeat` 1), then a second full round (`repeat` 2) without looking at the first values.
4. If `python -m benchmark validate` warns that two repeats differ by more than their tolerance, take a third round of that measurement (`repeat` 3); the benchmark then uses the median.
5. Measure on the same day as the photos, under the same `session_id`.

## Data entry, storage and deletion

- Enter data in the CSV files that `init` created; a spreadsheet works if it is saved as CSV (UTF-8).
- Run `python -m benchmark validate` after each session and fix every error before the next one.
- Keep the folder on the encrypted disk. Do not sync it to a cloud service without the coordinator's approval, and never email photos or send them to outside services, including AI services.
- **Withdrawal:** set `consent_status` to `withdrawn` and delete the person's photos, including backups. `validate` reports any photo that should already be gone.
- **Expiry:** delete photos when `delete_by` passes, unless the person renews consent.
