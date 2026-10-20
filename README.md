# Photo-Based Body Measurement API (experimental)

Estimates body measurements from a front photo and the person's height, using **Flask**, **OpenCV** and **MediaPipe**. Based on [JavTahir/Live-Measurements-Api](https://github.com/JavTahir/Live-Measurements-Api).

> ⚠️ **Status: experimental.** No measurement has been validated against tape measurements, so the output must not be used for tailoring yet. Most measurements are currently returned as `null` with status `unsupported`, because the original method could not estimate them (see [Measurements](#measurements)).

---

## How it works (version 0.2.0)

1. Validates the inputs: height, image files and size limits.
2. Detects 33 pose landmarks on the front photo with MediaPipe Holistic in static-image mode: one detection per photo, with no state shared between photos or requests.
3. Checks that every landmark a measurement needs is visible and inside the photo. Photos that don't show the whole body, including the feet, are rejected.
4. Converts pixels to centimetres by treating the nose-to-lowest-ankle span as the person's height. That span is shorter than full height, so lengths are overestimated. This known bias is reported in the `CALIBRATION_BIAS` warning and is not yet corrected.
5. Returns the measurements it can estimate, and `null` with a status and reason for the rest.

The side photo is accepted and checked, but no measurement uses it yet. MiDaS depth (`measurement/depth.py`), the brightness-threshold width scan and the ellipse circumference (`measurement/geometry.py`) are kept as tested functions for later comparison, but the API does not run them.

| Path | Purpose |
|---|---|
| `app.py` | Flask app: request handling and response format |
| `measurement/inputs.py` | Height and image validation |
| `measurement/landmarks.py` | Pose detection and landmark reliability checks |
| `measurement/estimator.py` | Measurements and their per-measurement status |
| `measurement/geometry.py` | Pixel scale, silhouette width, ellipse circumference |
| `measurement/depth.py` | MiDaS depth (not used by the API) |
| `tests/` | `unittest` suite |

## How to run

```bash
pip install -r requirements.txt
python app.py              # serves on http://0.0.0.0:8001
python -m unittest -v      # runs the tests; no network needed
```

The API runs on CPU and makes no network calls; photos are processed in memory and are neither stored nor logged.

---

## API

### `POST /upload_images`

Send `multipart/form-data` with:

| Field | Required | Notes |
|---|---|---|
| `front` | yes | JPEG or PNG photo from the front, whole body including feet visible. At least 320 px on the shorter side, at most 50 megapixels. |
| `left_side` | no | Same format rules. Checked, but not used by any measurement yet. |
| `height_cm` | yes | The person's height in centimetres, from 50 to 250. |

The whole upload may be at most 32 MB.

```bash
curl -X POST http://localhost:8001/upload_images \
  -F "front=@front.jpeg" \
  -F "left_side=@left_side.jpeg" \
  -F "height_cm=170"
```

### Success (200)

Output for the bundled sample photos, shortened:

```json
{
  "algorithm_version": "0.2.0",
  "units": "cm",
  "measurements": {
    "shoulder_width": 42.68,
    "arm_length": 59.68,
    "trouser_length": 84.81,
    "chest_width": null,
    "waist": null,
    "hip": null
  },
  "measurement_status": {
    "shoulder_width": {"status": "approximate", "reason": "Shoulder-joint landmark span x 1.1 (unvalidated correction); ..."},
    "waist": {"status": "unsupported", "reason": "Needs measured body depth. ..."}
  },
  "warnings": [
    {"code": "UNVALIDATED_ESTIMATES", "message": "..."},
    {"code": "CALIBRATION_BIAS", "message": "..."},
    {"code": "SIDE_IMAGE_UNUSED", "message": "..."}
  ],
  "debug_info": {"scale_factor": 0.1756, "focal_length": 600.0, "user_height_cm": 170.0}
}
```

`measurements` and `measurement_status` always contain all 14 measurement names listed below. Each status is one of:

| `status` | Value | Meaning |
|---|---|---|
| `approximate` | number | Estimated from landmarks; not validated against tape measurements. |
| `unavailable` | `null` | Supported, but the landmarks it needs were not reliable in this photo; `missing_landmarks` names them. |
| `unsupported` | `null` | The current method cannot estimate this measurement. |

`debug_info.focal_length` is a constant kept for compatibility; it carries no information.

### Errors

Every error is JSON with `error` (a message for people) and `code` (for programs):

| HTTP | `code` | When | Extra fields |
|---|---|---|---|
| 400 | `MISSING_FRONT_IMAGE` | No `front` file | `field` |
| 400 | `MISSING_HEIGHT` | `height_cm` missing or blank | `field` |
| 400 | `INVALID_HEIGHT` | Not a number, not finite, or outside 50–250 | `field` |
| 400 | `INVALID_IMAGE` | Empty or unreadable file | `field` |
| 400 | `IMAGE_TOO_SMALL`, `IMAGE_TOO_LARGE` | Outside the size limits above | `field` |
| 400 | `INVALID_POSE` | No person found, body or feet not fully visible, or standing too close | `pose`, `missing_landmarks` |
| 413 | `PAYLOAD_TOO_LARGE` | Upload larger than 32 MB | |
| 500 | `INTERNAL_ERROR` | Unexpected failure | |

---

## Measurements

| Name | Status in 0.2.0 | Method, or why it is unsupported |
|---|---|---|
| `shoulder_width` | approximate | Span between the shoulder-joint landmarks × 1.1 (unvalidated correction) |
| `arm_length` | approximate | Vertical distance from left shoulder to left wrist; shorter than the arm when it is angled |
| `trouser_length` | approximate | Vertical distance from left hip joint to left ankle; not waist-to-floor |
| `chest_width`, `waist_width`, `hip_width`, `thigh` | unsupported | Need a body outline; the original brightness-threshold scan often measured the background |
| `chest_circumference`, `waist`, `hip`, `thigh_circumference` | unsupported | Need measured body depth; the side photo is not used yet |
| `neck`, `neck_width` | unsupported | The original estimate measured the head |
| `shirt_length` | unsupported | A garment style length, not a body measurement |

All three approximate values share the calibration bias described above.

---

## Changes from the original prototype (0.1.0 → 0.2.0)

**Behaviour**
- Pose detection is stateless. The original shared one tracking-mode detector across requests, so identical requests returned different values (hips of 230, 278 and 285 cm), and one request's side photo skewed the next request.
- `height_cm` is required and validated. Missing or unreadable values used to become 152 cm silently, and `nan`, negative, `0` or `1e6` heights produced NaN, negative, zero or absurd values with 200 OK.
- Empty or unreadable images return a JSON 400 instead of an HTML 500 page, and uploads are limited to 32 MB.
- Photos without visible ankles are rejected, because the scale depends on them.
- Values that were fabricated, such as a waist equal to 10% of the photo width or hip widths that included background, are now `null` with status `unsupported`.
- The side photo is no longer run through the pose model and MiDaS; those results were always discarded.
- Height and measurements are no longer printed to the server log.

**Response schema** (existing keys are kept)
- Added: `measurement_status`, `units`, `warnings`, `algorithm_version`.
- Added on errors: `code` on every error, and `field` or `missing_landmarks` where relevant.
- Changed: unsupported measurements are `null` instead of numbers. Error messages for unreadable images and for missing feet are reworded.

**Removed** (still available in git history, commit `7138ba6`)
- The "A4 paper" fallback. It took the bounding box of the largest edge contour in the photo rather than detecting paper, and it only ran when no person was found, when no measurements were returned anyway.
- The unused heavy MediaPipe Pose model loaded at startup.
- The formulas behind the unsupported measurements: fixed-ratio scan rows, the ×1.15 / ×1.16 / ×1.35 corrections and the MiDaS "depth ratio".

---

## Libraries used

| Library | Purpose |
|---|---|
| `Flask` | HTTP API |
| `OpenCV` | Image decoding and processing |
| `MediaPipe` | Pose landmark detection |
| `Pillow` | Reading the image size before decoding |
| `PyTorch`, `torchvision` | [MiDaS](https://github.com/isl-org/MiDaS) depth; kept for comparison, not used by the API |

## 🤝 Contributions

PRs and suggestions are welcome! Fork this repo, raise an issue, or open a pull request.

## 📜 License

MIT License, © 2025 Javeria Tahir; see `LICENSE`.
