"""Turns front-photo pose landmarks into measurements, each with an explicit status."""
import math

from .geometry import scale_from_height
from .landmarks import CALIBRATION_LANDMARKS, PoseLandmark, unreliable_landmarks

APPROXIMATE = "approximate"  # a number is returned; not yet validated against tape measurements
UNSUPPORTED = "unsupported"  # the current method cannot estimate it; value is null
UNAVAILABLE = "unavailable"  # supported, but could not be estimated from this photo; value is null

# Every key the original API returned, in its original order.
MEASUREMENT_NAMES = (
    "shoulder_width", "chest_width", "chest_circumference", "waist_width", "waist",
    "hip_width", "hip", "neck", "neck_width", "arm_length", "shirt_length", "thigh",
    "thigh_circumference", "trouser_length",
)

# Unvalidated correction carried over from the original code. Changing it
# needs reference measurements, not a guess.
SHOULDER_CORRECTION = 1.1

_WIDTH_REASON = (
    "Needs a body outline. The original width scan used a fixed brightness threshold "
    "that cannot separate the person from the background."
)
_DEPTH_REASON = (
    "Needs measured body depth. The side photo is not used yet, and the original "
    "estimate was a fixed multiple of the front width."
)
UNSUPPORTED_REASONS = {
    "chest_width": _WIDTH_REASON,
    "waist_width": _WIDTH_REASON,
    "hip_width": _WIDTH_REASON,
    "thigh": _WIDTH_REASON,
    "chest_circumference": _DEPTH_REASON,
    "waist": _DEPTH_REASON,
    "hip": _DEPTH_REASON,
    "thigh_circumference": _DEPTH_REASON,
    "neck": "The original estimate used the nose-to-ear span, which measures the head, not the neck.",
    "neck_width": "The original estimate used the nose-to-ear span, which measures the head, not the neck.",
    "shirt_length": "A garment style length (1.2 x shoulder-to-hip), not a body measurement.",
}

LS, RS = PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER
LW, LH, LA = PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_ANKLE


def _shoulder_width_px(landmarks, image_width, image_height):
    left, right = landmarks[LS.value], landmarks[RS.value]
    return abs(left.x * image_width - right.x * image_width) * SHOULDER_CORRECTION


def _arm_length_px(landmarks, image_width, image_height):
    return abs(landmarks[LS.value].y * image_height - landmarks[LW.value].y * image_height)


def _trouser_length_px(landmarks, image_width, image_height):
    return abs(landmarks[LH.value].y * image_height - landmarks[LA.value].y * image_height)


# name -> (landmarks it needs, how it is computed, pixel length function)
APPROXIMATE_MEASUREMENTS = {
    "shoulder_width": (
        (LS, RS),
        "Shoulder-joint landmark span x 1.1 (unvalidated correction); "
        "not a tape measurement between the shoulder points.",
        _shoulder_width_px,
    ),
    "arm_length": (
        (LS, LW),
        "Vertical distance from the left shoulder to the left wrist landmark; "
        "shorter than the arm when it is angled or bent.",
        _arm_length_px,
    ),
    "trouser_length": (
        (LH, LA),
        "Vertical distance from the left hip-joint landmark to the left ankle landmark; "
        "not a waist-to-floor length.",
        _trouser_length_px,
    ),
}

WARNINGS = [
    {
        "code": "UNVALIDATED_ESTIMATES",
        "message": "Measurements have not been validated against tape measurements "
        "and must not be used for tailoring yet.",
    },
    {
        "code": "CALIBRATION_BIAS",
        "message": "The scale treats the nose-to-ankle span as full height, "
        "which tends to overestimate lengths.",
    },
]


def estimate_measurements(landmarks, image_width, image_height, height_cm):
    """Measurements for one front photo.

    Returns a dict with scale_factor (cm per pixel), measurements (name -> cm or
    None), measurement_status (name -> status details) and warnings. Raises
    ValueError if the calibration landmarks are unreliable.
    """
    missing = unreliable_landmarks(landmarks, CALIBRATION_LANDMARKS)
    if missing:
        raise ValueError(f"calibration landmarks not reliably detected: {', '.join(missing)}")
    nose_y = landmarks[PoseLandmark.NOSE.value].y * image_height
    lowest_ankle_y = max(
        landmarks[PoseLandmark.LEFT_ANKLE.value].y,
        landmarks[PoseLandmark.RIGHT_ANKLE.value].y,
    ) * image_height
    scale_factor = scale_from_height(nose_y, lowest_ankle_y, height_cm)

    measurements, statuses = {}, {}
    for name in MEASUREMENT_NAMES:
        measurements[name] = None
        if name in UNSUPPORTED_REASONS:
            statuses[name] = {"status": UNSUPPORTED, "reason": UNSUPPORTED_REASONS[name]}
            continue

        required, method, length_px = APPROXIMATE_MEASUREMENTS[name]
        missing = unreliable_landmarks(landmarks, required)
        if missing:
            statuses[name] = {
                "status": UNAVAILABLE,
                "reason": "Required landmarks were not reliably detected.",
                "missing_landmarks": missing,
            }
            continue

        value = round(length_px(landmarks, image_width, image_height) * scale_factor, 2)
        if not math.isfinite(value):
            statuses[name] = {"status": UNAVAILABLE, "reason": "The calculation did not produce a finite value."}
            continue
        measurements[name] = value
        statuses[name] = {"status": APPROXIMATE, "reason": method}

    return {
        "scale_factor": scale_factor,
        "measurements": measurements,
        "measurement_status": statuses,
        "warnings": [dict(warning) for warning in WARNINGS],
    }
