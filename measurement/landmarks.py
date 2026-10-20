"""Pose landmark detection and reliability checks."""
import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
PoseLandmark = mp_holistic.PoseLandmark

MIN_VISIBILITY = 0.5

# The scale is calibrated on the span from the nose to the lowest ankle.
CALIBRATION_LANDMARKS = (PoseLandmark.NOSE, PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE)

# The original front-photo check, plus the ankles the calibration depends on.
FULL_BODY_LANDMARKS = (
    PoseLandmark.NOSE,
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.RIGHT_ANKLE,
)


def detect_pose(image_bgr):
    """Return the 33 pose landmarks for one photo, or None if no person is found.

    Each call builds its own static-mode graph, so no tracking state or landmark
    smoothing carries over from earlier photos or other requests.
    """
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
    ) as holistic:
        results = holistic.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    return list(results.pose_landmarks.landmark)


def unreliable_landmarks(landmarks, required):
    """Names of the required landmarks that are not visible or lie outside the photo."""
    missing = []
    for landmark in required:
        point = landmarks[landmark.value]
        if point.visibility < MIN_VISIBILITY or not (0 <= point.x <= 1 and 0 <= point.y <= 1):
            missing.append(landmark.name)
    return missing


def validate_front_pose(landmarks, image_width, image_height):
    """Check the front photo shows a full body. Returns (is_valid, message, missing_landmarks)."""
    if landmarks is None:
        return False, "No person detected. Please make sure you're clearly visible in the frame.", []

    missing = unreliable_landmarks(landmarks, FULL_BODY_LANDMARKS)
    if missing:
        return False, "Couldn't detect full body. Please make sure your whole body, including your feet, is visible.", missing

    # If the shoulder width is small compared to head size, likely a selfie
    nose = landmarks[PoseLandmark.NOSE.value]
    left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_width = abs(left_shoulder.x - right_shoulder.x) * image_width
    head_to_shoulder = abs(left_shoulder.y - nose.y) * image_height
    if shoulder_width < head_to_shoulder * 1.2:
        return False, "Please step back to show more of your upper body, not just your face.", []

    return True, "Validation passed - proceeding with measurements", []
