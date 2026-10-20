from types import SimpleNamespace

from measurement.landmarks import PoseLandmark

# A front-facing person standing upright in a 1000 x 2000 px portrait photo.
# MediaPipe "left" is the person's left, which is on the right of the photo.
WIDTH, HEIGHT = 1000, 2000
UPRIGHT_FRONT = {
    PoseLandmark.NOSE: (0.50, 0.10),
    PoseLandmark.LEFT_EAR: (0.53, 0.10),
    PoseLandmark.RIGHT_EAR: (0.47, 0.10),
    PoseLandmark.LEFT_SHOULDER: (0.65, 0.20),
    PoseLandmark.RIGHT_SHOULDER: (0.35, 0.20),
    PoseLandmark.LEFT_ELBOW: (0.70, 0.35),
    PoseLandmark.RIGHT_ELBOW: (0.30, 0.35),
    PoseLandmark.LEFT_WRIST: (0.72, 0.50),
    PoseLandmark.RIGHT_WRIST: (0.28, 0.50),
    PoseLandmark.LEFT_HIP: (0.56, 0.50),
    PoseLandmark.RIGHT_HIP: (0.44, 0.50),
    PoseLandmark.LEFT_KNEE: (0.56, 0.70),
    PoseLandmark.RIGHT_KNEE: (0.44, 0.70),
    PoseLandmark.LEFT_ANKLE: (0.56, 0.90),
    PoseLandmark.RIGHT_ANKLE: (0.44, 0.90),
}


def make_landmarks(moved=None, hidden=()):
    """33 landmarks as MediaPipe returns them; `moved` overrides (x, y), `hidden` sets low visibility."""
    positions = dict(UPRIGHT_FRONT)
    positions.update(moved or {})
    landmarks = []
    for landmark in PoseLandmark:
        x, y = positions.get(landmark, (0.5, 0.5))
        visibility = 0.1 if landmark in hidden else 0.99
        landmarks.append(SimpleNamespace(x=x, y=y, visibility=visibility))
    return landmarks
