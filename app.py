from flask import Flask, request, jsonify

from measurement import ALGORITHM_VERSION
from measurement.estimator import estimate_measurements
from measurement.inputs import InputError, decode_image, parse_height_cm
from measurement.landmarks import detect_pose, validate_front_pose

MAX_UPLOAD_MB = 32  # two full-resolution phone photos fit comfortably
FOCAL_LENGTH = 600  # reported in debug_info for compatibility only; not used in any calculation

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

SIDE_IMAGE_UNUSED = {
    "code": "SIDE_IMAGE_UNUSED",
    "message": "The left_side photo was checked but no measurement uses it yet.",
}


def error_response(code, message, status=400, **details):
    body = {"error": message, "code": code}
    body.update(details)
    return jsonify(body), status


def read_upload(name):
    """Bytes of an uploaded file field, or None when it was not sent."""
    upload = request.files.get(name)
    if upload is None:
        return None
    data = upload.read()
    if not data and not upload.filename:
        return None  # an HTML form submits an empty part when no file is chosen
    return data


@app.errorhandler(413)
def payload_too_large(error):
    return error_response("PAYLOAD_TOO_LARGE", f"Uploads must total at most {MAX_UPLOAD_MB} MB.", 413)


@app.errorhandler(500)
def internal_error(error):
    return error_response("INTERNAL_ERROR", "Internal error while processing the images.", 500)


@app.route("/upload_images", methods=["POST"])
def upload_images():
    front_bytes = read_upload("front")
    if front_bytes is None:
        return error_response("MISSING_FRONT_IMAGE", "Missing front image for reference.", field="front")

    side_bytes = read_upload("left_side")
    try:
        front = decode_image(front_bytes, "front")
        if side_bytes is not None:
            decode_image(side_bytes, "left_side")  # validated now; no measurement uses it yet
        user_height_cm = parse_height_cm(request.form.get("height_cm"))
    except InputError as e:
        return error_response(e.code, e.message, field=e.field)

    # One stateless pose detection per request, shared by validation and measurement.
    image_height, image_width = front.shape[:2]
    landmarks = detect_pose(front)
    is_valid, error_msg, missing_landmarks = validate_front_pose(landmarks, image_width, image_height)
    if not is_valid:
        return error_response("INVALID_POSE", error_msg, pose="front", missing_landmarks=missing_landmarks)

    result = estimate_measurements(landmarks, image_width, image_height, user_height_cm)
    warnings = result["warnings"]
    if side_bytes is not None:
        warnings.append(dict(SIDE_IMAGE_UNUSED))

    return jsonify({
        "measurements": result["measurements"],
        "measurement_status": result["measurement_status"],
        "units": "cm",
        "warnings": warnings,
        "algorithm_version": ALGORITHM_VERSION,
        # Debug information to help troubleshoot measurements
        "debug_info": {
            "scale_factor": float(result["scale_factor"]),
            "focal_length": float(FOCAL_LENGTH),
            "user_height_cm": float(user_height_cm),
        },
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
