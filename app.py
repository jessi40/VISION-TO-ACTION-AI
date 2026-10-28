
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import os
import cv2
import base64
import numpy as np
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# YOLO model
model = YOLO("yolo11m.pt")


def image_to_data_url(image):
    """Convert an OpenCV image to a browser-compatible image URL."""
    success, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    if not success:
        raise ValueError("Could not encode image.")

    encoded_text = base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")

    return "data:image/jpeg;base64," + encoded_text


def detect_objects(image):
    """Detect objects and return the dimensions expected by index.html."""
    height, width = image.shape[:2]

    prediction = model.predict(image, verbose=False)
    result = prediction[0]

    objects = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        class_id = int(box.cls[0])
        confidence = float(box.conf[0]) * 100

        objects.append({
            "name": str(model.names[class_id]),
            "confidence": round(confidence, 2),

            "camera_x": round(center_x, 2),
            "camera_y": round(center_y, 2),

            # Required by the updated index.html
            "image_width": int(width),
            "image_height": int(height),

            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2)
        })

    annotated_image = result.plot()

    return annotated_image, objects


def make_response(image):
    """Create a consistent response for uploads and webcam frames."""
    annotated_image, objects = detect_objects(image)

    return {
        "original": image_to_data_url(image),
        "annotated": image_to_data_url(annotated_image),
        "results": objects
    }


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/detect_image", methods=["POST"])
def detect_image():
    """Handle the Upload & Detect button."""
    try:
        uploaded_file = request.files.get("image")

        if uploaded_file is None or not uploaded_file.filename:
            return jsonify({
                "error": "Please choose an image first."
            }), 400

        safe_name = secure_filename(uploaded_file.filename)

        if not safe_name:
            return jsonify({
                "error": "Invalid filename."
            }), 400

        file_bytes = uploaded_file.read()

        if not file_bytes:
            return jsonify({
                "error": "The uploaded image is empty."
            }), 400

        image_array = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                "error": "Could not read this image. Try JPG or PNG."
            }), 400

        # Keep a copy of the uploaded image in static/uploads
        saved_name = f"{uuid.uuid4().hex}_{safe_name}"
        saved_path = os.path.join(UPLOAD_FOLDER, saved_name)
        cv2.imwrite(saved_path, image)

        response_data = make_response(image)

        return jsonify(response_data)

    except Exception:
        app.logger.exception("Image detection failed")
        return jsonify({
            "error": "Image detection failed. Check the VS Code terminal."
        }), 500


@app.route("/detect_frame", methods=["POST"])
def detect_frame():
    """Handle frames sent by the webcam JavaScript."""
    try:
        data = request.get_json(silent=True) or {}
        image_data = data.get("image", "")

        if not image_data:
            return jsonify({
                "error": "No camera frame received."
            }), 400

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        raw_bytes = base64.b64decode(image_data, validate=True)
        image_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                "error": "Invalid camera frame."
            }), 400

        response_data = make_response(image)

        return jsonify(response_data)

    except Exception:
        app.logger.exception("Camera detection failed")
        return jsonify({
            "error": "Camera detection failed. Check the VS Code terminal."
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Vision to Action AI backend is running."
    })


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Image is too large. Maximum size is 10 MB."
    }), 413


if __name__ == "__main__":
    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000
    )



