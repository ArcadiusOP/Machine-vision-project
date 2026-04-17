from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Dict

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from traffic_analyzer import TrafficAnalyzer, open_video_source


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
PLOT_DIR = BASE_DIR / "outputs"
PLOT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 600 * 1024 * 1024

sessions: Dict[str, TrafficAnalyzer] = {}
sessions_lock = Lock()


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def make_session_id() -> str:
    return str(int(time.time() * 1000))


def source_error_message(source_type: str, source_value) -> str:
    if source_type == "upload":
        return (
            "OpenCV could not read this video. Try an MP4 encoded as H.264/AVC, "
            "or convert the file with HandBrake/VLC and upload it again."
        )
    if isinstance(source_value, int):
        return (
            f"OpenCV could not open webcam index {source_value}. Try 0, 1, or 2, "
            "and close other apps that may be using the camera."
        )
    return (
        "OpenCV could not open that camera URL. Use a direct RTSP, HTTP MJPEG, "
        "or direct video stream URL, not a normal webpage link."
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start_analysis():
    source_type = request.form.get("source_type", "upload")
    pixels_per_meter = float(request.form.get("pixels_per_meter", 14.0))
    speed_limit_kmh = float(request.form.get("speed_limit_kmh", 50.0))
    expected_direction = request.form.get("expected_direction", "right").lower()
    uturn_angle_threshold = float(request.form.get("uturn_angle_threshold", 130.0))

    if source_type == "upload":
        file = request.files.get("video")
        if file is None or file.filename == "":
            return jsonify({"error": "Choose a traffic video file first."}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Use MP4, AVI, MOV, MKV, or WEBM video."}), 400

        filename = f"{make_session_id()}_{secure_filename(file.filename)}"
        source = UPLOAD_DIR / filename
        file.save(source)
        source_value = str(source)
    else:
        camera_url = request.form.get("camera_url", "").strip().strip('"').strip("'")
        if not camera_url:
            return jsonify({"error": "Enter a camera stream URL, local file path, or webcam index."}), 400
        source_value = int(camera_url) if camera_url.isdigit() else camera_url

    capture = open_video_source(source_value)
    if not capture.isOpened():
        capture.release()
        return jsonify({"error": source_error_message(source_type, source_value)}), 400

    session_id = make_session_id()
    analyzer = TrafficAnalyzer(
        capture=capture,
        pixels_per_meter=pixels_per_meter,
        speed_limit_kmh=speed_limit_kmh,
        expected_direction=expected_direction,
        uturn_angle_threshold=uturn_angle_threshold,
    )

    with sessions_lock:
        sessions[session_id] = analyzer

    return jsonify({"session_id": session_id})


@app.route("/video_feed/<session_id>")
def video_feed(session_id: str):
    with sessions_lock:
        analyzer = sessions.get(session_id)
    if analyzer is None:
        return Response("Session not found", status=404)

    def generate():
        while True:
            ok, jpeg = analyzer.next_jpeg()
            if not ok:
                break
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/metrics/<session_id>")
def metrics(session_id: str):
    with sessions_lock:
        analyzer = sessions.get(session_id)
    if analyzer is None:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(analyzer.snapshot())


@app.route("/api/stop/<session_id>", methods=["POST"])
def stop(session_id: str):
    with sessions_lock:
        analyzer = sessions.pop(session_id, None)
    plot_path = None
    if analyzer is not None:
        plot_path = analyzer.save_session_plot(PLOT_DIR)
        analyzer.release()
    return jsonify({"stopped": True, "plot_path": str(plot_path) if plot_path else None})


@app.template_filter("json")
def to_json(value):
    return json.dumps(value)


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=True, threaded=True)
