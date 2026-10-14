"""
Flask web interface for the Face Mask Compliance Checker.

  Browser (HTML + CSS + JavaScript)  <--HTTP-->  Flask (this file)  -->  OpenCV (maskcore.py)

Routes
  GET  /                 the web page
  GET  /video_feed       live MJPEG stream of the annotated webcam (shown in an <img>)
  POST /api/camera       {"action": "start" | "stop"}  - turn the Python webcam on/off
  GET  /api/stats        JSON: running, fps, people, masked, compliance, banner
  POST /api/analyze      upload a photo (form field "image") -> annotated photo + results
  GET  /api/snapshot     download the current annotated frame as a JPEG

Run:  python app.py   then open  http://127.0.0.1:5000
"""
import base64
import threading
import time
from collections import Counter

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

import facekit as fk
import maskcore as mask          # the OpenCV mask-checking logic (unchanged)

app = Flask(__name__)
CASCADE, FACEMARK = fk.load_detectors()
DETECT_LOCK = threading.Lock()   # detectors are shared between request threads


# ----------------------------------------------------------------------------
# Core: run the whole mask pipeline on ONE frame (draws on the frame)
# ----------------------------------------------------------------------------
def analyse(frame, tracker, idx):
    """Return a list of {"id", "status"} for the people currently visible."""
    gray = fk.preprocess(frame)                                  # image operations
    with DETECT_LOCK:
        faces = fk.detect_faces(gray, CASCADE, max_faces=4)      # Haar cascade
        people = tracker.update(faces)                           # stable IDs
        landmarks = []
        if faces:
            ok, lms = FACEMARK.fit(gray, np.array(faces, dtype=np.int32))
            landmarks = lms if ok else []                        # 68 landmarks
    for face, state, lm in zip(faces, people, landmarks):
        status, color = mask.process(state, frame, gray, face, lm[0], idx)
        fk.draw_label(frame, face, f"Person {state.id}: {status}", color)
    mask.summary(frame, tracker.states, len(faces))              # banner + mask inset
    return [{"id": s.id, "status": Counter(s.history).most_common(1)[0][0]}
            for s in tracker.states if s.missed == 0 and s.history]


def summarise(people):
    """Turn the people list into the numbers the web page displays."""
    total = len(people)
    masked = sum(p["status"] == "MASK ON" for p in people)
    if total == 0:
        banner = "No faces detected"
    elif masked == total:
        banner = "ACCESS GRANTED"
    else:
        banner = "PLEASE WEAR A MASK PROPERLY"
    return {"people": people, "faces": total, "masked": masked,
            "compliance": round(100.0 * masked / total) if total else 0, "banner": banner}


# ----------------------------------------------------------------------------
# Webcam running in a background thread (the server owns the camera)
# ----------------------------------------------------------------------------
class Camera:
    def __init__(self):
        self.lock = threading.Lock()
        self.cap, self.running = None, False
        self.jpeg, self.people, self.fps, self.error = None, [], 0.0, None

    def start(self):
        with self.lock:
            if self.running:
                return True
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.error = "Cannot open the webcam. Close other apps that use it."
                return False
            self.cap, self.running, self.error = cap, True, None
            threading.Thread(target=self._loop, daemon=True).start()
            return True

    def stop(self):
        self.running = False

    def _loop(self):
        tracker, idx, prev = fk.Tracker(mask.PersonState), 0, time.time()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            idx += 1
            people = analyse(frame, tracker, idx)
            now = time.time()
            fps, prev = 1.0 / max(now - prev, 1e-6), now
            self.fps = 0.8 * self.fps + 0.2 * fps
            fk.put_text(frame, f"FPS: {self.fps:.0f}", (10, frame.shape[0] - 12))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.jpeg, self.people = buf.tobytes(), people
        self.cap.release()
        self.running, self.jpeg, self.people = False, None, []


camera = Camera()


def mjpeg_stream():
    """Yield the latest annotated frame forever (multipart/x-mixed-replace)."""
    while camera.running:
        if camera.jpeg is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + camera.jpeg + b"\r\n")
        time.sleep(0.03)


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/camera", methods=["POST"])
def api_camera():
    action = (request.get_json(silent=True) or {}).get("action")
    if action == "start":
        if not camera.start():
            return jsonify(ok=False, error=camera.error), 500
    elif action == "stop":
        camera.stop()
    else:
        return jsonify(ok=False, error="action must be 'start' or 'stop'"), 400
    return jsonify(ok=True, running=camera.running)


@app.route("/api/stats")
def api_stats():
    data = summarise(camera.people)
    data.update(running=camera.running, fps=round(camera.fps, 1), error=camera.error)
    return jsonify(data)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    file = request.files.get("image")
    if file is None:
        return jsonify(ok=False, error="No image uploaded"), 400
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(ok=False, error="That file is not a readable image"), 400
    scale = min(1.0, 800 / img.shape[1])
    img = cv2.resize(img, None, fx=scale, fy=scale)
    tracker = fk.Tracker(mask.PersonState)
    for i in range(mask.SMOOTH_FRAMES):          # fill the majority-vote history
        out = img.copy()
        people = analyse(out, tracker, i)
    _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    data = summarise(people)
    data.update(ok=True, image="data:image/jpeg;base64," + base64.b64encode(buf).decode())
    return jsonify(data)


@app.route("/api/snapshot")
def api_snapshot():
    if camera.jpeg is None:
        return jsonify(ok=False, error="Start the camera first"), 404
    name = f"mask_snapshot_{int(time.time())}.jpg"
    return Response(camera.jpeg, mimetype="image/jpeg",
                    headers={"Content-Disposition": f"attachment; filename={name}"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)