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
# Pipeline view: small thumbnails of each processing step
# ----------------------------------------------------------------------------
def thumb(img, width=160):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h = max(1, round(img.shape[0] * width / img.shape[1]))
    small = cv2.resize(img, (width, h))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 65])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def build_stages(frame, states):
    """frame must be the clean (undrawn) image. Returns the two groups shown on the page."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(eq, (3, 3), 0)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ops = [("Original", frame), ("Grayscale", gray), ("Histogram equalization", eq),
           ("Gaussian blur", blur), ("YCrCb colour conversion", ycrcb)]
    seg = []
    vis = [st for st in states if st.missed == 0 and hasattr(st, "mask_view") and hasattr(st, "box")]
    if vis:
        big = max(vis, key=lambda st: st.is_biggest)
        x, y, w, h = big.box
        H, W = frame.shape[:2]
        rx0, ry0 = max(x - w // 4, 0), max(y - h // 2, 0)
        rx1, ry1 = min(x + w + w // 4, W), min(y + h + h // 4, H)
        crop, m = frame[ry0:ry1, rx0:rx1], big.mask_view
        if m is not None and m.shape[:2] == crop.shape[:2]:
            seg = [("Face region", crop), ("Skin mask", m),
                   ("Skin only", cv2.bitwise_and(crop, crop, mask=m))]
    return {"ops": [{"name": n, "img": thumb(i)} for n, i in ops],
            "seg": [{"name": n, "img": thumb(i)} for n, i in seg]}


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
        state.box = face
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
    """Photo upload: same pipeline as the live camera, returns the clean photo + boxes."""
    file = request.files.get("image")
    if file is None:
        return jsonify(ok=False, error="No image uploaded"), 400
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(ok=False, error="That file is not a readable image"), 400
    h, w = img.shape[:2]
    if w > 960:
        img = cv2.resize(img, (960, round(h * 960 / w)))
    tracker = fk.Tracker(mask.PersonState)
    boxes, stable = [], []
    for i in range(mask.SMOOTH_FRAMES):          # repeat so the majority vote has history
        boxes, stable, _ = analyse_boxes(img.copy(), tracker, i)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    data = summarise(stable)
    data.update(ok=True, boxes=boxes, w=img.shape[1], h=img.shape[0],
                image="data:image/jpeg;base64," + base64.b64encode(buf).decode(),
                stages=build_stages(img, tracker.states))
    return jsonify(data)


@app.route("/api/snapshot")
def api_snapshot():
    if camera.jpeg is None:
        return jsonify(ok=False, error="Start the camera first"), 404
    name = f"mask_snapshot_{int(time.time())}.jpg"
    return Response(camera.jpeg, mimetype="image/jpeg",
                    headers={"Content-Disposition": f"attachment; filename={name}"})


# ----------------------------------------------------------------------------
# Browser camera: the page sends one JPEG frame at a time, we annotate and return it
# ----------------------------------------------------------------------------
SESSIONS = {}                      # session id -> {"tracker", "idx", "seen"}
SESSION_LOCK = threading.Lock()


def get_session(sid):
    now = time.time()
    with SESSION_LOCK:
        for k in [k for k, v in SESSIONS.items() if now - v["seen"] > 120]:
            del SESSIONS[k]        # forget visitors who left
        sess = SESSIONS.setdefault(sid, {"tracker": fk.Tracker(mask.PersonState), "idx": 0})
        sess["seen"] = now
        sess["idx"] += 1
        return sess


@app.route("/api/frame", methods=["POST"])
def api_frame():
    file = request.files.get("image")
    sid = (request.form.get("sid") or "anon")[:64]
    if file is None:
        return jsonify(ok=False, error="No frame"), 400
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(ok=False, error="Bad frame"), 400
    sess = get_session(sid)
    people = analyse(img, sess["tracker"], sess["idx"])
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    data = summarise(people)
    data.update(ok=True, image="data:image/jpeg;base64," + base64.b64encode(buf).decode())
    return jsonify(data)


def analyse_boxes(frame, tracker, idx, want_stages=False):
    """Like analyse(), but returns data only (boxes + statuses) - no drawing, no image."""
    clean = frame.copy() if want_stages else None
    gray = fk.preprocess(frame)
    with DETECT_LOCK:
        faces = fk.detect_faces(gray, CASCADE, max_faces=4)
        people = tracker.update(faces)
        landmarks = []
        if faces:
            ok, lms = FACEMARK.fit(gray, np.array(faces, dtype=np.int32))
            landmarks = lms if ok else []
    boxes = []
    for face, state, lm in zip(faces, people, landmarks):
        status, _ = mask.process(state, frame, gray, face, lm[0], idx)
        state.box = face
        boxes.append({"id": state.id, "status": status, "box": [int(v) for v in face]})
    stable = [{"id": s.id, "status": Counter(s.history).most_common(1)[0][0]}
              for s in tracker.states if s.missed == 0 and s.history]
    stages = build_stages(clean, tracker.states) if want_stages else None
    return boxes, stable, stages


@app.route("/api/detect", methods=["POST"])
def api_detect():
    file = request.files.get("image")
    sid = (request.form.get("sid") or "anon")[:64]
    if file is None:
        return jsonify(ok=False, error="No frame"), 400
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(ok=False, error="Bad frame"), 400
    sess = get_session(sid)
    want = request.form.get("stages") == "1"
    boxes, stable, stages = analyse_boxes(img, sess["tracker"], sess["idx"], want)
    data = summarise(stable)
    data.update(ok=True, boxes=boxes, w=img.shape[1], h=img.shape[0])
    if stages:
        data["stages"] = stages
    return jsonify(data)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)