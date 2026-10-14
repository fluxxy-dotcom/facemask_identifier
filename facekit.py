"""
facekit.py - shared helpers for the CSC 126 face-based projects.

Provides: landmark-model download/loading, Haar face detection, a simple
multi-person tracker (persistent IDs), head-pose estimation (solvePnP),
drawing helpers and the webcam main loop.

Requires OpenCV 4.x *contrib* (cv2.face). OpenCV 5 is NOT supported.
"""
import os
import time
import urllib.request
from collections import deque

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Landmark model (68-point LBF). Searched in several places so you only need
# to download it once and can share it between projects.
# ----------------------------------------------------------------------------
MODEL_URL = ("https://raw.githubusercontent.com/kurnianggoro/GSOC2017/"
             "master/data/lbfmodel.yaml")
MODEL_MIN_BYTES = 50_000_000     # valid file is ~56 MB; smaller = corrupt
MODEL_SEARCH = ["models/lbfmodel.yaml", "../models/lbfmodel.yaml",
                "../drowsiness-detector/models/lbfmodel.yaml",
                "../phone-usage-detector/models/lbfmodel.yaml",
                "../facemask-checker/models/lbfmodel.yaml",
                "../attention-monitor/models/lbfmodel.yaml",
                "../expression-kiosk/models/lbfmodel.yaml",
                "../head-cursor/models/lbfmodel.yaml"]

# 68-point landmark groups (iBUG convention)
JAW = list(range(0, 17))
R_BROW, L_BROW = list(range(17, 22)), list(range(22, 27))
NOSE = list(range(27, 36))
R_EYE, L_EYE = list(range(36, 42)), list(range(42, 48))
MOUTH_OUTER, MOUTH_INNER = list(range(48, 60)), list(range(60, 68))

# 6 landmarks + generic 3-D face model for head pose
# (camera convention: x right, y down, z away from camera; nose at origin)
POSE_IDX = [30, 8, 36, 45, 48, 54]
MODEL_3D = np.array([[0, 0, 0], [0, 330, 65],
                     [-225, -170, 135], [225, -170, 135],
                     [-150, 150, 125], [150, 150, 125]], dtype=np.float64)


def ensure_model():
    """Return the path of a valid landmark model, downloading it if needed."""
    for path in MODEL_SEARCH:
        if os.path.exists(path) and os.path.getsize(path) >= MODEL_MIN_BYTES:
            return path
    path = MODEL_SEARCH[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Downloading landmark model (~56 MB), please wait...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print("Model saved to", path)
    return path


def load_detectors():
    """Return (Haar face detector, LBF landmark detector)."""
    model_path = ensure_model()
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(model_path)
    return cascade, facemark


# ----------------------------------------------------------------------------
# Image operations
# ----------------------------------------------------------------------------
def preprocess(frame):
    """Grayscale -> histogram equalization -> light blur."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)             # evens out lighting
    return cv2.GaussianBlur(gray, (3, 3), 0)  # suppresses sensor noise


def detect_faces(gray, cascade, max_faces=4, min_size=80):
    """Haar face boxes: largest `max_faces`, ordered left-to-right."""
    faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(min_size, min_size))
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:max_faces]
    return sorted([tuple(int(v) for v in f) for f in faces], key=lambda f: f[0])


def dist(p, q):
    return float(np.linalg.norm(np.asarray(p, float) - np.asarray(q, float)))


def eye_aspect_ratio(eye):
    """EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|); falls toward 0 when closed."""
    return (dist(eye[1], eye[5]) + dist(eye[2], eye[4])) / (2.0 * dist(eye[0], eye[3]))


# ----------------------------------------------------------------------------
# Head pose from landmarks (facial landmark topic)
# ----------------------------------------------------------------------------
def head_pose(shape, frame_w, frame_h):
    """
    Return (pitch, yaw, roll) in degrees, or None if the fit is bad.
      pitch > 0 : head tilted DOWN
      yaw   > 0 : face turned toward the LEFT side of the (mirrored) screen
    """
    image_pts = shape[POSE_IDX].astype(np.float64)
    f = float(frame_w)
    cam = np.array([[f, 0, frame_w / 2.0], [0, f, frame_h / 2.0], [0, 0, 1]])
    ok, rvec, tvec = cv2.solvePnP(MODEL_3D, image_pts, cam, np.zeros((4, 1)),
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tvec[2][0] <= 0:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    ang = cv2.RQDecomp3x3(rot)[0]
    return float(ang[0]), float(ang[1]), float(ang[2])


# ----------------------------------------------------------------------------
# Multi-person tracker: keeps a stable ID for each face between frames
# ----------------------------------------------------------------------------
class Person:
    """Base class for per-person state. Subclass and add your own fields."""
    _next_id = 1

    def __init__(self, center):
        self.id = Person._next_id
        Person._next_id += 1
        self.center = center
        self.missed = 0
        self.reset()

    def reset(self):
        """Called on creation and when the user presses 'c' (recalibrate)."""


class Tracker:
    def __init__(self, factory, max_missed=30):
        self.factory = factory
        self.max_missed = max_missed
        self.states = []
        self.dropped = []        # people removed during the last update()

    def update(self, faces):
        """Match face boxes to people (nearest centre). Returns states in order."""
        assigned, used = [], set()
        for (x, y, w, h) in faces:
            centre = np.array([x + w / 2.0, y + h / 2.0])
            best, best_d = None, float(max(w, h))
            for s in self.states:
                if s.id in used:
                    continue
                d = np.linalg.norm(centre - s.center)
                if d < best_d:
                    best, best_d = s, d
            if best is None:
                best = self.factory(centre)
                self.states.append(best)
            best.center, best.missed = centre, 0
            used.add(best.id)
            assigned.append(best)
        for s in self.states:
            if s.id not in used:
                s.missed += 1
        self.dropped = [s for s in self.states if s.missed > self.max_missed]
        self.states = [s for s in self.states if s.missed <= self.max_missed]
        return assigned


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
def put_text(frame, text, org, color=(255, 255, 255), scale=0.6):
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def draw_label(frame, face, text, color):
    x, y, w, h = face
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    put_text(frame, text, (x, max(y - 10, 20)), color, 0.6)


def paste_inset(frame, images, titles, size=(110, 80), margin=10):
    """Stack small BGR/gray images in the top-right corner (segmentation view)."""
    x = frame.shape[1] - size[0] - margin
    y = margin
    for img, title in zip(images, titles):
        if img is None or img.size == 0:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        small = cv2.resize(img, size)
        frame[y:y + size[1], x:x + size[0]] = small
        cv2.putText(frame, title, (x + 2, y + 12), cv2.FONT_HERSHEY_PLAIN,
                    0.9, (0, 255, 255), 1)
        y += size[1] + 6


def roi_from_points(shape, idx, frame_shape, pad_x=0.0, pad_y=0.0):
    """Clipped (x0, y0, x1, y1) box around landmark points `idx`, with padding."""
    pts = shape[idx].astype(np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    px, py = int(w * pad_x), int(h * pad_y)
    H, W = frame_shape[:2]
    return (max(x - px, 0), max(y - py, 0),
            min(x + w + px, W), min(y + h + py, H))


# ----------------------------------------------------------------------------
# Webcam main loop shared by every project
# ----------------------------------------------------------------------------
def run(title, factory, process, summary=None, on_key=None, on_drop=None,
        after_frame=None, max_faces=4, camera=0):
    """
    process(state, frame, gray, face, shape, frame_idx) -> (status, BGR color)
        is called once per person per frame; it may draw on `frame`.
    summary(frame, states, n_faces)     -> draw overall information (optional)
    on_key(key, states)                 -> extra keyboard handling (optional)
    on_drop(state)                      -> a person left for good (optional)
    after_frame(frame, states)          -> extra windows etc. (optional)
    Keys: q quit | c recalibrate everyone | s save screenshot
    """
    cascade, facemark = load_detectors()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera index/permissions.")
    tracker = Tracker(factory)
    prev, frame_idx = time.time(), 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)          # mirror view feels natural
        frame_idx += 1
        gray = preprocess(frame)

        faces = detect_faces(gray, cascade, max_faces)
        people = tracker.update(faces)
        if on_drop:
            for s in tracker.dropped:
                on_drop(s)
        if faces:
            ok_fit, landmarks = facemark.fit(gray, np.array(faces, dtype=np.int32))
            if ok_fit:
                for face, state, lm in zip(faces, people, landmarks):
                    out = process(state, frame, gray, face, lm[0], frame_idx)
                    if out:
                        draw_label(frame, face, f"Person {state.id}: {out[0]}", out[1])

        if summary:
            summary(frame, tracker.states, len(faces))
        fps = 1.0 / max(time.time() - prev, 1e-6)
        prev = time.time()
        put_text(frame, f"FPS: {fps:.0f}", (10, frame.shape[0] - 12))
        if after_frame:
            after_frame(frame, tracker.states)

        cv2.imshow(title + "  (q=quit, c=recalibrate, s=save)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            for s in tracker.states:
                s.reset()
        elif key == ord("s"):
            os.makedirs("screenshots", exist_ok=True)
            name = f"screenshots/shot_{int(time.time())}.png"
            cv2.imwrite(name, frame)
            print("Saved", name)
        if on_key:
            on_key(key, tracker.states)

    cap.release()
    cv2.destroyAllWindows()
