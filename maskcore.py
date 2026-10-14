"""
CSC 126 Midterm Project - Real-Time Face Mask Compliance Checker
Input : live webcam feed (multi-person)
Topics: (a) Image Operations   - grayscale, equalization, blur, colour conversion, overlays
        (b) Image Segmentation - ADAPTIVE skin segmentation in YCrCb colour space
        (f) Facial Landmarks   - 68-point model locates nose / mouth / chin zones

How it works
  1. Landmarks locate three zones on the lower face: nose, mouth and chin.
  2. The person's own skin colour is learned from the forehead (adaptive), so it
     works for different skin tones and lighting.
  3. Each zone is segmented: how much of it is still skin?
        nose + mouth covered      -> MASK ON
        only mouth covered        -> MASK BELOW NOSE
        only chin covered         -> MASK ON CHIN
        nothing covered           -> NO MASK
Controls: q quit | s save screenshot
"""
from collections import Counter, deque

import cv2
import numpy as np

import facekit as fk

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SKIN_TOL = 14           # allowed Cr/Cb distance from the person's skin colour
NOSE_SKIN_MIN = 0.50    # zone counts as "uncovered" when >= this fraction is skin
MOUTH_SKIN_MIN = 0.40
CHIN_SKIN_MIN = 0.50
SMOOTH_FRAMES = 9       # majority vote over the last N frames (stops flicker)

GREEN, ORANGE, RED, YELLOW = (0, 200, 0), (0, 140, 255), (0, 0, 255), (0, 255, 255)


class PersonState(fk.Person):
    def reset(self):
        self.history = deque(maxlen=SMOOTH_FRAMES)


# ----------------------------------------------------------------------------
# Segmentation (topic b): learn this person's skin colour, then segment
# ----------------------------------------------------------------------------
def learn_skin_colour(ycrcb, shape, face):
    """Median (Cr, Cb) of the forehead; falls back to None if it isn't skin."""
    h_face = face[3]
    brow_top = int(shape[17:27, 1].min())
    y1 = brow_top - int(0.04 * h_face)
    y0 = y1 - int(0.18 * h_face)
    x0, x1 = int(shape[19][0]), int(shape[24][0])
    H, W = ycrcb.shape[:2]
    y0, y1, x0, x1 = max(y0, 0), min(y1, H), max(x0, 0), min(x1, W)
    if y1 - y0 < 3 or x1 - x0 < 3:
        return None
    patch = ycrcb[y0:y1, x0:x1].reshape(-1, 3)
    # keep only pixels inside a generic skin range (ignores hair / hats)
    ok = ((patch[:, 1] > 133) & (patch[:, 1] < 173) &
          (patch[:, 2] > 77) & (patch[:, 2] < 127))
    if ok.mean() < 0.3:
        return None
    return float(np.median(patch[ok, 1])), float(np.median(patch[ok, 2]))


def skin_segment(ycrcb, ref):
    """Binary skin mask. Adaptive when `ref` is known, generic range otherwise."""
    if ref is None:
        mask = cv2.inRange(ycrcb, (30, 133, 77), (255, 173, 127))
    else:
        cr, cb = ref
        mask = cv2.inRange(ycrcb, (30, max(cr - SKIN_TOL, 0), max(cb - SKIN_TOL, 0)),
                           (255, min(cr + SKIN_TOL, 255), min(cb + SKIN_TOL, 255)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def skin_ratio(mask, box):
    x0, y0, x1, y1 = box
    zone = mask[y0:y1, x0:x1]
    return float(np.count_nonzero(zone)) / zone.size if zone.size else 0.0


# ----------------------------------------------------------------------------
# Landmark zones (topic f)
# ----------------------------------------------------------------------------
def lower_face_zones(shape, frame_shape):
    """Return nose, mouth and chin boxes (x0, y0, x1, y1) from the landmarks."""
    nose = fk.roi_from_points(shape, [31, 32, 33, 34, 35], frame_shape, 0.15, 0.2)
    mouth = fk.roi_from_points(shape, fk.MOUTH_OUTER, frame_shape, 0.20, 0.30)
    H, W = frame_shape[:2]
    cy0 = int((shape[57][1] + shape[8][1]) / 2)          # between lip and chin
    chin = (max(int(shape[6][0]), 0), max(cy0, 0),
            min(int(shape[10][0]), W), min(int(shape[8][1]) - 2, H))
    return nose, mouth, chin


def classify(nose_skin, mouth_skin, chin_skin):
    """Turn the three skin ratios into a mask status."""
    nose_open = nose_skin >= NOSE_SKIN_MIN
    mouth_open = mouth_skin >= MOUTH_SKIN_MIN
    chin_open = chin_skin >= CHIN_SKIN_MIN
    if nose_open and mouth_open and chin_open:
        return "NO MASK"
    if not nose_open and not mouth_open:
        return "MASK ON"
    if nose_open and not mouth_open:
        return "MASK BELOW NOSE"
    if nose_open and mouth_open and not chin_open:
        return "MASK ON CHIN"
    return "CHECK MASK"


STATUS_COLOR = {"MASK ON": GREEN, "NO MASK": RED, "MASK BELOW NOSE": ORANGE,
                "MASK ON CHIN": ORANGE, "CHECK MASK": YELLOW}


# ----------------------------------------------------------------------------
# Per-person processing
# ----------------------------------------------------------------------------
def process(state, frame, gray, face, shape, frame_idx):
    x, y, w, h = face
    H, W = frame.shape[:2]
    # work on a padded face region only (faster than the whole frame)
    rx0, ry0 = max(x - w // 4, 0), max(y - h // 2, 0)
    rx1, ry1 = min(x + w + w // 4, W), min(y + h + h // 4, H)
    roi = cv2.GaussianBlur(frame[ry0:ry1, rx0:rx1], (5, 5), 0)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

    shifted = shape - np.array([rx0, ry0], dtype=shape.dtype)   # ROI coordinates
    ref = learn_skin_colour(ycrcb, shifted, (x - rx0, y - ry0, w, h))
    mask = skin_segment(ycrcb, ref)

    nose, mouth, chin = lower_face_zones(shifted, roi.shape)
    ratios = [skin_ratio(mask, z) for z in (nose, mouth, chin)]
    raw = classify(*ratios)
    state.history.append(raw)
    status = Counter(state.history).most_common(1)[0][0]          # majority vote
    color = STATUS_COLOR[status]

    # visual evidence: the three zones, coloured by how much skin remains
    for zone, ratio, name in zip((nose, mouth, chin), ratios, ("nose", "mouth", "chin")):
        zx0, zy0, zx1, zy1 = zone
        zc = GREEN if ratio >= 0.45 else RED
        cv2.rectangle(frame, (zx0 + rx0, zy0 + ry0), (zx1 + rx0, zy1 + ry0), zc, 1)
        fk.put_text(frame, f"{ratio:.0%}", (zx1 + rx0 + 3, zy0 + ry0 + 12), zc, 0.4)
    state.mask_view = mask                                          # for the inset
    state.is_biggest = (w * h)
    return status, color


# ----------------------------------------------------------------------------
# Overall summary (kiosk-style banner)
# ----------------------------------------------------------------------------
def summary(frame, states, n_faces):
    visible = [s for s in states if s.missed == 0 and s.history]
    statuses = [Counter(s.history).most_common(1)[0][0] for s in visible]
    masked = sum(st == "MASK ON" for st in statuses)
    total = len(statuses)
    pct = 100.0 * masked / total if total else 0.0
    fk.put_text(frame, f"Faces: {total}   Masked: {masked}   Compliance: {pct:.0f}%",
                (10, 28), GREEN if masked == total else RED, 0.7)
    if total:
        ok = masked == total
        text = "ACCESS GRANTED" if ok else "PLEASE WEAR A MASK PROPERLY"
        col = (0, 160, 0) if ok else (0, 0, 220)
        cv2.rectangle(frame, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0] - 30),
                      col, -1)
        cv2.putText(frame, text, (15, frame.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    # segmentation inset for the largest visible face
    if visible:
        big = max(visible, key=lambda s: s.is_biggest)
        fk.paste_inset(frame, [big.mask_view], [f"skin P{big.id}"], size=(110, 110))


if __name__ == "__main__":
    fk.run("Face Mask Compliance Checker", PersonState, process, summary=summary)
