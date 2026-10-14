# Face Mask Compliance Checker - Web Interface (CSC 126 Midterm)

A Flask web app: **HTML + CSS + JavaScript** in the browser, **Python + OpenCV** on the server.

```
Browser (index.html, style.css, app.js) <--HTTP/JSON--> Flask (app.py) --> OpenCV (maskcore.py + facekit.py)
```

## Folder layout
| File | Role |
|---|---|
| `app.py` | Flask server: routes, background webcam thread, MJPEG stream |
| `maskcore.py` | The OpenCV mask-checking logic (adaptive skin segmentation + landmark zones) |
| `facekit.py` | Shared helpers: model loading, Haar detection, tracking, drawing |
| `templates/index.html` | The page structure |
| `static/style.css` | Styling (light and dark mode) |
| `static/app.js` | Buttons, polling and photo upload using `fetch()` |

## Run
```bash
pip install -r requirements.txt   # OpenCV 4.x contrib + Flask
python app.py
```
Open **http://127.0.0.1:5000**. The landmark model downloads on the first run (or is reused from a sibling project folder).

## How the pieces connect
- **Live video:** `<img src="/video_feed">` shows an MJPEG stream; Python reads the webcam, runs OpenCV, draws boxes/labels and streams each annotated frame.
- **Live numbers:** `app.js` calls `GET /api/stats` twice a second and updates the banner, compliance bar and people list.
- **Buttons:** `POST /api/camera` (start/stop), `GET /api/snapshot` (download a JPEG - handy for your documentation screenshots).
- **Photo upload:** `POST /api/analyze` sends an image; Python returns the annotated picture (base64) and the results as JSON.

## Notes
The server owns the camera, so run it on the same computer you demo on and close other apps that use the webcam. Skin-coloured masks, beards and strong coloured light can fool the skin test.
