# facemask_identifier
# Real-Time Face Mask Compliance Checker

**CSC 126 Midterm Project** - Caraga State University, College of Computing and Information Sciences

A web application that checks, in real time, whether people are wearing a face mask **properly**
(nose *and* mouth covered). It works on a live webcam feed or an uploaded photo, handles up to four
people at once, and shows an overall result: `ACCESS GRANTED` or `PLEASE WEAR A MASK PROPERLY`.

## Why this project

Checking masks by eye at entrances is tiring and inconsistent, and a common problem is a mask worn
under the nose or on the chin. This project checks which parts of the lower face are still visible
instead of only checking whether a mask exists.

## Course topics covered

| Topic | What the project does | OpenCV functions |
|---|---|---|
| **a. Image operations** | Grayscale, histogram equalization and Gaussian blur before face detection; BGR to YCrCb conversion for colour analysis | `cvtColor`, `equalizeHist`, `GaussianBlur` |
| **b. Image segmentation** | Adaptive skin segmentation in YCrCb: the person's skin colour is learned from the forehead, pixels within a tolerance are marked as skin, and the mask is cleaned with morphology | `inRange`, `morphologyEx`, `getStructuringElement`, `bitwise_and` |
| **f. Facial landmarks** | Haar cascade finds faces, a 68-point LBF model locates the nose, mouth and chin zones | `CascadeClassifier`, `face.createFacemarkLBF`, `fit` |

## How it works

1. The browser camera sends small frames (320 px wide) to the Flask server.
2. `facekit.preprocess` converts the frame to grayscale, equalizes it and blurs it.
3. A Haar cascade detects faces; the 68 landmarks are fitted inside each face box.
4. `maskcore.process` learns the skin colour from the forehead, segments skin in YCrCb, and measures
   the fraction of skin in the nose, mouth and chin zones.
5. The three ratios are turned into a status:

| Status | Rule |
|---|---|
| `MASK ON` | nose and mouth covered |
| `MASK BELOW NOSE` | nose uncovered, mouth covered |
| `MASK ON CHIN` | nose and mouth uncovered, chin covered |
| `NO MASK` | nose, mouth and chin all uncovered |
| `CHECK MASK` | any other combination |

A zone counts as *uncovered* when skin fills at least 50% of the nose, 40% of the mouth or 50% of the
chin. The shown status is a majority vote over the last 9 frames to stop flicker.

## Project structure

```
app.py            Flask web server and API routes
maskcore.py       Mask-checking logic (segmentation, zones, classification)
facekit.py        Shared helpers: model download, face detection, tracker, drawing
templates/        index.html (web page)
static/           app.js and style.css (browser camera, overlay, pipeline view)
requirements.txt  Python dependencies
Procfile          Start command for Railway deployment
sample_inputs/    Sample photos to test the app          (add your own)
screenshots/      Screenshots of the output              (add your own)
```

## Setup

Requires **Python 3.11** (see `.python-version`) and an internet connection on the first run.

```bash
# 1. Clone the repository
git clone https://github.com/fluxxy-dotcom/facemask_identifier.git
cd facemask_identifier

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install the dependencies
pip install -r requirements.txt
```

> The project needs the **contrib** build of OpenCV (`opencv-contrib-python-headless`), which provides
> `cv2.face`. Do not install plain `opencv-python` in the same environment.

## Run

```bash
python app.py
```

Open **http://127.0.0.1:5000** in a browser, click **Open Camera** and allow camera access.
You can also click **or upload a photo instead** and choose an image from `sample_inputs/`.

On the first run the app downloads the 68-point landmark model (about 56 MB) to `models/lbfmodel.yaml`.
This can take a minute; later runs reuse the file.

Optional desktop mode (OpenCV window, `q` quit, `c` recalibrate, `s` save screenshot). The headless
OpenCV build in `requirements.txt` cannot open windows, so for this mode first run
`pip uninstall opencv-contrib-python-headless` and `pip install opencv-contrib-python==4.10.0.84`:

```bash
python maskcore.py
```

## Using the app

- **Live webcam**: shows a box, an ID and a status for each person, the banner, the number of people,
  how many have a mask on, and the compliance percentage.
- **Pipeline view** (below the video): thumbnails of each image operation and of the segmentation
  (face region, skin mask, skin only).
- **Save snapshot**: downloads the current frame with the labels.
- **Photo upload**: runs the same pipeline on a single image.

## Deployment

The `Procfile` starts the app with gunicorn for Railway:

```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300
```

The browser camera needs HTTPS, which Railway provides. Live site: [Railway link]

## Results

See the `screenshots/` folder and the project documentation PDF.

| No mask | Mask on | Mask below nose |
|---|---|---|
| ![No mask](screenshots/no_mask.png) | ![Mask on](screenshots/mask_on.png) | ![Below nose](screenshots/below_nose.png) |

## Limitations

- The Haar cascade detects frontal faces only.
- Skin segmentation depends on lighting; strong shadows or coloured light can shift skin colour.
- A mask close to skin tone, or a beard or make-up, can be misread because the method tests for skin.
- The forehead should be visible so the skin colour can be learned.
- At most four faces are processed per frame.

## Possible improvements

A more robust face detector, a trained mask classifier to complement segmentation, stronger lighting
normalisation (such as CLAHE), and logging of compliance over time.
