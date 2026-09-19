# FaceRecognition

A real-time face recognition demo built with Java + OpenCV. A webcam feed is
scanned for faces (Haar cascade detection), and each detected face is
classified as a specific trained person ("idan") or "unknown" (LBPH
recognition).

## How it works (pipeline)

1. **Collect data** — photos/short videos of the target person and of other
   people go into `data/idan/` and `data/unknown/`.
2. **(optional) Extract frames** — `VideoFrameExtractor` pulls stills out of
   any videos in those folders so you don't have to export frames by hand.
3. **Train** — `Trainer` detects the face in each image with the Haar
   cascade, crops+resizes it to a standard 200×200 grayscale patch, and feeds
   all of them into an `LBPHFaceRecognizer` with label `0` (idan) or `1`
   (unknown). The trained model is saved to `models/model.xml`.
4. **Recognize live** — `Camera` runs the same detect → crop → resize
   pipeline on every webcam frame, then calls `recognizer.predict()` to get
   back a label and a confidence (distance) score. A face is only shown as
   "idan" if it predicts label `0` **and** the confidence is under a strict
   threshold; everything else is "unknown". An on-screen overlay also gives
   the user positioning feedback (move left/right/closer/farther) to help
   frame their face well.

## Architecture

| File | Role |
|---|---|
| `Camera.java` | Main app — live webcam detection + recognition loop. |
| `Trainer.java` | Offline script that builds `models/model.xml` from `data/`. |
| `FaceUtils.java` | Shared crop+resize helper used by **both** `Camera` and `Trainer`, so training and live inference preprocess faces identically (see "train/inference skew" below). |
| `FaceDataset.java` | Loads labelled face crops from a `data/` folder; shared by `Trainer` and `Evaluator`. Also groups video-extracted frames so they can't leak across a train/test split. |
| `Evaluator.java` | Measures accuracy with grouped k-fold cross-validation and sweeps the accept threshold, reporting false-accept / false-reject rates. Uses the same preprocessing and accept rule (`FaceUtils.isIdan`) as `Camera`. |
| `VideoFrameExtractor.java` | Turns videos under `data/` into training stills. |
| `Photo.java` / `Video.java` | Detection-only smoke tests (no recognition) against a static image/video — useful for sanity-checking the cascade file without a trained model or webcam. |

```
FaceRecognition/
├── pom.xml
├── models/
│   ├── haarcascade_frontalface_alt2.xml   # face DETECTION model (generic, from OpenCV, tracked in git)
│   └── model.xml                          # face RECOGNITION model (yours, gitignored)
├── data/                                  # your training photos/videos, gitignored
│   ├── idan/
│   └── unknown/
└── src/main/java/com/idanshaviner/facerecognition/
    ├── Camera.java
    ├── Trainer.java
    ├── FaceUtils.java
    ├── FaceDataset.java
    ├── Evaluator.java
    ├── VideoFrameExtractor.java
    ├── Photo.java
    └── Video.java
```

## Why these choices

- **Haar cascade for detection.** Fast, ships with OpenCV, needs no training
  of its own, and is good enough for a face filling most of the frame at
  webcam distance. A DNN-based detector (OpenCV's `dnn` face detector,
  MTCNN, etc.) would be more robust to angle/occlusion/lighting, at the cost
  of a heavier model and slower inference — overkill for "one person, one
  webcam, roughly facing the camera."
- **LBPH for recognition**, instead of deep embeddings (FaceNet, ArcFace,
  dlib's `face_recognition`). LBPH trains instantly on CPU from a handful of
  images per class and produces a small, human-readable XML model — a good
  fit for a personal project trained on a few dozen selfies. The trade-off:
  it doesn't generalize the way an embedding model does — it's a closed-set
  classifier that has to be retrained from scratch for every new person,
  rather than comparing against a shared embedding space.
- **Confidence threshold on top of the classifier.** `LBPHFaceRecognizer`
  always predicts *some* trained label — it has no built-in "none of these"
  output. To approximate open-set rejection, this project (a) trains an
  explicit "other people" class (label `1`) so the model learns a boundary
  around "idan," and (b) still applies a strict distance threshold on top —
  in LBPH, a *lower* score means a *better* match, so a weak match against
  label `0` still gets rejected as "unknown" instead of trusted outright.
- **Crop to the face box before resizing**, rather than using the whole
  frame, so the recognizer focuses on the face and isn't thrown off by
  background changes between training photos and live video.
- **Positioning overlay is independent of recognition** — it's computed
  purely from the raw detection box's size/position, so it doesn't affect
  (or get affected by) the crop fed into the recognizer.

## Problems encountered & how they were resolved

1. **Train/inference skew (found and fixed).** `Camera` used to crop
   *inward* from the Haar box before recognizing, to trim out hair/
   background — but `Trainer` trained on the *raw* Haar box. That mismatch
   meant LBPH was comparing histograms computed from differently-framed
   images at train time vs. predict time, which silently hurts accuracy.
   Fixed by moving both crops into one shared `FaceUtils.extractFaceROI()`
   used by both `Camera` and `Trainer`, so training and inference now see
   identical preprocessing.
2. **A `.gitignore` typo almost leaked private photos.** The rule meant to
   exclude the training-data folder was written as `data\` (trailing
   backslash) instead of `data/`, so it silently did nothing — personal
   training photos could have ended up committed to git. Fixed the pattern
   and verified with `git check-ignore` that photos are excluded while
   lightweight `.gitkeep` files keep the folder structure visible in the repo.
3. **No build system.** The project only built inside one IntelliJ setup, via
   a hand-edited `.iml` pointing at a hardcoded absolute jar path on one
   machine. Added `pom.xml` so it builds with `mvn compile`, with the OpenCV
   jar path exposed as an overridable property. OpenCV itself doesn't publish
   official Java artifacts to Maven Central (community repackagings exist,
   e.g. `org.openpnp:opencv`), and the Java classes must match the native
   library exactly, so the pom points at the locally built jar — that
   guarantees jar + native lib are the same build, contrib `face` module
   included.
4. **Inconsistent, version-pinned native library loading.** `Camera`/
   `Trainer` called `System.loadLibrary("opencv_java4100")` (hardcoding the
   OpenCV version), while `Photo`/`Video` used the version-agnostic
   `Core.NATIVE_LIBRARY_NAME`. Standardized on the latter everywhere so an
   OpenCV upgrade doesn't silently break half the entry points.
5. **Hardcoded personal file paths.** `Photo`/`Video` pointed at one
   specific file on one machine's disk and used a different absolute cascade
   path than `Camera`/`Trainer`. Both now take their input path as a CLI
   argument and share the same repo-relative cascade path as the rest of the
   project.
6. **OpenCV needs Java bindings + the contrib `face` module.**
   `LBPHFaceRecognizer` lives in `opencv_contrib`, and the Java wrapper only
   exists if OpenCV was built with Java enabled. The Homebrew `opencv`
   installed on this machine (`/opt/homebrew`) has no `opencv-*.jar`, so the
   project uses a separate native arm64 build in `/usr/local`
   (`share/java/opencv4/opencv-4100.jar` + `libopencv_java4100.dylib`,
   version string `4.10.0-dev`, so most likely built from source with Java
   and contrib enabled). If you reinstall OpenCV, make sure both are enabled
   and update `opencv.jar` in `pom.xml` and the library path when running.

## Prerequisites

- JDK 17+ (built and tested on JDK 23, arm64 — the JVM and the OpenCV
  native library must be the same CPU architecture)
- OpenCV 4.x built with **Java bindings** and the **`opencv_contrib` `face`
  module**. Check for a jar at `<prefix>/share/java/opencv4/opencv-*.jar` and
  a `libopencv_java*.dylib`/`.so`/`.dll` next to it. If you don't have those,
  build OpenCV from source with `-DBUILD_JAVA=ON` and
  `-DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules`. This project was
  verified against `/usr/local/share/java/opencv4/opencv-4100.jar`.
- Maven, optional — plain `javac`/`java` work fine too (shown below).

## How to run

Set these once per shell to match your OpenCV install:

```bash
export OPENCV_JAR=/usr/local/share/java/opencv4/opencv-4100.jar
export OPENCV_LIB_DIR=/usr/local/share/java/opencv4
```

**Build:**
```bash
mvn compile -Dopencv.jar="$OPENCV_JAR"
```
or, without Maven:
```bash
mkdir -p target/classes
javac -cp "$OPENCV_JAR" -d target/classes $(find src/main/java -name "*.java")
```

**1. Collect training data** — put photos/short videos of yourself in
`data/idan/` and of other people in `data/unknown/` (folders already exist,
gitignored). Have videos instead of photos? Extract stills first:
```bash
java -cp "target/classes:$OPENCV_JAR" -Djava.library.path="$OPENCV_LIB_DIR" \
  com.idanshaviner.facerecognition.VideoFrameExtractor
```

**2. Train** (writes `models/model.xml`):
```bash
java -cp "target/classes:$OPENCV_JAR" -Djava.library.path="$OPENCV_LIB_DIR" \
  com.idanshaviner.facerecognition.Trainer
```

**2b. Evaluate (recommended before trusting the threshold).** Runs grouped
k-fold cross-validation over `data/` and prints false-accept (strangers
accepted as you) and false-reject (you rejected) rates across thresholds,
plus the threshold at the equal-error point and at a target false-accept rate:
```bash
java -cp "target/classes:$OPENCV_JAR" -Djava.library.path="$OPENCV_LIB_DIR" \
  com.idanshaviner.facerecognition.Evaluator
```
Options: `--folds N` (default 5), `--seed S` (default 42), `--far-target F`
(default 0.01), `--no-detect` (images are already cropped faces, skip Haar).
Frames extracted from the same video are kept on one side of each split;
separate photos count as independent. You need at least 2 independent photos
or videos in each folder, and far more for stable numbers — with ~30 test
faces per class, one face moves a rate by ~3%.

**3. Run live recognition** (press **Esc** to quit; console logs predicted
label/confidence per face per frame):
```bash
java -cp "target/classes:$OPENCV_JAR" -Djava.library.path="$OPENCV_LIB_DIR" \
  com.idanshaviner.facerecognition.Camera
```

**Detection-only smoke tests:**
```bash
java -cp "target/classes:$OPENCV_JAR" -Djava.library.path="$OPENCV_LIB_DIR" \
  com.idanshaviner.facerecognition.Photo /path/to/photo.jpg

java -cp "target/classes:$OPENCV_JAR" -Djava.library.path="$OPENCV_LIB_DIR" \
  com.idanshaviner.facerecognition.Video /path/to/video.mp4
```

## Known limitations / possible next steps

- Single-person closed set — adding a second known person means retraining
  with a new label and updating `Camera`'s label→name mapping by hand;
  there's no data-driven label↔name mapping file.
- The confidence threshold (`82`) has no documented derivation and hasn't
  yet been validated on real faces — run `Evaluator` on your own data to
  measure it and pick a better value (`Camera.STRICT_THRESHOLD`). Measured on synthetic non-face patterns, LBPH
  distances span roughly 0–250: flat/gradient/checkerboard images scored
  160–251 (rejected), but random noise scored ~57 against label 0 (accepted
  as "idan"). Noise is never fed to the recognizer in practice — only Haar
  detections are — but it shows the threshold alone isn't a strong
  rejector, and that it needs to be tuned on real held-out faces.
- The trained model has only 62 samples (33 "idan", 29 "unknown").
- `Trainer` always uses the *first* detected face in an image
  (`faceDetections.toArray()[0]`), which could pick up a false positive or a
  background face in a group photo.
- Haar cascades are more sensitive to pose/lighting than modern DNN-based
  detectors — a natural upgrade path is swapping in OpenCV's `dnn` face
  detector while keeping the recognition stage unchanged.
- No automated tests. (`Evaluator`'s FAR/FRR/threshold math was checked
  against a hand-computed example, but that check isn't in the repo yet.)

## Interview talking points (quick reference)

- **Two-stage pipeline:** Haar cascade *detects* a face; LBPH *recognizes*
  whose face it is — two different algorithms chained for two different
  sub-problems.
- **Why LBPH, not deep embeddings:** few-shot friendly, CPU-only, small
  interpretable model — at the cost of generalization and per-person
  retraining.
- **Why a threshold sits on top of the classifier:** LBPH can't natively say
  "I don't recognize this" — combining a trained negative class with a
  distance threshold approximates open-set rejection.
- **Train/inference consistency:** found and fixed a real skew bug where
  preprocessing differed between training and live prediction; now
  centralized in one shared function.
- **Data hygiene:** found and fixed a `.gitignore` bug that would have let
  personal training photos slip into git history.
- **Packaging:** moved from an IDE-only, single-machine setup to a portable
  Maven build, and can explain why OpenCV's native Java bindings don't fit
  the usual "just add a Maven dependency" story.
