# FaceRecognition — Interview Study Guide

Goal: after reading this you can explain what the project does, walk through
every file, defend each design choice, name its weaknesses without flinching,
and say what you'd do next.

**How to read it:** sections 1–3 give you the mental model, 4 is the code
walkthrough, 5–7 are decisions/problems/weaknesses, 8 is a Q&A drill, 9 is
hands-on practice. Section 0 matters most — read it first.

---

## 0. Honesty note: what's verified vs. what only you know

Interviewers probe. The fastest way to lose credibility is to state a
guess as fact, so this guide separates three kinds of claims:

| Tag | Meaning |
|---|---|
| **[code]** | Read directly from the source in this repo. |
| **[measured]** | I ran it on your machine and saw the result. |
| **[you]** | Only you know this (why you chose X, what went wrong live). I have not invented an answer — fill these in from your own memory. |

Things I verified by running (headless, no camera): the project compiles with
`javac` and `mvn compile`; the native OpenCV library loads; the Haar cascade
and `models/model.xml` load; `predict()` runs; the LBPH distance scale (§3.3);
the crop helper doesn't corrupt the source frame.

Things I have **not** verified: live webcam behavior, real recognition
accuracy (your `data/` folders are empty, so nothing can be evaluated here),
how the threshold `82` was chosen, and which version of `Trainer` produced the
existing `model.xml`.

**Provenance of the "problems" in §6:** the git history has three tiny
commits and no debugging story. The issues I list were found when the repo
was audited and cleaned up in commit `e99e043`. Present them as "when I
audited the project I found…", which is true. Don't claim you debugged them
live unless you did.

---

## 1. The pitch

**30 seconds:**
> A real-time face recognition app in Java on OpenCV. It reads a webcam,
> finds faces with a Haar cascade, then classifies each face as me or
> unknown with an LBPH recognizer trained on my photos. It also draws an
> alignment guide and tells the user to move so their face is well framed.

**2 minutes (add):**
> It's two stages because detection and recognition are different problems:
> detection asks "is there a face, and where?", recognition asks "whose is
> it?". Training is offline: I collect photos of me and of other people,
> crop each detected face to 200×200 grayscale, and store LBPH histograms
> with label 0 for me and 1 for everyone else. Live, I run the same
> crop-and-resize on each detected face and ask the model for the nearest
> training sample. I only call it "idan" if it says label 0 *and* the
> distance is under a threshold, because LBPH can't say "I don't know" on
> its own.

---

## 2. Architecture at a glance

```
TRAINING (offline)                                   INFERENCE (live)
──────────────────                                   ────────────────
data/idan/*.jpg   (label 0)                          Webcam frame (BGR, up to 1920x1080)
data/unknown/*.jpg (label 1)                                    │
   ▲                                                            ▼
   │ (optional) VideoFrameExtractor                  cvtColor → grayscale
   │  video → every 5th frame → jpg                             │
   │                                                            ▼
   ▼                                                 Haar cascade detectMultiScale
Trainer                                                         │  list of face rectangles
 ├─ imread as grayscale                                         ▼
 ├─ Haar cascade → first face box                   skip if box < 100x100 px
 ├─ FaceUtils.extractFaceROI ──── same code ────►   FaceUtils.extractFaceROI (crop + 200x200)
 ├─ LBPHFaceRecognizer.train(images, labels)                    │
 └─ save → models/model.xml ─────── loaded by ─────►   recognizer.predict → (label, distance)
                                                                │
                                                                ▼
                                                     label==0 AND distance<82 ? "idan" : "unknown"
                                                                │
                                                                ▼
                                                     draw boxes, name, "move left/closer…" → HighGui window
```

The single most important structural idea: **`FaceUtils.extractFaceROI` is
the one place where a face becomes a model input, and both Trainer and Camera
call it.** That's what prevents train/inference skew (§6.1).

Files [code]:

| File | Responsibility |
|---|---|
| `Trainer` | offline: images → LBPH model file |
| `Camera` | live loop: detect → recognize → overlay |
| `FaceUtils` | shared crop + 200×200 resize |
| `VideoFrameExtractor` | video → still frames for training data |
| `Photo`, `Video` | detection-only smoke tests (no recognition) |
| `models/haarcascade_frontalface_alt2.xml` | pretrained **detector** (generic, tracked in git) |
| `models/model.xml` | your **recognizer** (personal, gitignored) |
| `data/idan`, `data/unknown` | training images (personal, gitignored) |

---

## 3. The concepts (know these cold)

### 3.1 Detection vs. recognition
- **Detection**: find where faces are. Output: rectangles. Person-agnostic.
- **Recognition**: given a cropped face, decide whose it is. Output: label +
  score.
- Here: Haar cascade = detection; LBPH = recognition.

### 3.2 Haar cascade (Viola–Jones)
How it detects a face, in the order an interviewer might drill:

1. **Haar-like features**: differences between sums of pixels in adjacent
   rectangles (e.g. eyes region is darker than cheeks; nose bridge is brighter
   than eye sockets). Cheap, position/scale-specific.
2. **Integral image**: precompute cumulative sums so the sum over *any*
   rectangle costs 4 lookups. This is why thousands of features can be
   evaluated per window in real time.
3. **AdaBoost**: from a huge pool of possible features, pick the few hundred
   that best separate faces from non-faces, combining weak classifiers into a
   strong one.
4. **Cascade**: stages of increasing complexity. Early stages are tiny and
   reject most non-face windows immediately; only windows passing every stage
   are a face. Most of the image is discarded in the first stages, which is
   why it's fast.
5. **Multi-scale sliding window**: `detectMultiScale` scans an image pyramid.
   The two params you should know: `scaleFactor` (pyramid step; OpenCV default
   1.1) and `minNeighbors` (how many overlapping hits are needed to keep a
   detection; default 3; higher = fewer false positives, more misses). The
   project uses the defaults **[code]**.

Limits: trained mainly on frontal faces, sensitive to head rotation,
occlusion, and strong lighting. "alt2" is one of several frontal-face
cascades shipped with OpenCV. **[you]** Why alt2 specifically? I can't
verify a reason from the repo. If asked and you don't remember, say it's one
of the standard OpenCV frontal cascades you chose and didn't benchmark against
the others (and then say you'd benchmark them if it mattered).

### 3.3 LBPH (Local Binary Patterns Histograms)
1. **LBP code per pixel**: compare a pixel to its 8 neighbors (radius 1,
   neighbors 8 **[code: model.xml has radius=1, neighbors=8]**). Each neighbor
   ≥ center → bit 1, else 0. Read as an 8-bit number (0–255) describing local
   texture. Because it only uses *comparisons*, it's robust to uniform
   lighting changes — the main reason LBPH is a classic choice for faces.
2. **Grid + histograms**: split the LBP image into an 8×8 grid (64 cells);
   in each cell build a 256-bin histogram of codes; normalize; concatenate.
   Result: one vector of 64 × 256 = **16,384** numbers per face. The grid keeps
   *spatial* information (which region has which texture); a single global
   histogram would lose it.
3. **"Training" = memorization.** LBPH learns no weights. `train()` just
   computes and stores each training image's 16,384-vector plus its label.
4. **Prediction = 1-nearest-neighbor.** Compute the query's vector, compute a
   chi-square histogram distance to every stored vector, return the label of
   the closest one and that distance.
5. **The "confidence" trap.** OpenCV calls the returned value `confidence`,
   but it's a *distance*: **lower = better match**. That's why the code says
   `confidence < 82` to accept.

Facts from your actual model file **[measured]**:
- `<cols>16384</cols>` = 8×8×256, matching the above.
- 62 stored histograms → 62 × 16,384 floats ≈ 1M numbers → why the XML is
  8.9 MB. (Model size scales with training-set size, not "model complexity".)
- Labels: **33 "idan", 29 "unknown"** (a small dataset).
- `<threshold>` in the file is 1.79e308 (≈ infinity): OpenCV's *built-in*
  rejection threshold is disabled, so every query gets some label. Rejection
  only happens in `Camera`'s own `< 82` check.
- Histogram values like `0.00173611` = 1/576, i.e. one pixel in a ~24×24
  cell — consistent with the normalization described above.

Distance scale, measured by feeding non-face patterns through your model:

| Input | Label | Distance |
|---|---|---|
| flat gray | 0 | 251 |
| horizontal gradient | 1 | 245 |
| flat black | 0 | 196 |
| checkerboard | 1 | 159 |
| random noise (5 seeds) | 0 | ~57 |

So the scale is roughly 0–250, and **82 is not obviously "strict"**. Random
noise scored ~57 and would be accepted as "idan". This never happens live
because only Haar-detected faces are fed in, but the takeaway is important:
a threshold picked without validation on real held-out faces isn't trustworthy.
See §7.

### 3.4 The open-set problem (why two classes AND a threshold)
- LBPH is a **closed-set classifier**: it always returns one of the labels it
  was trained on. A stranger's face still gets *some* label.
- Fix used here, two layers **[code]**:
  1. Train an explicit **"unknown" class (label 1)** from other people's faces,
     so strangers usually land nearer label-1 samples.
  2. Add a **distance threshold** so that even a nearest-neighbor of label 0
     is rejected if it's too far away.
- Why both: class 1 handles "looks like a different person"; the threshold
  handles "matches nobody well". Neither alone is sufficient.
- Weakness: "unknown" is defined only by the 29 people/images you trained on;
  faces unlike any of them are decided mostly by the threshold.

---

## 4. Code walkthrough

Package `com.idanshaviner.facerecognition`, Maven layout
`src/main/java/...`. Run everything from the repo root because paths like
`models/model.xml` are relative.

### 4.1 `FaceUtils.extractFaceROI(Mat grayFrame, Rect faceRect)`
```java
Mat faceROI = new Mat(grayFrame, faceRect);          // a *view* into the frame, no copy
Imgproc.resize(faceROI, faceROI, new Size(200, 200)); // resize replaces the view with a new 200x200 Mat
return faceROI;
```
- Why 200×200: LBPH needs same-size inputs (histogram grid must line up), and
  a fixed size makes the model independent of how far away the face was.
  **[you]** why 200 specifically — likely a common LBPH tutorial default.
- `new Mat(gray, rect)` is a view; the resize allocates a new Mat, and I
  checked that the source frame is untouched **[measured: 0 modified pixels]**.
- Called by *both* Trainer and Camera: the whole point of the class.

### 4.2 `Trainer`
Flow **[code]**:
1. `System.loadLibrary(Core.NATIVE_LIBRARY_NAME)` — load the native OpenCV
   library (see §4.6).
2. `loadImages("data/idan", 0, …)` and `loadImages("data/unknown", 1, …)`.
3. For each file: skip non-`.jpg/.jpeg/.png`; `imread(..., IMREAD_GRAYSCALE)`;
   skip if unreadable; run the Haar cascade; skip if no face; take
   `toArray()[0]` (the **first** detection); `FaceUtils.extractFaceROI`;
   add image + label.
4. If nothing loaded → print a message and exit (added in cleanup so it
   doesn't crash on empty folders).
5. `LBPHFaceRecognizer.create()` → `train(images, labelsAsMat)` → `save(...)`.

Decisions/notes:
- Grayscale at load: LBP works on intensity; also matches Camera.
- Label scheme 0/1 is hard-coded; `Camera` also hard-codes "label 0 = idan".
- Uses `[0]` face: simple, but in a group photo or with a false positive it
  may pick the wrong face (§7).
- `Converters.vector_int_to_Mat(labels)` converts Java `List<Integer>` into the
  `Mat` that OpenCV's `train` expects.

### 4.3 `Camera` (the main app), section by section
1. **Open camera** `new VideoCapture(0)` (device 0 = default webcam); bail if
   not opened.
2. **Request 1920×1080.** It's a *request*; the driver may give something else.
   Higher resolution = more detail for the recognizer but slower Haar scanning
   every frame. **[you]** whether you noticed lag; I haven't measured FPS.
3. **Load** the cascade and the model (`recognizer.read("models/model.xml")`).
4. **Per frame:** `cvtColor(BGR→GRAY)` (OpenCV frames are **BGR**, not RGB),
   then `detectMultiScale(gray, faces)`.
5. **Alignment guide (UI only).** A green rectangle 300×350 centered
   horizontally, with its center at 35% of frame height (a face sits in the
   upper part of a frame, not the middle). Purely visual/UX.
6. **For each detected face:**
   - **Skip if < 100×100 px** — tiny detections are unreliable and too little
     detail for LBPH.
   - Red box (BGR `(0,0,255)`).
   - **Positional hint**: if face center is left/right/above/below the green
     box, say "move leftwards/rightwards/downwards/upwards". Left/right is
     checked before up/down, so only one of them shows at a time.
   - **Distance hint**: compare box width/height to an ideal 250×300 ± 75 →
     "move closer / farther". Uses the raw detection size.
   - **Recognize:** `FaceUtils.extractFaceROI(gray, faceRect)` →
     `predict(roi, label[], confidence[])` (Java uses 1-element arrays as
     out-parameters).
   - **Decision:** `label==0 && confidence<82 → "idan"` else `"unknown"`.
   - Console-log label, confidence, name, size (handy for tuning the
     threshold — this is your only debugging aid).
   - Draw name and hints above the box.
7. `HighGui.imshow` then `HighGui.waitKey(2)`: `waitKey` is what actually
   pumps the GUI event loop and repaints; 27 = Esc → break.
8. Cleanup: `camera.release()`, `destroyAllWindows()`.

Things to be aware of **[code]**:
- Only `gray` is released each frame; other per-frame Java `Mat`s (`faceROI`,
  `MatOfRect`) rely on garbage collection to free native memory. Fine for a
  demo, something you'd tidy up for long runs.
- Guidance hints and recognition are independent: hints use the detection box;
  recognition uses the resized crop.
- No temporal smoothing: each frame is decided alone, so the label can
  flicker. Averaging the last N frames is an obvious upgrade.
- Directions in the hints: verify live that "move leftwards" feels right for
  the person standing in front of the (un-mirrored) camera.

### 4.4 `VideoFrameExtractor`
- Walks `data/idan` and `data/unknown`, for each `.mov/.mp4/.avi` reads frames
  with `VideoCapture` and saves **every 5th frame** as
  `frame_<videoName>_<n>.jpg` **into the same folder**.
- Why every 5th: consecutive frames are nearly identical; sampling gives
  variety (pose/expression/lighting drift) without thousands of duplicates.
  **[you]** how you chose 5.
- No face filtering here — Trainer discards frames without a face later.
- Side effect to know: re-running it re-extracts frames from the videos again
  (overwriting same-named frames) and the extracted stills also get fed to the
  Trainer, so keep that in mind when counting samples.

### 4.5 `Photo` and `Video`
Detection-only utilities: open an image/video path (CLI arg), run the cascade,
draw boxes. Use them to sanity-check the detector or a new cascade file
without a trained model or webcam. `Photo` opens an image through
`VideoCapture` (works, because OpenCV treats a still as a 1-frame video) and
`waitKey(0)` blocks until a keypress. `Video` still has an early version's
history: it once rendered via Swing (`Mat → JPEG bytes → BufferedImage →
JLabel`); that dead code was removed in favor of `HighGui.imshow`.

### 4.6 Java + OpenCV mechanics (common trip-up questions)
- OpenCV is C++. Java calls it through **JNI**: the jar holds Java wrapper
  classes; a native library (`libopencv_java4100.dylib`) does the work.
- `System.loadLibrary(Core.NATIVE_LIBRARY_NAME)` loads it by name (resolves to
  `opencv_java4100`). The JVM searches `java.library.path`, hence
  `-Djava.library.path=/usr/local/share/java/opencv4` in the run commands.
  Previously `Camera`/`Trainer` hard-coded `"opencv_java4100"` (version
  baked into code); now version-agnostic.
- The **jar and dylib must be the same build**, and the **dylib must match the
  JVM's CPU architecture** [measured: both arm64 here].
- `Mat` is a Java handle to native memory; `release()` frees it eagerly.
- `MatOfRect` = `Mat` of rectangles; `.toArray()` gives `Rect[]`.
- Colors are **BGR**: `Scalar(0,0,255)` is red, `Scalar(0,255,0)` green,
  `Scalar(255,255,0)` is cyan (the name label).
- Why the pom uses a `system`-scope local jar: OpenCV publishes no official
  Java artifact on Maven Central (community repackagings exist), and the jar
  must match the native lib, so the pom points at your locally built one.
  `system` scope is discouraged in general — an honest trade-off to
  acknowledge.
- Your OpenCV install **[measured]**: native arm64 build in `/usr/local`, version
  `4.10.0-dev`, includes Java bindings and the contrib `face` module. The
  Homebrew OpenCV on the machine has no jar. So this OpenCV was most likely
  built from source. **[you]** confirm that's what you did.

---

## 5. Design decisions (the "why" table)

| Decision | Alternatives | Why this one | Trade-off |
|---|---|---|---|
| Haar cascade detector | DNN detectors (OpenCV YuNet, MTCNN, RetinaFace) | Ships with OpenCV, tiny, CPU-fast, no model download, fine for one frontal face near a webcam | Weak on rotation/occlusion/lighting; more false positives |
| LBPH recognizer | Deep embeddings (FaceNet, ArcFace, OpenCV `FaceRecognizerSF`); Eigenfaces/Fisherfaces | Works from few images, CPU-only, instant "training", readable model, robust to uniform lighting shifts | Doesn't generalize; retrain per person; model grows with samples; far less accurate than embeddings |
| Explicit "unknown" class + threshold | Threshold only; embedding distance to enrolled vectors | LBPH has no reject option; two layers approximates open-set | "Unknown" limited to the 29 samples; threshold untuned |
| Same crop function for train & inference | Separate crops | Eliminates train/inference skew | Removed the inward "hair-trimming" crop Camera used to do |
| Grayscale 200×200 | Color; other sizes | LBP needs intensity only; fixed size for fixed histogram grid | Discards color cues (irrelevant to LBP anyway) |
| Min face 100×100 in Camera | No filter | Small boxes are noisy and low-detail | Won't recognize far-away people |
| Every 5th video frame | Every frame; random | Near-duplicate frames add cost, not information | 5 is arbitrary **[you]** |
| `models/model.xml` gitignored | Commit the model | Derived from your face; regenerate with Trainer | Fresh clone can't run `Camera` until trained |
| Maven, local OpenCV jar | Maven Central repackaging (e.g. `org.openpnp:opencv`), IDE-only `.iml` | Guarantees jar + native lib match and include `face` | `system` scope is non-portable; must set `opencv.jar` |
| Standard `src/main/java` + package | Flat `src/` with lowercase classes | Convention; tool-friendly; clearer | — |

---

## 6. Problems and how they were handled

### 6A. Found and fixed in the audit (commit `e99e043`) — all real, all in git

**1. Train/inference skew (the best story).**
- *What:* `Trainer` fed the model the raw Haar box; `Camera` cropped inward
  (removed ⅛ of the width from each side and 1/6 of the height from the top
  and bottom, keeping the central 3/4 × 2/3) before predicting. LBPH compares
  spatial histograms, so a differently-framed crop at prediction time means
  every grid cell is looking at different facial content than it did in
  training.
- *Impact:* silent accuracy loss — no crash, no error, just worse matches.
- *Fix:* one `FaceUtils.extractFaceROI` used by both. I aligned inference to
  training (dropped Camera's inner crop) rather than changing training,
  because that keeps the existing `model.xml` valid without retraining.
- *Caveat to say out loud:* I preserved the crop the *current* Trainer code
  uses; I can't prove the existing `model.xml` was made by exactly that code.
  If recognition feels off, retraining with the current Trainer is the clean
  test.
- *Lesson:* preprocessing must be a shared function, not duplicated code.

**2. `.gitignore` typo would have leaked personal photos.**
- *What:* rule was `data\` (backslash) instead of `data/`. Git treated it as
  a non-matching pattern.
- *How found:* created a test file and `git status` showed it as untracked,
  not ignored. Verified the fix with `git check-ignore -v`.
- *Fix:* `data/` rules with negations for `.gitkeep` so the folders exist in
  the repo but photos never do.
- *Lesson:* test ignore rules; a commit message saying "ignore private data"
  isn't proof it works.

**3. Orphaned 8.9 MB `model.xml` in the repo root.** Code read
`models/model.xml`, but the tracked file sat in the root. Moved it to the path
the code uses and stopped tracking it. (It remains in git history; repo is
private. Purging history would need a rewrite and force-push — not done.)

**4. Committed build/IDE/unused files.** `.class`, `.idea/`, `.iml`, and an
unused `ring-client-api` npm manifest (no code referenced it) were removed.

**5. Inconsistent, version-pinned native loading** → unified on
`Core.NATIVE_LIBRARY_NAME`.

**6. Hard-coded personal paths** in `Photo`/`Video` → CLI arguments, and both
now use the repo-relative cascade path like the other classes.

**7. No portable build** → Maven `pom.xml`, `mvn compile` verified.

### 6B. Things I said in early drafts that were wrong (good "I verify my work" story)
While preparing this guide I re-checked my own README and corrected two
claims: I'd said the OpenCV came from an Intel/Rosetta Homebrew (it's a native
arm64 build, most likely from source) and that LBPH distances span 0–128
(measured: up to ~250). Worth remembering that *you* should verify anything
you plan to state as fact.

### 6C. Your real war stories — **[you]** fill these in
The repo can't tell me what actually went wrong while you built this. Prepare
one or two honest ones. Prompts:
- Getting OpenCV's Java bindings + `face` module working. What did you try?
- `UnsatisfiedLinkError` / `java.library.path` problems?
- Camera permission on macOS, or picking camera index 0 (you also have an
  iPhone Continuity Camera on this machine)?
- Collecting training data: how many photos, what lighting, how did you get
  "unknown" faces?
- False positives (strangers labeled "idan") or misses (you labeled "unknown"):
  how did you land on the threshold 82?
- Why you built the alignment-guide overlay.
- Why you extracted frames from video instead of taking photos.

---

## 7. Weaknesses — say them first, then say the fix

| Weakness | Evidence | What you'd do |
|---|---|---|
| **No evaluation** | No held-out test, no accuracy/FAR/FRR numbers | Hold out ~20–30% or k-fold; plot FAR vs FRR across thresholds; pick the threshold at the equal-error rate or a target false-accept rate |
| **Threshold untuned/loose** | 82 on a ~0–250 scale; noise scored ~57 | Derive from the evaluation above; never accept "idan" on distance alone — also require a margin over the best label-1 distance |
| **Tiny dataset** | 62 samples (33/29) | More, varied images: lighting, angle, glasses, expression; augmentation (flip, brightness) |
| **Weak "unknown" class** | Only 29 samples | Diverse public face dataset for the negative class |
| **No liveness/anti-spoofing** | A photo of you on a phone can match | Liveness cues (blink, depth), or don't use it for security |
| **First face only in Trainer** | `toArray()[0]` | Pick the largest detection; skip images with ≠ 1 face |
| **Haar detector limits** | Pose/lighting sensitive | Swap to a DNN detector (OpenCV YuNet); recognition stage unchanged |
| **LBPH ceiling** | Not an embedding model | `FaceRecognizerSF` (SFace) or dlib/ArcFace embeddings + cosine distance; scales to many people without retraining a classifier |
| **No face alignment** | Rotated/tilted faces differ | Align by eye landmarks before cropping |
| **No lighting normalization** | — | `equalizeHist` or CLAHE before LBPH |
| **Per-frame decisions** | Label can flicker | Majority vote over last N frames |
| **Hard-coded labels/thresholds** | Constants in source | Config file with label→name map |
| **No automated tests** | — | Unit-test `FaceUtils`; a golden-image test for Trainer→Camera consistency |
| **Perf unmeasured** | 1080p Haar every frame | Downscale for detection, upscale boxes; measure FPS |
| **Native memory hygiene** | Only `gray` released | Release/try-with-resources for Mats |
| **Privacy/bias** | Biometric data; small-sample | Keep data local (already gitignored); consent; test across demographics before any real use |

---

## 8. Likely interview questions, with answers

**Walk me through how it works.** → §1 pitch, then §2 diagram.

**Why two algorithms?** Detection locates faces person-agnostically; recognition
needs a cropped, normalized face and a trained model. Different problems, so
chain them.

**How does a Haar cascade work?** Haar features via integral images; AdaBoost
picks discriminative features; a cascade of stages rejects non-faces early; run
over an image pyramid.

**How does LBPH work?** LBP code per pixel from 8 neighbors; 8×8 grid of
256-bin histograms concatenated (16,384-dim); training just stores the
vectors; prediction is 1-NN by chi-square distance.

**Why LBPH and not a neural network?** Few-shot, CPU-only, instant training,
inspectable model, fine for one person. Cost: weak generalization and
retraining per identity. For real use I'd use embeddings (SFace/ArcFace).

**What does "confidence" mean here?** A distance; lower is better. It's a
naming quirk in OpenCV's API.

**How do you handle unknown people?** LBPH always returns a known label, so I
train a second "unknown" class and add a distance threshold. Honest caveat: the
threshold isn't validated.

**How did you pick 82?** **[you]**. If it was trial and error, say so: "I tuned
it by watching live distances printed to the console; I haven't done a proper
ROC analysis, that's the next step." (Only say this if true.)

**How would you measure accuracy?** Hold-out or k-fold split; compute false
accept and false reject rates across thresholds; choose an operating point.
For security-type use, prioritize a low false-accept rate.

**What's train/inference skew and did you hit it?** Different preprocessing at
train vs. predict time. Yes — Trainer used the raw box, Camera used an inward
crop. Fixed by sharing one function. (Say "I found it in an audit" per §0.)

**Why crop and resize to 200×200 grayscale?** Focus on the face, fixed
histogram grid, LBP needs only intensity, distance independence.

**What breaks it?** Head turn, glasses on/off, low light, a photo of you, a
very different camera than training. Fixes in §7.

**Can it recognize multiple people?** Not without retraining and adding labels;
code hard-codes label 0 = idan. Embedding approach scales better.

**Why is model.xml so big?** LBPH stores one 16,384-float histogram per
training image: 62 × 16,384 ≈ 1M floats → 8.9 MB XML.

**Why not commit model.xml or data?** They're derived from/are biometric data;
regenerate with Trainer; keeps the repo small and safe.

**Why Maven with a system-scope jar?** OpenCV has no official Maven Central
Java artifact and the jar must match the native lib. It keeps things consistent
but isn't portable; alternatives are a community repackaging or documented
manual setup.

**What's `java.library.path` for?** Where the JVM looks for the native OpenCV
library when `System.loadLibrary` runs.

**What would you change first?** Build an evaluation harness (hold-out + FAR/FRR)
and tune the threshold from data. Then temporal smoothing and a DNN detector.

**Security implications?** No liveness detection, so spoofable with a photo;
LBPH is a demo-grade recognizer; biometrics need consent and secure storage.

**What did you learn?** **[you]** Suggested angles: preprocessing consistency
matters as much as the algorithm; validating assumptions (the gitignore
typo); the gap between "runs" and "measured accuracy".

---

## 9. Hands-on practice (do these before the interview)

Run from the repo root. Set once:
```bash
export OPENCV_JAR=/usr/local/share/java/opencv4/opencv-4100.jar
export OPENCV_LIB_DIR=/usr/local/share/java/opencv4
mvn -q compile -Dopencv.jar="$OPENCV_JAR"
```

1. **Live test (the one thing I couldn't do):**
   ```bash
   java -cp "target/classes:$OPENCV_JAR" -Djava.library.path="$OPENCV_LIB_DIR" \
     com.idanshaviner.facerecognition.Camera
   ```
   Watch the console: what distances do *you* get vs. a friend? That's the
   real basis for defending the threshold. Note where "idan" flips to "unknown".
2. **Detector-only:** run `Photo` on a group photo; see how many faces it finds
   and any false positives.
3. **Retrain from scratch:** put photos in `data/idan` and `data/unknown`, run
   `Trainer`, then compare live distances against the old model. This also tests
   the crop-consistency fix.
4. **Experiment with the threshold:** change 82 → 60 and → 100, rebuild,
   observe false accepts/rejects.
5. **Reproduce skew on purpose:** temporarily make `FaceUtils` crop inward in
   `Camera` only, watch distances worsen, revert. Being able to demonstrate the
   bug is more convincing than describing it.
6. **Read the model:** open `models/model.xml`, find `grid_x`, the 16384-wide
   histograms, and the `labels` matrix.
7. **Write the evaluator** (great talking point): a small class that splits
   `data/` into train/test, trains, and prints FAR/FRR per threshold.

### Self-quiz (answer without looking)
- What is the output of detection? Of recognition?
- What does `detectMultiScale` do, and what do `scaleFactor`/`minNeighbors` control?
- What's in the 16,384 numbers?
- Why is "confidence" backwards?
- Why can't LBPH say "unknown" by itself, and how does this project work around it?
- What is train/inference skew, and where does it live in this code?
- Why must the OpenCV jar and dylib match, and match the JVM's architecture?
- Name three weaknesses and the fix for each.

### Number cheat sheet
200×200 face size · 8×8 grid · 256 bins · 16,384 features · radius 1 /
8 neighbors · 62 training samples (33 idan / 29 unknown) · threshold 82 ·
observed distance scale ~0–250 · min face 100×100 · ideal box 250×300 ±75 ·
guide box 300×350 centered at 35% height · request 1920×1080 · every 5th
video frame · Esc = key 27 · `waitKey(2)` · labels: 0 = idan, 1 = unknown.
