# Texas Wildlife Camera — ISAN 3392

A local FastAPI application with a custom HTML, CSS, and plain JavaScript
dashboard. Debug tests webcam, screen sharing, and recordings; Demo presents four
independent replay cameras. YOLO11n remains the lightweight default. No cloud
service, account, database, frontend framework, Node.js, or Docker is needed.

## How the project fits together

`app.py` serves `templates/index.html` and the files in `static/`, accepts video
uploads, and manages one WebSocket dashboard connection. JavaScript captures
browser media, samples/resizes JPEGs, and displays returned results. It does not
make ML decisions. `wildlife/runtime.py` schedules sources and uses one worker
thread and one shared YOLO model for sequential inference. Every camera has its
own existing `AlertGate`, replay position, cooldown, and error state.

`wildlife/detection.py` still imports the official Ultralytics package and uses
PyTorch's GPU/CPU selection. It adds optional structured box results while
preserving the original two-result API used by `detect.py` and the course
notebook. `wildlife/events.py` retains CSV/snapshot logging and adds camera names.
`wildlife/sources.py` retains OpenCV replay and adds looping. The older direct
OpenCV webcam helper remains tested, but the new dashboard captures live media
in the browser. `wildlife/protocol.py` validates settings and formats messages.

Training, annotation checking and test-set evaluation remain separate scripts.
The interface runs inference on learned weights; it never trains on browser
frames. Keep the app stopped during training to avoid sharing scarce GPU memory.

## Stage 1 — minimal working detection

Run PowerShell in this project folder. Use regular 64-bit CPython 3.13 (not the
experimental free-threaded `3.13t` build). Python 3.13 is installed on this laptop.
No environment activation or PowerShell execution-policy change is required:

```powershell
py -V:3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
nvidia-smi
```

The RTX 4050 is CUDA-capable. `nvidia-smi` confirms the driver can see the GPU;
it does not prove Python can use it. The displayed CUDA version is the driver's
supported level, not a requirement to install that exact toolkit.

Install the following matching GPU wheels from PyTorch's official index, then
the project dependencies. These pinned versions intentionally keep the course
project stable instead of automatically moving to newer releases:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe check_environment.py
```

The wheel pair and CUDA 12.6 command are from the official
[PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/).
For a different driver/Python version, use the official
[Windows/Pip/CUDA installation selector](https://pytorch.org/get-started/locally/)
and keep torch/torchvision as a matched pair. Update the pins if you choose a
different pair. Standard inference with these wheels does not require installing
the CUDA toolkit separately. If CUDA fails, check the NVIDIA driver and selected
Python environment. A CPU fallback is available:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_environment.py
```

`check_environment.py` prints installed versions, CUDA availability and a real
GPU tensor test. It records `outputs/environment.json` and the complete installed
package list in `outputs/requirements-working.txt`. Keep these with your course
report. See `VALIDATION.md` for what was actually checked during development.
The verified environment is also saved in `environment-verified.json` and the
complete application dependency snapshot in `requirements-lock.txt`.

Put your own image and saved video in `data/` (create the folder first). Replace
the example filenames with real files:

```powershell
New-Item -ItemType Directory -Force data
.\.venv\Scripts\python.exe detect.py "data/cow.jpg"
.\.venv\Scripts\python.exe detect.py "data/trail_recording.mp4" --fps 3 --confidence 0.5
```

The first run downloads official `yolo11n.pt`; internet is needed once for weights
and package installation. Later prediction is local. Stage 1 writes an annotated
JPG or sampled MJPEG AVI under `outputs/demo/`; it does not create event alerts.
OpenCV handles decoding and video output without loading the video into RAM.
The AVI shows `REPLAY` and relative footage seconds. Sampling is by footage time;
offline processing can take longer or shorter than the recording duration.

[YOLO11n](https://docs.ultralytics.com/models/yolo11/) is the small nano detector.
Its pretrained COCO model recognizes 80 classes including `cow`, but **does not
include deer or wild boar**. A deer predicted as horse is an error, not a deer
detection. Pretrained inference alone cannot establish accuracy for Texas wildlife.

## Stage 2 — local dashboard

For an existing installation, remove the previous interface and install the new
dependencies (the existing CUDA PyTorch installation remains in place):

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y streamlit
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe check_environment.py
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in a current desktop Edge, Chrome, or Firefox
browser. Use one server worker and one dashboard tab. A second WebSocket is
rejected to keep inference and CSV writes under one owner. Press Ctrl+C in the
server terminal to shut down cleanly. Do not bind to `0.0.0.0`; trusted file
selection is intended for your local machine, not a public server. No external
fonts, scripts, CDNs, or build tools are used.

For stricter WebSocket transport limits, the equivalent command is:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000 --ws-max-size 600000 --ws-max-queue 1
```

### Debug tab

1. Select **Browser webcam**, **Browser screen share**, or **Uploaded video**.
2. For webcam, select a camera if listed. **Refresh camera list** enumerates
   cameras without opening them; browsers may hide names until permission is
   granted. The list refreshes after a successful webcam start.
3. For a recording, select MP4, AVI, MOV, or MKV up to 200 MB. The Python server
   decodes it with OpenCV, so browser-native playback codec support is unnecessary.
4. Select `yolo11n.pt` or enter a trusted detection model path, for example
   `runs/train/wildlife/weights/best.pt`. Paths are relative to the project folder
   unless absolute. Only load weights you trust. URLs are not accepted for custom
   weights or demonstration video paths.
5. Choose confidence (default 50%), detection rate (1–10 FPS, default 3),
   consecutive confirmations (default 3), and per-class cooldown (default 30 s).
6. Press **Start detection**. Live sources use the browser's normal permission
   prompt. Screen sharing also lets you choose the tab, window, or screen.
7. Press **Stop** before changing settings. Changing tabs also stops the current
   run. **Clear** only clears the displayed feed; saved history is preserved.

Webcam uses `getUserMedia()`; sharing uses `getDisplayMedia()`. Neither captures
audio. The browser keeps the original preview, with annotated detections in the
main monitor. All media tracks stop on Stop, source change, page exit, socket
closure, or the browser's own End sharing action. If permission is declined or
the picker is cancelled, the page shows an error and you can try Start again.
These APIs require a secure context; localhost is supported for local use.
Screen sharing availability and permission behavior vary by browser. See
[MDN webcam permissions](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
and [screen sharing](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getDisplayMedia).

The monitor displays bounding boxes, original labels and scores, actual measured
processing FPS, inference time, source status, local processing time, and replay
footage time. The first frame reports warming up rather than an invented FPS.
An idle browser preview may keep playing at its native rate while detection runs
at only 3 FPS. Annotated frames are sampled stills, not full-rate video streaming.

### Demo tab

The responsive two-by-two grid represents **North Trail**, **Creek Crossing**,
**Feeder Area**, and **Main Gate**. Each feed is explicitly marked **REPLAY**.
For each camera, select an uploaded recording or type a trusted local video path
such as `C:/Videos/north.mp4`. Uploads take priority when both fields are filled.
Clear a file selection to return to path input. Local paths avoid uploading large
files; the server still reads frames individually rather than loading the whole
video into RAM. Cameras without a selection stay empty.

Choose model, display mode, confidence, detection rate (1–3 FPS per camera,
default 1), confirmations, cooldown, and optional looping. **Start all cameras** starts selected feeds.
**Stop all cameras** releases them all. A bad recording stops only its own feed;
the other feeds continue. Each panel shows its own status, video/image, animal
scores, detection FPS/inference time, and the most recent detection's footage time.

**Live view (default)** uses the browser's native HTML video playback at normal
speed, typically around 24–30 visible FPS for footage recorded at those rates.
It is raw footage with **no bounding boxes**, and still carries a REPLAY badge.
Live view does not mean a live physical trail-camera connection. Its LIVE VIEW
badge describes the display mode. Muted inline playback starts from the Start
button workflow and follows Loop videos. Browser codec/decoder support determines
whether a recording plays smoothly; this has not yet been physically verified.

**Detection view** preserves the annotated JPEG-frame display. Its visible frame
updates follow Python's detection rate. Stop before changing display mode; the
selector is disabled during startup, monitoring, and shutdown. The raw video and
annotated JPEG are never displayed together.

Visible playback FPS and **Detection FPS** are intentionally separate. Python
still performs only the configured 1–3 samples per second per Demo camera through
one shared sequential worker. A 30 FPS visible video does not request 30 FPS
inference. Alert confirmation, per-camera cooldowns, CSV entries and annotated
snapshots still run in Python, even though the Live view display has no boxes.
Metadata and notifications continue to update with class, confidence, camera,
Python footage timestamp, local processing timestamp and snapshot link.

For uploaded files, the browser plays an object URL referencing the user's File
while the existing upload copy supplies Python detection. For trusted local paths,
Python registers an opaque, unpredictable media token only after selecting and
validating the video. `/api/media/{token}` supports byte ranges and HEAD for
playback/seeking. It cannot retrieve arbitrary filesystem paths or unselected
files; URLs contain no filesystem paths. Tokens expire on source cleanup, Stop,
disconnect, and shutdown. The endpoint is a local session capability, not a
general-purpose file server or a public sharing link.

Live view waits for browser media readiness (up to eight seconds per feed), then
acknowledges readiness over the existing socket. Python resets all replay clocks
to one near-future start epoch; the browser starts the selected videos together
as closely as practical. No second detection timer or WebSocket is created.
Server messages also carry a current playback position for occasional correction
when drift exceeds one second, accounting for loop boundaries and inference
delay. This display clock is separate from the sampled footage timestamp used by
alerts. Minor drift, particularly with variable-frame-rate footage, is possible.

A browser playback/codec/autoplay error names the affected camera and leaves
Python detection and unrelated videos running. Stop, select **Detection view**,
and restart for the OpenCV/JPEG fallback. A file that OpenCV can decode may still
be unsupported by the browser, especially AVI, MOV or MKV variants.

Stop releases video elements and clears their sources, cancels readiness/start
timers, revokes object URLs, expires server media references, and follows the
existing upload/source/socket cleanup. Replacement, tab change, socket failure,
and page exit also release browser resources. Looping reuses the same video,
URL/token, upload and socket; it does not register a new session each cycle.

While a training process is running, leave it alone and do not launch the
dashboard or inference. After training finishes, the local-server command is:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` then. No server or browser was started to verify this
change during the active training run.

If Demo cameras stay at **Ready** with no frames, check versions before changing
video paths. A cached older client can omit `display_mode`; an implicit Live
default then waits for `play`, which that older client never sends. This affects
both apparent selector choices. Current code rejects missing mode requests and
stops visibly when the server's returned mode differs from the requested one.
Badges also synchronize from restored select values on initialization/pageshow.

For manual verification after training finishes:

1. Stop the dashboard cameras. In the **Uvicorn terminal only**, press Ctrl+C and
   wait for shutdown. Do not interrupt a training terminal or kill all Python
   processes. Python backend edits do not reload under the command below.
2. From this project folder, start a fresh backend:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
   ```

3. Confirm the new backend is loaded without starting detection:

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8000/api/health | Select-Object status, playback_protocol
   ```

   Expect `ready` and `demo-display-mode-1`. A missing marker means the wrong or
   old backend is serving the port. Editing Python files or refreshing the page
   alone cannot update an already imported server module.
4. Close other dashboard tabs. Open `http://127.0.0.1:8000`, press F12, enable
   **Network → Disable cache**, and use **Ctrl+Shift+R**. All three static requests
   (`demo-playback.js`, `app.js`, `styles.css`) must use `?v=live-view-3`, return
   successfully, and contain current code. The app now sends `Cache-Control:
   no-store` for the page and static assets. Review Console errors for undefined
   `DemoPlayback`, missing elements, media/codec errors, socket failures, or
   rejected play promises. Enable Verbose/Debug messages to see `[Demo]` traces.
5. Test one camera with a trusted local browser-compatible MP4. While stopped,
   selecting **Detection view** must immediately change badges/notes. Start it.
   In Network → WS → `/ws` → Messages, expect client `start` with
   `display_mode: "detection"`, server `started` echoing `detection`, then `frame`
   messages. **No client `play` is needed.** Annotated JPEGs and timestamps should
   update. The terminal logs immediate scheduler startup.
6. Stop, choose **Live view**, and restart the same source. Expect `start(live)`
   → `started(live)` → client `play` → server `playing` → repeated `frame`.
   Local-path `started` data includes an opaque `media_url`; its range requests
   should return 206 when a Range header is used. Console traces show `prepare`,
   `prepared`, `play`, and `playing`. Raw video should appear; sampled detection
   text and event timestamps still come from Python. A media loading failure must
   name its camera and resolve preparation instead of blocking detection forever.
7. Repeat both modes with an uploaded file. Uploads use a `blob:` URL in the
   browser and therefore correctly have no server `media_url`. Finally test
   looping, Stop/restart, alerts, snapshots and multiple cameras.

Startup traces contain modes, camera IDs and media-registration booleans, not
filesystem paths or media tokens. The matching-version flow is tested with mocks;
actual browser loading, codec support and visible playback still need these manual
checks. Read-only access to an existing localhost server timed out during debugging,
so its prior restart/browser-cache state could not be confirmed.

All feeds share one model and run inference sequentially. Four cameras do not
launch four GPU jobs. Actual detection FPS can be lower than the requested rate. Each feed
has separate confirmation/cooldown state; a cow in North Trail cannot confirm
an alert in Creek Crossing. Looping rewinds displayed footage time but maintains
a continuous event clock so cooldowns keep working across loops. Replay may
generate another event on later loops; it is never presented as a live sighting.

The combined newest-first feed retains the latest 20 events, with animal, camera,
confidence, replay footage time, local processing time, and snapshot link. Each
tab keeps a separate displayed feed. Clearing it does not delete files.

### Frame sampling, alerts, and files

JavaScript resizes live frames to at most 960 pixels on the longest side and
encodes JPEG at moderate quality. It sends the next sample only after the prior
response, dropping sampling opportunities if inference is slow. Python has a
single replaceable pending frame per browser source and discards frames older
than two seconds. Uploaded/local recordings seek to the latest playback time,
skipping stale frames. The one-worker scheduler performs inference, source reads,
and event writes sequentially. No complete videos or growing frame queues are
retained in Python RAM. Sampling can miss brief animals between frames.

An alert requires the same animal class in consecutive processed frames. Missing
classes reset counts. Cooldowns apply per class, not per animal identity: multiple
cows can become one event, while one lingering cow may alert again later. State
resets on Start. These rules are the preserved Python `AlertGate`, not learned
CNN behavior. Live alerts name a LIVE source and processing clock time; replay
alerts explicitly name REPLAY and relative footage time.

**Processing time** is the laptop's local wall clock with timezone. **Footage
time** is seconds from the recording's start, not its original capture date/time.
The app never infers recording dates. Variable-frame-rate recordings have
approximate timing based on OpenCV's reported FPS; use constant-FPS recordings
when timing precision matters.

Each confirmed class event writes one annotated JPG to `outputs/snapshots/` and
appends a CSV row to `outputs/events.csv`. Fields include unique event ID,
processing timestamp, LIVE/REPLAY source, footage seconds (blank for live),
animal label, confidence (0–1), model path, snapshot path, and `camera_name`.
Existing logs automatically gain a camera column without losing old rows;
older events are labeled Debug because they predate the four-camera interface.
Snapshots are linked through a dedicated JPG route; arbitrary local files are
not served. An event snapshot may contain several boxes while its row describes
one class. Delete old outputs manually when disk space is low.

Stop/disconnect/shutdown closes every OpenCV source and removes temporary uploads
created by the current run. Trusted originals, CSV history, and snapshots remain.
The chosen model stays loaded across Stop/Start for reuse. Restart the server if
you overwrite weights at the same path. Stop closes the WebSocket; the next Start
opens a fresh connection and reuses the server's loaded model. An abrupt process
kill/power loss can leave copies in `outputs/uploads/`; remove those manually with
the server stopped.

### Troubleshooting and limits

- **Page does not connect:** check the terminal, local URL and port. If port 8000
  is busy, stop the other server or use `--port 8001` and open the matching URL.
- **Another dashboard connected:** close other dashboard tabs, then reload.
- **No cameras / permission declined:** use localhost, refresh cameras, close
  other camera apps, and review browser/Windows camera permissions yourself.
  Screen sharing must start from the Start button, not automatically on page load.
- **Video error:** check its path and codec. MP4/H.264 or MJPEG AVI are useful
  starting formats; actual decoder support depends on OpenCV. Not every MOV/MKV
  encoding works. Convert an unsupported or damaged recording.
- **Model error:** verify the file exists and is a trusted YOLO detection model.
  Pretrained YOLO11n includes cow but no deer or wild boar. Never rename a horse
  prediction to deer. Use a genuinely trained three-class `best.pt` for wildlife.
- **Slow / GPU memory error:** stop training and other GPU programs; use YOLO11n,
  3 FPS Debug or 1 FPS per Demo camera. Reduce selected camera count if needed.
  CPU fallback works but can be much slower. Requested rates are targets.
- **Save failed:** ensure `outputs/` is writable and disk has space. Errors stop
  the affected feed and show a short message; technical details stay in the terminal.
- **Frozen display after stopping:** the last annotated image remains visible;
  source status says Stopped. It is no longer being processed.

Confidence is a model score, not measured wildlife accuracy. This demo does not
identify individual animals, guarantee sightings, send external notifications,
or establish performance without representative held-out labeled data. Physical
webcam/screen-sharing operation, browser console/layout checks and four-video GPU
performance must still be checked; see `VALIDATION.md` for exact tested scope.

## Stage 3 — custom supervised training

### Prepare the downloaded datasets first

Preparation uses CPU file/image operations and does **not** train a model:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py
```

The script reads `data/deer`, `data/boar`, and `data/cow` without changing them.
It creates `data/wildlife_dataset` with these exact detection classes:
`0: deer`, `1: wild_boar`, `2: cow`. The pretrained model's existing `cow`
label still works in the interface. The old custom `cattle` class is no longer
accepted by the dataset checker; do not use an old differently named model for
evaluation of the new dataset.

| Class | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| deer | 231 | 50 | 50 |
| wild_boar | 231 | 50 | 49 |
| cow | 231 | 50 | 50 |

These are image counts, not animal/box counts. An image can contain several
animals. Dog/human-only boar-source images are excluded; mixed images retain
only their boar boxes. Source class 0 is remapped to the correct final ID.
Existing detection boxes are preserved, while polygon vertices become enclosing
rectangles using their minimum/maximum normalized x/y coordinates. Invalid
annotations reject the image rather than quietly dropping a target animal.
Only tiny floating-point boundary errors are clamped.

Seed 42 makes selection repeatable for unchanged input files. Original validation
and test images stay in their source split where practical, with excess images
left out rather than moved into training. Deer keeps its 7 validation and 3 test
images and assigns 43 and 47 unaugmented training images to the corresponding
new holdouts. This is copying into a new split, never moving source files.

Exact SHA-256 duplicates are removed before selection. Cow has augmented training
variants, so only one export per pre-`.rf.` filename family is selected, with
source test/validation taking priority over training. This also prevents known
cross-split filename families from leaking into training. These checks cannot
infer every related video frame or camera visit. Review those relationships
manually before claiming independent test performance.

The output includes short JPEG/label names, `dataset.yaml`, a per-image
`source_manifest.csv`, a detailed `dataset_summary.json` listing every excluded
image and reason, copies of the source attribution/license files, and annotated
`contact_sheets/train.jpg`, `val.jpg`, and `test.jpg`. The manifest records source
paths, splits, original/final classes, licenses, URLs, and source-image hashes.
Licenses are CC BY 4.0 for deer/cow and Public Domain for boar, as stated in the
download metadata. Retain the attribution files when sharing the prepared data.

Inspect the contact sheets before training: rectangular boxes enclosing a
polygon include some background, and conversion does not fix an incorrect
original annotation. Short generated names avoid the source cow download's long
Windows paths; the reader uses Windows extended paths without renaming originals.

An existing output is refused unless you explicitly run:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py --rebuild
```

Rebuild only replaces a folder carrying this script's ownership marker. It
builds and validates a separate staging copy before replacing the previous
generated output. Unmarked folders and output symlinks/junctions are refused.
The original downloads, event logs, snapshots, and model weights are not replaced.

Run the same dataset checker used by training without starting a training run:

```powershell
.\.venv\Scripts\python.exe -c "from wildlife.dataset import check_dataset; check_dataset('data/wildlife_dataset/dataset.yaml', ('train', 'val', 'test'), require_all_classes=True); print('Dataset checks passed.')"
```

### Train later, after reviewing the generated data

Do this after Stage 1 and 2 work. Stop the interface while training so both do not
compete for the 6 GB GPU. No dataset or performance claims are supplied.
Collect lawfully usable, representative Texas wildlife images across lighting,
distance, seasons, and backgrounds. Manually annotate every target animal's
bounding box, including partially visible animals when identifiable. Check class
definitions and annotation quality before training.

Create a dataset outside or inside `data/` with this layout:

```text
wildlife_dataset/
  images/train/     images/val/     images/test/
  labels/train/     labels/val/     labels/test/
```

For `images/train/visit01_001.jpg`, create `labels/train/visit01_001.txt`.
Each animal has one line in Ultralytics detection format:

```text
class_id x_center y_center width height
```

Values are space-separated. IDs are `0=deer`, `1=wild_boar`, `2=cow`. Divide
horizontal coordinates and width by image width, and vertical coordinates and
height by image height. Coordinates and sizes are normalized to 0–1. For example,
`0 0.5 0.5 0.2 0.4` describes a hypothetical centered deer box 20% of image width
and 40% of image height; this is a format illustration, not supplied training data.
Use an **empty** `.txt` only for a manually verified negative image. A missing
label file is an error here. An image folder alone is not a detection dataset.
See the official [label format](https://docs.ultralytics.com/datasets/detect/).

Split groups before training, for example roughly 70% train / 15% validation /
15% test while preserving examples of all classes in every split. Keep the entire
source video, burst, animal encounter, and closely related camera visits in one
split. Never randomly scatter nearby video frames across splits. Prefer holding
out whole locations/visits when possible. Keep a small manual CSV inventory of
source group, image filename, and assigned split for your report. Split fractions
are guidelines; independent, representative examples matter more than exact sizes.
The checker detects byte-identical images across splits, but cannot detect similar
frames or prove independence. You must check groups yourself.

The preparation script already writes a usable YAML with the absolute generated
dataset path. The scripts default to that file. When you are ready to train:

```powershell
.\.venv\Scripts\python.exe train.py --data data/wildlife_dataset/dataset.yaml --imgsz 640 --batch 4 --epochs 20
```

The script requires explicit valid annotations and all classes in train/val.
It initializes YOLO11n from pretrained weights and learns from your labeled boxes.
Defaults are 640 pixels, batch 4, 20 epochs, no dataset RAM cache, zero loader
workers, and automatic GPU/CPU selection. Use `--device cpu` to force CPU. If CUDA
runs out of memory, stop other GPU programs and try `--batch 2` or `--batch 1`;
then consider `--imgsz 512`. Training is much heavier than 3 FPS inference. A fixed
seed improves repeatability but does not guarantee identical results on all devices.

Ultralytics prints the run directory, normally `runs/train/wildlife/`. Repeated
runs may use `wildlife2`, etc. Select its `weights/best.pt` in the interface.
`best.pt` is selected using validation performance, not test performance.

After deciding training settings using training/validation data, evaluate your
chosen model on the **untouched test set**:

```powershell
.\.venv\Scripts\python.exe evaluate.py --data data/wildlife_dataset/dataset.yaml --model "runs/train/wildlife/weights/best.pt" --batch 4
```

The checker requires test annotations for all three classes. Evaluation explicitly
uses `split="test"`, not the validation default. Results and plots go under the
printed `runs/test/` folder, including `per_class_metrics.csv` with precision,
recall, mAP50, and mAP50–95 for each class. Missing metrics are marked N/A, never
replaced with an overall class average. Precision asks how many predicted animals
are correct; recall asks how many labeled animals were found. IoU measures box
overlap. AP summarizes the precision/recall tradeoff per class; mAP averages AP
over classes, with mAP50–95 also requiring increasingly precise box overlap.
Per-class columns use the common shorthand mAP for each class's AP.
Reported precision/recall use Ultralytics' evaluation operating point, so they
are not measurements of the app's chosen confidence threshold or alert rules.
See [Ultralytics evaluation](https://docs.ultralytics.com/modes/val/).

Do not tune settings based on test results and then report that same test as
untouched. Save the split inventory, environment record, commands, and results for
your course submission. Accuracy is unknown until measured on representative data.

## Explain it in class

This is **supervised learning** because humans provide the correct animal class
and box location during training. YOLO's CNN backbone learns visual features
through convolutions, from local patterns toward useful animal representations.
The detection head predicts classes and boxes. Training compares predictions
with labeled answers and updates weights to reduce loss. Inference applies those
learned weights to new frames without learning from the webcam. Starting with
pretrained weights is **transfer learning**; changing a confidence slider filters
predictions and does not retrain the model. A score of 90% is a model confidence
score, not a guaranteed 90% probability of correctness. Alert confirmation and
cooldowns are ordinary Python rules after the CNN; they are not learned behavior.

## Short course notebook and checks

While training is running, the guarded test command below uses mocks and fails
if a test attempts a real YOLO constructor or GPU-availability check. It runs
in-process ASGI requests with temporary fixtures; it does not bind a listening
server, open a dashboard, or touch the separate training process:

```powershell
.\.venv\Scripts\python.exe tests/run_without_models.py
```

`tests/demo_playback_checks.js` provides additional isolated fake-DOM/video/socket
checks for raw versus annotated display, continued alerts, drift/loop behavior,
per-camera errors, disabled settings and resource cleanup. Those development
JavaScript checks do not decode videos or prove visual smoothness. The production
app remains plain browser JavaScript with no Node.js runtime or build dependency.

Optional notebook tools are separate to keep the app environment smaller:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m notebook notebooks/course_workflow.ipynb
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Choose the project `.venv` kernel if prompted. The notebook covers environment,
one real image, alert timing, and opt-in training/evaluation commands; it does not
download or invent wildlife data. `tests/` checks timing, file logging, source
cleanup and annotations with temporary test fixtures, without camera access.

Project map: `detect.py` is the minimal demo; `app.py` is the interface;
`wildlife/` contains the small reusable functions; `train.py` and `evaluate.py`
run separately. Review Ultralytics' [license](https://www.ultralytics.com/license)
before distributing or changing how this project is used.
