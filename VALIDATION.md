# Project validation

## Demo stuck-Ready investigation — November 11, 2026

Before edits, all 42 Python and six JavaScript checks passed. Those checks did
not exercise mixed-version startup or the actual frontend Start handler through
the full WebSocket flow. The existing `live-view-2` asset query and initialization
`demoPlayback.configure(...)` attempt were present on disk, not assumed effective.

A pre-fix mocked ASGI reproduction sent a legacy Demo request without
`display_mode` (as a cached pre-mode client would). For **both** nominal selector
choices, the backend returned `display_mode=live`, camera status `Ready`, and
`connection.task is None`. The implicit Live default requires a `play` command,
but that older client does not send one. Thus no scheduler/frames/timestamps or
alerts can appear. A separate JS reproduction with a missing returned mode left
`running=true`, Start disabled, both displays hidden, and no `play` sent. This is
the demonstrated silent protocol-mismatch failure, not a path/codec diagnosis.

The exact original browser cache/server combination remains **unconfirmed**.
Read-only requests to port 8000 (page, health and all three versioned assets)
timed out; no reachable server response was available for byte comparison or a
restart check. No browser console or real failing-session WebSocket capture was
obtained. The evidence establishes the reproduced failure mechanism, not proof
that a particular browser cache entry or process was stale.

Targeted changes:

- Reject network Demo starts that omit the explicit display mode before model
  selection; stop visibly on missing/wrong returned mode in the frontend.
- Keep the initialization fix, add stopped/pageshow synchronization and a busy
  guard, and move startup display configuration inside error handling so missing
  DOM errors cannot leave Start disabled silently.
- Revise all asset URLs to `live-view-3` and send no-store page/static headers.
  Add a read-only `playback_protocol: demo-display-mode-1` health marker so manual
  restart verification can distinguish old imported Python from fresh files.
- Log startup stages using enum modes, validated camera IDs and media-presence
  booleans. Neither diagnostic logs nor JS traces expose filesystem paths or
  media tokens. Python logs confirm scheduler entry; JS traces distinguish
  preparation entry, completion, `play`, and `playing`.
- Show a camera-specific error when Live preparation has no media reference,
  while allowing detection to proceed through the existing handshake.

The playback architecture, sequential worker, sampling, confirmations, cooldowns,
CSV history and annotated snapshots were preserved. No datasets, training or
evaluation scripts, notebook, weights, or `runs/` contents were modified. No real
model was loaded, no GPU inference was run, and no listening server/dashboard was
launched. Tests use guarded mocks and temporary fixtures only.

**45 Python tests passed** with `python tests/run_without_models.py` using the
project `.venv`. New cases reject legacy mode-less starts without model loading,
verify current page/assets/health/required IDs, and cover all four combinations:

| Mode | Source | Verified message order |
| --- | --- | --- |
| Detection | Trusted path | start → started → frame → frame; no play |
| Detection | Upload | start → started → frame → frame; no play |
| Live | Trusted path | start → started → play → playing → frame → frame |
| Live | Upload | start → started → play → playing → frame → frame |

Local media range GETs return 206 in the integration tests. Frames update footage
metadata and repeated detections create events; Live frames carry no display JPEG
and Detection frames retain the annotated JPEG. These are mocked integrations,
not real model/codec/browser tests.

**14 JavaScript checks passed**: the six existing playback/cleanup cases and
eight startup cases in `runDemoStartupChecks()`. The new checks invoke actual
`startDemo()`/`receive()` code for both modes and source types; verify restored
selector/badge agreement and busy locking; reject missing/wrong server modes;
catch missing DOM during configuration; and simulate the configured media-load
timeout to verify that preparation resolves and sends `play`. With matching code, preparation
waits for media readiness/error/timeout, never a detection frame, so there is no
browser-waits-for-frame/server-waits-for-play circular dependency.

Real browser video visibility/smoothness, console errors and the original failing
session's versions are not claimed verified. README now includes exact backend
shutdown/restart, protocol-marker, hard-refresh, asset, console and WS checks.
Run manual video/inference verification only when training has finished and GPU
use is allowed. Do not interrupt the trainer to perform those checks.

## Demo playback modes — November 11, 2026

Implemented default **Live view** and optional **Detection view** while the
separate model-training process was running. No training process was stopped or
restarted. No dashboard, TCP server, browser, real YOLO model or GPU inference was
started. Datasets, training/evaluation scripts, model weights, `runs/` and the
course notebook were not modified for this change. Existing historical GPU/server
checks below describe earlier work, not verification of the new playback mode.

Live view displays raw muted inline browser video. Python retains its configured
1–3 detection samples/second per Demo camera, one sequential worker, confirmation
rules, per-camera cooldowns, annotated snapshots, CSV history and full event
metadata. Live view frame messages omit the JPEG display payload (`image: null`)
while continuing to send detection/notification data. Detection view retains the
previous annotated-JPEG behavior and immediate scheduler start. Debug remains
annotated. The display selector lives in the disabled-while-busy settings group.

Uploaded files use browser object URLs alongside their existing Python upload
copies. Selected validated local videos get session-scoped opaque media tokens.
The media endpoint uses the pinned Starlette FileResponse range implementation;
it accepts tokens, not filesystem paths. Invalid/unselected/expired tokens yield
404. Range GET, HEAD, partial 206 responses and invalid-range 416 responses were
tested with temporary bytes and mocked video validation. No real dataset video
or model was used for these checks.

Live view adds a browser-readiness handshake, one shared replay epoch, a short
coordinated start delay and optional corrections for drift greater than one
second. A separate current playback clock accounts for inference delay; alert
footage timestamps continue to come from the sampled Python source. Looping
reuses existing sockets, registrations and uploads. Browser codec/autoplay errors
are isolated by camera and recommend Stop → Detection view as a fallback.

**42 Python tests passed**, including all 33 earlier tests, with:

```powershell
.\.venv\Scripts\python.exe tests/run_without_models.py
```

The runner guards the real Ultralytics constructor and CUDA availability checks.
Tests use in-process FastAPI TestClient requests, mocked model loading/inference,
small temporary software video/byte fixtures, and CPU operations. These requests
do not launch a network server or dashboard. New coverage includes configuration
defaults/rate limits, local media access restrictions and expiry, byte ranges,
annotated snapshot/CSV events without a Live view JPEG, Detection view fallback,
uploaded Live view cleanup, fresh tokens/model reuse across restart, duplicate
play rejection, four-camera epoch/cleanup and loop/playback-clock behavior.
Disconnect testing found and fixed cancellation interrupting file cleanup;
cleanup is now shielded and shutdown tolerates an already-closed socket.

**Six JavaScript behavioral checks passed** in an isolated development runtime
using `tests/demo_playback_checks.js`, fake DOM/video/socket objects and fake
timers. They exercised the actual `demo-playback.js` and `app.js` code, with no
browser/media/network access:

- Raw-only Live view display while detection text and notifications continue.
- Drift threshold, loop boundaries, and no duplicate timers or URLs on updates.
- Annotated-only Detection view fallback.
- Isolated browser playback failure with camera-specific fallback guidance.
- Disabled settings during startup/run/stop, replacement and Stop cleanup.
- Cancelled loading, tab changes, socket failures and page exit revoking every
  created object URL exactly once.

After training finishes, still verify native video smoothness, actual 24–30 FPS
playback where supported by the footage/browser, autoplay/codec behavior, seeking,
four-video decoder/GPU performance, layout and long-run synchronization in a real
browser. None of those physical performance checks are claimed here. No model
training or inference benchmark was run for this change.

## Dataset preparation — November 11, 2026

Ran `prepare_dataset.py` against the existing local downloads. No model training
was started. The new detection classes are exactly `0: deer`, `1: wild_boar`,
`2: cow`; pretrained YOLO's `cow` alert support remains in place. Training and
evaluation now default to `data/wildlife_dataset/dataset.yaml`.

| Generated class | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| deer | 231 | 50 | 50 | 331 |
| wild_boar | 231 | 50 | 49 | 330 |
| cow | 231 | 50 | 50 | 331 |
| **All images** | **693** | **150** | **149** | **992** |

Deer retains all 7 original validation images and all 3 original test images.
Seed 42 assigns 43 unaugmented training images to validation and 47 to test;
the remaining 231 stay in training. Boar retains 50 original validation and all
49 original test images; cow uses 50 original validation and 50 original test
images. Excess source holdouts are excluded, never moved into training.

The source inventory contained 3,322 images. There were no exact image-hash
duplicates. Excluded images total **2,330**:

- 5 boar-source images with no target annotations (their label files are empty).
- 1,454 cow-source related variants, including training variants of two filename
  families also present in source holdouts. One export per known family remains
  eligible, with test/validation taking priority.
- 871 additional eligible images beyond the target counts: 165 boar and 706 cow.

By source, zero deer, 170 boar and 2,160 cow images are excluded. Each exclusion
and its original path appears in `dataset_summary.json`; excluded originals
remain in their downloaded folders.

Generated labels contain **3,162 target boxes**: 2,578 existing rectangles and
584 converted polygons. Fifteen deer rectangles had tiny normalized boundary
overruns (maximum about 0.00000244); those edges were clamped using a 0.00001
tolerance. This changes only the generated labels. Larger errors are rejected.
The 15 clamps are a subset of the 2,578 existing rectangles, not extra boxes.
Selected mixed boar images omit 35 dog annotations and 29 human annotations.

Generated artifacts:

- `data/wildlife_dataset/dataset.yaml`
- `data/wildlife_dataset/source_manifest.csv` (992 image records)
- `data/wildlife_dataset/dataset_summary.json` (counts, transitions and all skips)
- `data/wildlife_dataset/contact_sheets/train.jpg`, `val.jpg`, `test.jpg`
- `data/wildlife_dataset/source_licenses/` (original metadata and attribution)

Preparation built a staging copy and ran the existing dataset checker before
publication. The final generated YAML was checked again with all three splits
and `require_all_classes=True`: **passed**. Short generated JPEG/label stems
match. The manifest includes the requested provenance, exact source licenses and
SHA-256 hashes. Extended Windows paths allow long downloaded cow filenames to be
read without renaming source files.

The complete suite passed: **33 tests**, including 10 new focused preparation
tests for polygons, box validity/clamping, filtering/remapping, duplicate and
family prevention, exact deer split counts, short names, manifests, contact
sheets, reproducibility, source immutability and safe rebuild/refusal. Existing
application, events, source and dataset checks still pass. Rebuild behavior was
exercised on temporary fixtures, not by replacing the real downloads.

All **6,653 original source files** were SHA-256 fingerprinted before preparation
and checked afterward: **zero modified or missing files**. Original data folders
were used only as read-only inputs.

All three contact sheets were visually inspected: sampled boxes/classes align
with the converted annotations. This is a label-conversion spot check, not proof
of every source annotation's correctness. Similar deer scenes appear across
splits; exact hashes and filename families do not identify all shared visits or
adjacent video frames. Review encounter/video groups before claiming an
independent held-out accuracy estimate. No wildlife metrics are claimed.

## FastAPI refactor — November 9, 2026

Verified on this Windows laptop with regular CPython 3.13.7 and an NVIDIA
GeForce RTX 4050 Laptop GPU (6 GB VRAM). An isolated `.venv` is installed.

Installed versions:

| Dependency | Verified version |
| --- | --- |
| PyTorch | 2.8.0+cu126 |
| torchvision | 0.23.0+cu126 |
| Ultralytics | 8.3.228 |
| FastAPI | 0.121.1 |
| Uvicorn | 0.38.0 |
| WebSockets | 15.0.1 |
| python-multipart | 0.0.20 |
| Jinja2 | 3.1.6 |
| HTTPX (tests) | 0.28.1 |
| OpenCV | 4.12.0.88 |
| NumPy | 2.2.6 |

`environment-verified.json` records the hardware/runtime check and
`requirements-lock.txt` records all installed dependency versions. To reproduce
the exact application environment, install the official torch/torchvision pair
from the README first, then use `pip install -r requirements-lock.txt` with the
project Python. Optional notebook packages are not part of this lock file.

Checks completed:

- `pip check`: no broken requirements.
- Streamlit and its unused interface dependencies were uninstalled from `.venv`.
  A dependency-graph audit found no remaining app-unneeded packages. The ML
  package pins were preserved; the complete new installed snapshot is recorded.
- Real Uvicorn startup, main HTML, CSS, JavaScript and `/api/health` requests
  returned HTTP 200. A WebSocket JPEG carrying the official bus image ran actual
  YOLO11n inference on the GPU and returned boxes, labels, scores, timing and
  an annotated JPEG. This simulates browser transport, not physical webcam use.
- A real multipart upload of the two-second sample AVI was decoded by OpenCV
  and processed with the same cached GPU model through a new WebSocket. Six
  replay frames returned footage timestamps. Stop closed the socket and removed
  the temporary upload. The trusted original remained on disk.
- The test Uvicorn server received a graceful shutdown request and printed
  `Application shutdown complete`. No server exceptions occurred in that smoke run.
- CUDA availability and an actual tensor multiplication on the RTX 4050 passed.
- `detect.py` was rerun on the official bus image and the derived sample AVI.
  Both completed, preserving the original two-result detector API. New outputs
  are under `outputs/smoke/refactor/`. The original annotated sample was visually
  inspected during Stage 1; the new browser interface was not visually verified.
- All **23 behavioral tests** passed using:

  ```powershell
  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
  ```

  Coverage includes startup/static files; valid and invalid WebSocket messages;
  live/replay detection formatting; repeated confirmation, cooldown and model
  reuse across Stop/Start connections; one-pending-frame replacement; stale frame
  dropping; per-camera alert isolation; four-source cleanup; loop playback with
  a continuous cooldown clock; combined notification fields; CSV camera names and
  old-schema history preservation; snapshot serving; temporary upload cleanup;
  safe model/video paths; invalid source isolation; and one sequential worker.
  The original dataset and webcam failure/cleanup tests remain. Inference and
  browser streams are mocked in unit tests. A separate HTTP/WebSocket smoke check
  uses real YOLO, OpenCV, the GPU and a saved video.
- Python sources and notebook code cells compile. The notebook's code cells were
  also run with no user-supplied data; image/video cells correctly skip absent
  files. Training/evaluation help commands run successfully.

The bus sample is from
[Ultralytics assets](https://github.com/ultralytics/assets/blob/main/im/bus.jpg).
The short derived video is a pipeline smoke fixture, not wildlife training data
and not an accuracy benchmark. Test fixtures in `tests/` are also only software
checks. No synthetic dataset is supplied for training.

## Browser verification limit

Browser automation access was unavailable, and the native browser fallback was
stopped by Computer Use because it could not determine the current browser URL
confidently enough to enforce its policy. UI automation was left stopped.
Server logs did show HTML/CSS/JavaScript requests and a browser WebSocket
connection, but that does not establish visual or console correctness.

Debug/Demo switching, browser console errors, readable visual layout at laptop
and narrow widths, live permission flows, and media-track shutdown need manual
browser verification. The responsive CSS and accessible labels are implemented,
but are not reported as visually tested.

Also not measured: physical webcam/screen sharing, four-video GPU performance,
broad recording codec coverage, wildlife accuracy, training memory usage, and
test-set metrics. Camera isolation/four-source cleanup are tested with mocks,
not four physical cameras or four real GPU video feeds. No custom model was
trained and no wildlife performance scores are claimed.
