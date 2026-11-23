# Texas Wildlife Camera

A lightweight trail-camera prototype that detects Texas wildlife and creates timestamped alerts. The project uses a custom YOLO11n model and a local FastAPI dashboard to monitor webcam, screen-share, and recorded-video sources.

## Features

- Detects deer, wild boar, and cows
- Supports webcam, screen sharing, and uploaded videos
- Includes Debug and four-camera Demo dashboards
- Provides live playback and an optional bounding-box view
- Records animal type, confidence score, time, camera, and snapshot
- Runs locally without sending footage to a cloud service

## Technology

- Python
- YOLO11n and Ultralytics
- PyTorch and OpenCV
- FastAPI and WebSockets
- HTML, CSS, and JavaScript

## Model Results

The strongest image-test model achieved:

- 82.8% mAP50
- 83.8% precision
- 75.3% recall
- Evaluation on 149 held-out images

These are object-detection measurements, not ordinary accuracy. Performance can vary with lighting, camera angle, distance, and animal appearance.

## Run Locally

1. Open PowerShell in this folder.
2. Create and activate a virtual environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install PyTorch and the project dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

4. Start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

5. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Select a trained `best.pt` file in the dashboard to use the custom wildlife classes. Model weights and datasets are intentionally excluded from GitHub because of their size and licensing requirements.

## Training and Evaluation

```powershell
.\.venv\Scripts\python.exe train.py --data data/wildlife_dataset/dataset.yaml --imgsz 640 --batch 4 --epochs 30
.\.venv\Scripts\python.exe evaluate.py --data data/wildlife_dataset/dataset.yaml --model "runs/train/wildlife/weights/best.pt" --batch 4
```

## Project Structure

```text
app.py                 FastAPI application
wildlife/              Detection, alerts, media, and runtime logic
templates/             Dashboard HTML
static/                CSS and JavaScript
train.py               Model training
evaluate.py            Held-out evaluation
prepare_dataset.py     Dataset preparation
tests/                 Automated tests
```

See [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) for detailed setup, architecture, validation, and troubleshooting notes.

## Current Limitations

- This is a local prototype, not a deployed outdoor trail-camera system.
- Wild boars can still be confused with cows in difficult footage.
- Nighttime, weather, and long-term outdoor performance need additional testing.
