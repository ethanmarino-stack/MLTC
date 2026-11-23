# Texas Wildlife Camera — Project Roadmap

Last updated: November 11, 2026

## One-Slide Summary

We started with a concept for a lightweight “Ring camera for trails” that could detect Texas wildlife and create timestamped alerts. We built a local FastAPI dashboard around YOLO11n, assembled and verified a three-class dataset, trained a baseline model, and tested it on footage outside the original datasets. Those real-world tests exposed class confusion that ordinary test metrics did not fully show. We then used an active-learning process—adding carefully labeled frames from representative videos—to improve the model. Deer and cow detection now generalize well, while the current development cycle focuses on improving wild-boar detection without contaminating the final test footage.

## Development Timeline

| Stage | What We Did | Why It Mattered | Result |
|---|---|---|---|
| 1. Project concept | Defined a Texas wildlife camera for hunters, property owners, and trail safety. | Established a clear business problem and practical use case. | Target output: animal name, confidence score, timestamp, snapshot, and notification. |
| 2. Initial prototype | Built detection scripts and an early Streamlit interface. | Proved that local YOLO inference, webcam input, saved video, alerts, and snapshots were possible. | Working proof of concept. |
| 3. Custom dashboard | Replaced Streamlit with FastAPI plus custom HTML, CSS, and JavaScript. | Gave the project a more professional product-style interface while keeping all machine-learning logic in Python. | Debug tab, four-camera Demo tab, uploads, webcam/screen sharing, replay feeds, and notification history. |
| 4. Dataset selection | Downloaded premade YOLO datasets for deer, wild boar, and cows. | Avoided manually collecting and labeling the entire initial dataset. | Three target classes: `deer`, `wild_boar`, and `cow`. |
| 5. Dataset preparation | Safely remapped class IDs, removed duplicates, verified bounding boxes, balanced image counts, and preserved holdout data. | Different downloads used different class numbers; copying files directly would have trained incorrect labels. | Initial prepared dataset: 992 images. |
| 6. Baseline training | Fine-tuned YOLO11n for 20 epochs on an RTX 4050 laptop GPU. | Established measurable baseline performance with a lightweight model. | Baseline test mAP50: **80.0%**; mAP50–95: **53.7%**. |
| 7. Real-video testing | Tested deer, boar, and cow footage through the dashboard. | Showed whether the model transferred beyond images from the downloaded datasets. | Deer and boars were frequently mislabeled as cows despite acceptable dataset metrics. |
| 8. First active-learning cycle | Auto-labeled 60 representative video frames in Roboflow and added them only to training. | Focused training on examples the model had previously misunderstood. | New frames: 27 deer, 26 wild boar, and 7 cow images. |
| 9. Second model | Continued training from the baseline model for 30 epochs. | Preserved previous learning while adapting to video-style images. | Test mAP50 increased to **82.8%**; mAP50–95 increased to **58.4%**; precision increased from **73.5% to 83.8%**. |
| 10. Unseen-video evaluation | Tested completely different footage that was not used for training. | Provided a more honest real-world evaluation. | Deer and cow detection performed well. Unseen boars were still confused with cows/deer or missed. |
| 11. Boar-focused expansion | Auto-labeled and verified 82 additional boar images from separate training-only videos. | Addressed the weakest class without changing the already strong deer and cow data. | Current dataset: 1,134 images; 339 wild-boar training images. |
| 12. Boar-focused Model 3 | Continued training from `wildlife2/weights/best.pt` for 30 epochs after adding 82 boar images. | Tested whether targeted boar examples would improve generalization. | Boar precision improved, but overall test mAP50 decreased to **80.3%**. Model 2 remains the current statistical winner. |
| 13. Smooth Live view | Separated visible video playback from YOLO’s detection-sampling rate in the Demo dashboard. | A trail-camera dashboard should look live even when the lightweight model examines only 1–3 frames per second. | Implemented with Live view as the default and Detection view as the annotated fallback. Automated tests passed; real-browser verification remains. |

## Model Progress

| Model | Training change | Test precision | Test recall | Test mAP50 | Test mAP50–95 |
|---|---|---:|---:|---:|---:|
| Model 1 | Initial balanced image dataset | 73.5% | 76.5% | 80.0% | 53.7% |
| Model 2 | Added 60 labeled video frames | 83.8% | 75.3% | 82.8% | 58.4% |
| Model 3 | Added 82 boar-focused video frames | 83.5% | 71.7% | 80.3% | 56.4% |

### Model 3 Validation Results

| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Deer | 93.4% | 89.1% | 92.1% | 79.2% |
| Wild boar | 82.4% | 74.9% | 90.7% | 68.9% |
| Cow | 80.9% | 63.5% | 77.7% | 43.2% |
| **Overall** | **85.6%** | **75.8%** | **86.8%** | **63.8%** |

Compared with Model 2 validation, Model 3 increased wild-boar precision from 76.1% to 82.4% and wild-boar mAP50 from 88.7% to 90.7%, but wild-boar recall decreased from 82.6% to 74.9%. This means it may make fewer incorrect boar predictions while missing more actual boars. The unchanged test set and untouched boar videos are required before choosing the final model.

### Model 3 Test Results

| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Deer | 90.5% | 87.9% | 93.5% | 71.7% |
| Wild boar | 79.0% | 64.8% | 75.8% | 58.1% |
| Cow | 80.9% | 62.4% | 71.6% | 39.4% |
| **Overall** | **83.5%** | **71.7%** | **80.3%** | **56.4%** |

Model 3 did not beat Model 2 on the unchanged image test set. Compared with Model 2, its overall mAP50 fell from 82.8% to 80.3%, overall mAP50–95 fell from 58.4% to 56.4%, and wild-boar mAP50 fell from 77.4% to 75.8%. Wild-boar precision increased from 76.4% to 79.0%, showing a precision/recall tradeoff rather than a general improvement. However, the same-footage comparison revealed an important deployment difference: Model 2 completely missed the animal in the untouched `boar3.mp4`, while Model 3 correctly and consistently detected it as wild boar at 75–83% confidence. Both models still misclassified the difficult `boar2.mp4` as cow. Model 2 remains the image-benchmark winner, but Model 3 is the preferred demonstration/deployment candidate because detecting a real animal is more useful to the trail-camera use case than missing it entirely.

### Model 2 Per-Class Test Results

| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Deer | 93.1% | 92.5% | 94.9% | 75.6% |
| Wild boar | 76.4% | 67.0% | 77.4% | 57.5% |
| Cow | 82.0% | 66.5% | 76.1% | 41.9% |

## Current Dataset

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Deer | 258 | 50 | 50 |
| Wild boar | 339 | 50 | 49 |
| Cow | 238 | 50 | 50 |
| **Total** | **835** | **150** | **149** |

The new video frames are training-only. Validation and test images remain separate. The final boar videos are also kept completely outside the dataset so they can provide an honest real-world test.

## What We Learned

1. A strong dataset score does not guarantee strong performance on new footage.
2. Confidence is the model’s strength of belief, not the probability that a prediction is correct.
3. Real-world testing revealed errors that the original validation and test images did not expose.
4. Targeted active learning was more useful than simply adding more epochs.
5. Class IDs must be remapped carefully when combining YOLO datasets from different sources.
6. Closely related video frames must not be divided across training and final testing because that would inflate results.
7. A small YOLO11n model can run locally and quickly on consumer laptop hardware.
8. The class covered CNNs for image recognition/object detection and covered precision and recall as evaluation metrics, but it did not specifically cover mAP50.
9. Object detection must evaluate both the predicted class and the location of its bounding box, so ordinary classification accuracy is not sufficient.
10. mAP50 extends the precision/recall concepts from class: a detection counts as correctly located when its predicted box has at least 50% Intersection over Union with the labeled box, and Average Precision is then averaged across the three animal classes.
11. mAP50 is not the same as “percent accuracy,” and a notification confidence score is not the model’s measured accuracy.
12. Visible video FPS and detection FPS can be separated. The browser can play raw footage at approximately 24–30 FPS while Python samples only 1–3 FPS for YOLO, preserving smooth playback without greatly increasing GPU usage.

## Understanding the Evaluation Metrics

- **Precision:** Of the animal detections produced by the model, how many were correct?
- **Recall:** Of the animals actually present, how many did the model detect?
- **Intersection over Union (IoU):** The overlap between a predicted bounding box and the labeled bounding box divided by their combined area.
- **Average Precision (AP):** A combined summary of precision and recall for one class across confidence thresholds.
- **mAP50:** The mean AP across deer, wild boar, and cow when a predicted box requires at least 50% IoU to count as correctly located.
- **mAP50–95:** A stricter measurement averaged across IoU requirements from 50% through 95%.

The Model 2 result of 82.8% mAP50 should therefore be presented as an object-detection score, not as 82.8% ordinary accuracy.

## Live View Design

The original Demo view displayed only frames after YOLO processed them. At one detection sample per second, the footage therefore looked like a one-FPS slideshow even though inference itself was fast. The updated design separates presentation from analysis:

| Component | Intended rate | Purpose |
|---|---:|---|
| Browser Live view | Approximately 24–30 FPS | Smooth raw footage without bounding boxes. |
| Python/YOLO detection | Configurable 1–3 FPS per camera | Lightweight background animal detection. |
| Alert system | Event-based | Confirmed animal notifications, confidence, footage time, processing time, snapshots, cooldowns, and CSV history. |

Live view does not require YOLO to analyze every displayed frame. Detection view remains available when annotated bounding boxes are useful for debugging or explaining how the model works.

The implementation now uses raw HTML video playback for Live view while Python continues sequential detection in the background. Uploaded videos use browser object URLs. Trusted local files use session-only opaque media tokens with byte-range support instead of exposing filesystem paths. Alerts, confirmation rules, cooldowns, annotated snapshots, and CSV records remain in Python. Forty-two mocked Python tests and six JavaScript checks passed. Browser smoothness, codec/autoplay behavior, long-run synchronization, and four-video performance still require hands-on verification.

## Remaining Roadmap

1. Confirm the empty-forest control produces no false animal alerts.
2. Verify smooth Live view in a real browser while preserving the existing Detection view.
3. Confirm that raw playback remains smooth while alerts continue at the configured detection-sampling rate.
4. Check codec/autoplay behavior, looping, cleanup, and long-run synchronization.
5. Measure dashboard performance with four simultaneous replay feeds.
6. Use Model 3 for the demonstration while reporting that Model 2 achieved the best image-test metrics.
7. Capture final dashboard screenshots and notification examples.
8. Document the shared `boar2.mp4` cow-confusion limitation and possible future work, including more diverse Texas feral-hog footage, night footage, additional species, physical trail cameras, and remote push notifications.

## Suggested Slideshow Structure

1. **Problem and Use Case** — A Ring-style camera concept for trails and large properties.
2. **Project Goal** — Detect Texas wildlife and create timestamped local alerts.
3. **System Design** — Browser dashboard, FastAPI server, Python detection pipeline, YOLO11n, and separate Live/Detection display rates.
4. **Data Preparation** — Premade datasets, class remapping, duplicate removal, and train/validation/test separation.
5. **Evaluation Method** — Precision and recall from class, bounding-box overlap, mAP50, and why mAP is not ordinary accuracy.
6. **Baseline Results** — Initial test metrics and the first working dashboard.
7. **Real-World Problem** — Unseen footage exposed cow-biased predictions.
8. **Active Learning** — Auto-labeled video frames targeted the model’s actual mistakes.
9. **Improvement** — Model 2 metrics and stronger deer/cow video performance.
10. **Current Challenge and Next Steps** — Boar-focused Model 3, smooth Live view, final video tests, limitations, and deployment ideas.

## Short Presentation Script

“We began with the idea of a Ring camera for trails. The system would identify an animal and generate a notification containing the animal, time, confidence score, and snapshot. We selected YOLO11n because it is lightweight enough for local use. Because this is object detection instead of ordinary classification, we evaluate both the predicted class and the bounding-box location. Our main metric, mAP50, builds on precision and recall from class and requires at least 50 percent overlap between the predicted and labeled boxes. After combining three premade datasets, our initial model achieved an 80 percent test mAP50. Real-video testing then showed that it often confused deer and boars with cows. Instead of hiding that limitation, we used it to guide the next training cycle. We automatically labeled representative video frames, retrained the model, and increased test mAP50 to 82.8 percent while raising precision from 73.5 to 83.8 percent. A third boar-focused experiment lowered the overall image-test mAP50 to 80.3 percent, but it correctly detected an unseen boar video that the statistically stronger model missed entirely. We therefore selected Model 3 for the practical demonstration while reporting Model 2 as the image-benchmark winner. Both models still share one difficult boar-to-cow confusion case. We are also separating smooth browser playback from YOLO sampling so the dashboard can display video at normal speed while detection and alerts remain lightweight.”

