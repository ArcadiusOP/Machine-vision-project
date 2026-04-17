# Vision-Based Speed Estimation & Traffic Analysis

A working machine vision mini-project with a web interface for traffic video analysis. It can analyze an uploaded video file or connect to a live camera/stream URL.

The project estimates vehicle speed with optical flow and flags:

- Overspeeding
- Wrong-way driving
- Possible illegal U-turns
- ML-classified illegal-driving decisions from track features

## How It Works

The app uses OpenCV background subtraction to find moving vehicle-sized regions. Each moving region is tracked across frames with centroid association. Inside every tracked vehicle box, Lucas-Kanade optical flow estimates pixel motion between frames. Pixel speed is converted to km/h using the `pixels per meter` calibration value from the UI.

Event detection is rule-based:

- Overspeeding: estimated speed is above the configured speed limit.
- Wrong-way driving: the tracked direction opposes the expected road direction.
- Illegal U-turn: the current direction turns sharply away from the initial direction.

The project also includes a trained Random Forest classifier. It learns from vehicle-track features such as speed, direction, path length, straightness, turn angle, and wrong-way score. If `models/violation_model.joblib` exists, the app uses it for wrong-way and U-turn decisions. Overspeeding still uses the configured speed limit as a hard rule because that violation is deterministic after speed estimation.

This design does not require YOLO weights or an internet download, which makes it easier to run for lab demonstrations.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open the app at:

```text
http://127.0.0.1:5000
```

## Train The Illegal-Driving Model

A starter labelled dataset is included at:

```text
data/violation_training_data.csv
```

Train or retrain the model with:

```powershell
python train_violation_model.py --dataset data/violation_training_data.csv --output models/violation_model.joblib
```

The CSV must include these feature columns:

```text
speed_kmh,speed_over_limit,direction_dx,direction_dy,path_length,displacement,straightness,turn_angle,wrong_way_score,seen_count
```

And these label columns:

```text
overspeeding,wrong_way,illegal_uturn
```

Use `0` for normal behavior and `1` for violation behavior. To improve accuracy for your own demo location, add rows from your own videos. More real examples of legal movement, wrong-way movement, and U-turn movement will make the classifier more reliable.

## Use A Larger Trajectory Dataset For Illegal Driving

For illegal-driving decisions, use trajectory datasets instead of image detection datasets. Good open choices are:

- inD: intersection road-user trajectories
- rounD: roundabout trajectories
- highD: highway vehicle trajectories

These datasets provide per-vehicle motion tracks, which match this project's violation classifier better than image-only datasets.

After downloading one of those datasets, point the converter at the folder that contains files ending in `_tracks.csv`:

```powershell
python tools/trajectory_to_violation_dataset.py --root "D:\datasets\ind" --output data\violation_training_data_big.csv --speed-limit 50 --expected-direction right
```

Then train the illegal-driving classifier on the larger dataset:

```powershell
python train_violation_model.py --dataset data\violation_training_data_big.csv --output models\violation_model.joblib
```

The converter creates labels from motion patterns:

- overspeeding: average speed is above the configured speed limit
- wrong-way: net motion is opposite the expected traffic direction
- illegal U-turn: trajectory changes direction sharply and is not straight

For the most accurate final model, manually review and correct the generated labels for a few hundred tracks from your target camera angle.

## Train Your Own Vehicle Detector With BDD100K

For a stronger project without using pretrained weights, use the BDD100K detection dataset. It is a large driving dataset with road-scene images and labels for vehicles, traffic lights, and traffic signs.

Download BDD100K from:

```text
https://bdd-data.berkeley.edu/
```

Expected folder layout after download:

```text
BDD100K_ROOT/
  images/
    100k/
      train/
      val/
  labels/
    det_20/
      det_train.json
      det_val.json
```

Convert BDD100K detection labels to YOLO format:

```powershell
python tools/bdd100k_to_yolo.py --bdd-root "D:\datasets\bdd100k" --output data/bdd100k_yolo
```

Train a detector from scratch:

```powershell
python train_vehicle_detector.py --data data/bdd100k_yolo/bdd100k.yaml --epochs 50 --imgsz 640 --batch 8
```

The script saves the best trained detector here:

```text
models/vehicle_detector.pt
```

When that file exists, the web app automatically uses your trained detector instead of the basic background-subtraction detector.

## Using The Web Interface

1. Choose `Upload video file` or `Connect live traffic camera`.
2. For upload mode, select a traffic video.
3. For live mode, enter an RTSP/HTTP stream URL, a local video path, or `0` for a webcam.
4. Set `Pixels per meter`.
5. Set the speed limit and expected traffic direction.
6. Click `Start analysis`.

## Calibration Tips

Speed accuracy depends on `pixels per meter`. For a real road video, measure a known road marking or lane width in pixels:

```text
pixels_per_meter = measured_pixels / real_world_meters
```

Example: if a 3.5 m lane appears 49 pixels wide, use `14` pixels per meter.

## Project Structure

```text
app.py                  Flask web server and API routes
traffic_analyzer.py     OpenCV detection, optical flow, tracking, and alerts
violation_model.py      Track feature extraction and model loading
train_violation_model.py Model training script
vehicle_detector.py     Optional custom detector loader
train_vehicle_detector.py Custom detector training script
tools/bdd100k_to_yolo.py BDD100K to YOLO converter
tools/trajectory_to_violation_dataset.py Trajectory dataset converter for violation training
data/                   Labelled training dataset
models/                 Trained model output
templates/index.html    Web interface
static/styles.css       Interface styling
static/app.js           Browser-side controls and live metrics
uploads/                Uploaded videos, created automatically
```

## Notes For Report

You can describe the system as a hybrid motion-based traffic analyzer:

- Foreground extraction isolates moving road users.
- Multi-object tracking maintains vehicle identities.
- Sparse optical flow estimates local motion vectors.
- Pixel-to-world calibration converts motion to approximate speed.
- A trained Random Forest classifier identifies ambiguous illegal-driving patterns.
- Speed-limit comparison identifies overspeeding.

Limitations:

- It estimates speed from image-plane motion, so perspective distortion affects accuracy.
- Shadows and camera shake can create false detections.
- Occlusion can break object tracks.
- A production system should use camera calibration, lane geometry, and a trained vehicle detector.
