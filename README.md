# Machine-vision-project
Vision-based traffic analysis system with YOLOv8 vehicle detection, persistent tracking, homography-based speed estimation, vehicle counting, congestion analysis, and live violation alerts through a Flask web dashboard.

#**Project Files**
app.py Main Flask server. It:

opens the website

accepts uploaded videos or live camera sources

starts/stops analysis sessions

serves the live annotated video feed

returns live metrics to the frontend
traffic_analyzer.py Core machine vision logic. It:

reads video frames

detects vehicles

tracks them across frames

estimates speed using optical flow / motion tracking

checks for overspeeding, wrong-way driving, and illegal U-turns

computes confidence values

draws boxes, labels, and the overlay on the video
violation_model.py Loads the trained ML model and prepares feature values. It:

defines the training/prediction feature set

converts track history into model input features

loads violation_model.joblib

returns violation probabilities
train_violation_model.py Training script for the illegal-driving classifier. It:

reads the training CSV

trains the Random Forest model

evaluates it

saves the trained model into models/
vehicle_detector.py Optional detector loader. If you train your own detector and save it as models/vehicle_detector.pt, this file loads it and uses it for vehicle detection.
train_vehicle_detector.py Trains a custom vehicle detector from scratch using a YOLO-style architecture and a converted dataset like BDD100K.
templates/index.html The actual webpage structure. Defines:

header

form controls

video display area

metric cards

alerts section
static/styles.css All styling for the web interface.
static/app.js Frontend behavior. It:

switches between upload/live mode

sends form data to Flask

starts/stops analysis

updates dashboard metrics live

updates alerts live
data/Indian_Traffic_Violations.csv External traffic violation records dataset used for recalibrating the model, especially overspeeding behavior.
data/violation_training_seed.csv Small seed dataset containing motion-based examples for overspeeding, wrong-way, and illegal U-turn behavior.
data/violation_training_data.csv Final merged training dataset used by the model.
models/violation_model.joblib The trained illegal-driving classifier used by the app.
models/vehicle_detector.pt Optional custom trained detector model if you train one.
tools/prepare_indian_violation_training.py Converts the Indian violations dataset into training rows and merges it with the seed dataset.
tools/trajectory_to_violation_dataset.py Converts trajectory datasets like inD/highD/rounD into training rows for the violation model.
tools/bdd100k_to_yolo.py Converts BDD100K labels into YOLO format for detector training.
README.md Setup, usage, training instructions, and project explanation.
Web Dashboard Elements
Top Header Shows the project title and short project identity text. Purely presentational.
Traffic Source Lets the user choose:

Upload video file

Connect live traffic camera
Video File Used when upload mode is selected. You choose a saved video from your computer.
Camera URL or webcam index Used when live mode is selected. Can accept:

direct stream URL

webcam index like 0
Pixels per meter Calibration value. Used to convert motion in pixels into estimated real-world speed.
Speed limit (km/h) Threshold for overspeeding detection.
Expected traffic direction Defines legal motion direction:

left to right

right to left

top to bottom

bottom to top
This is used for wrong-way detection.
U-turn angle threshold Controls how sharp a direction change must be before it is considered a possible illegal U-turn.
Start analysis Starts the video processing.
Stop Stops the current analysis session.
Video Display Area Shows the processed video feed with:

vehicle boxes

track trails

speed text

status label

confidence label
Vehicle Box Overlay For each vehicle it shows:

vehicle ID

estimated speed

status like NORMAL, OVERSPEED, or VIOLATION

confidence percentage
Vehicles Number of currently confirmed tracked vehicles.
Average speed Average speed of the currently confirmed vehicles.
Overspeeding Number of vehicles currently flagged as overspeeding.
Wrong-way Number of vehicles currently flagged as moving in the wrong direction.
U-turns Number of vehicles currently flagged for possible illegal U-turns.
Overall confidence Average confidence across all currently confirmed tracked vehicles.
Alert confidence Average confidence only across vehicles that are currently flagged for a violation.
Alerts Live list of detected violations. Examples:

Vehicle 5: overspeeding (72.4 km/h, 91% confidence)
•
Vehicle 3: wrong-way driving (67% confidence)
•
Vehicle 9: possible illegal U-turn (64% confidence)
