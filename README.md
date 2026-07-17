# Smart Camera System

A computer vision application for intelligent camera monitoring, motion detection, camera tamper detection, and object tracking.

The system continuously analyzes a video stream, detects activity inside a configurable region of interest (ROI), distinguishes between scene motion and camera movement, and triggers YOLO-based object tracking only when relevant events occur.

---

## Features

### Motion Detection

Detects movement using frame differencing and contour analysis.

* Background-independent
* Configurable sensitivity
* ROI-based monitoring

### Region of Interest (ROI)

Monitor only the area that matters.

* Click two points to define ROI
* Reset ROI with right-click or double-click
* Full-frame fallback when no ROI is selected

### Camera Shift Detection

Uses optical flow to identify camera movement and avoid false alarms.

Examples:

* Someone bumps the camera
* Camera mount shifts
* PTZ movement

### Camera Drop Detection

Detects persistent displacement from the original camera position.

Examples:

* Camera physically moved
* Camera rotated
* Camera mount failure

### Camera Health Monitoring

Automatically validates camera feed quality.

Detects:

* Lost signal
* Frozen image
* Lens obstruction
* Excessive scene variance

### YOLO Object Tracking

When valid motion is detected:

1. Motion is confirmed
2. Camera movement is ruled out
3. YOLO tracking is activated

Tracked objects receive:

* Persistent IDs
* Trajectory visualization
* Historical movement paths

---

## Architecture

```text
Video Source
      │
      ▼
Camera Manager
      │
      ▼
Frame Preprocessing
      │
      ▼
Motion Detection
      │
      ├── Camera Shift Detection
      │
      ├── Camera Drop Detection
      │
      └── Motion Confirmed
                │
                ▼
          YOLO Tracker
                │
                ▼
           Visualization
```

---

## Technologies

* Python
* OpenCV
* Ultralytics YOLO
* NumPy
* YAML Configuration

---

## Installation

### Clone Repository

```
git clone https://github.com/shoogar7/smart-camera-system.git
cd smart-camera-system
```

### Create Virtual Environment

```
python -m venv venv
.venv\Scripts\activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

---

## Configuration

Configuration is loaded from:

```
config.yaml
```

Example:

```

tracker:
  model_path: yolov8n.pt
  track_classes: [0]

motionManager:
  motion_threshold: 3000
```

---

## Usage

Run:

```
python src/main.py
```

Two windows will appear:

### Normal View

Main camera stream.

Functions:

* Draw ROI using two left-clicks
* Delete ROI with right-click
* Displays tracking results
* Displays FPS

### Detection View

Visual representation of motion detection processing.

---

## Workflow

```text
Motion Detected
       │
       ▼
Camera Shift?
       │
   Yes ─────► Ignore Event
       │
      No
       │
       ▼
Run YOLO Tracking
       │
       ▼
Display Tracked Objects
```

---

## Logging

The application provides detailed logging for:

* Motion events
* Camera shifts
* Camera drops
* Camera restart attempts
* Tracking activation
* Camera signal validation

Example:

```
WARNING - Motion Detected in ROI
WARNING - Camera Shifted
INFO    - Tracking
WARNING - Camera Dropped LEFT
ERROR   - No Camera Signal
```

---

## Current Limitations

* Single camera input
* GUI-based ROI selection
* No event recording
* Limited camera health metrics

---

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

Because this application relies directly on the YOLOv8 architecture, it inherits their open-source copyleft requirements. See the LICENSE file in this repository for full details and usage permissions.