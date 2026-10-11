# M1-02: System Requirements Specification — BGTMS

## 1. System Inputs & Ingestion Requirements
- **Video Input Feeds:**
  - Standard RTSP/H.264/H.265 CCTV feeds from intersection overhead cameras (1080p / 720p at 25–30 FPS).
  - Video resolution: 1920x1080 native, downscaled to 640x640 or 1280x720 for deep learning inference.
- **Camera Calibration Data:**
  - 4-point homography polygon per approach for metric conversion (pixels -> real-world meters).
  - Approaching stop-line ROIs and lane delineation polygonal masks.
- **External Data Ingestion:**
  - Amazon S3 bucket storage (`s3://bgtms-ai-datasets`) containing training datasets (IDD, Indian Vehicle Dataset, UA-DETRAC).

## 2. Functional Requirements
### FR-1: Heterogeneous Object Detection
- The system shall detect 8 primary classes: `car`, `motorcycle`, `auto_rickshaw`, `bus`, `truck`, `bicycle`, `pedestrian`, and `emergency_vehicle`.
- Detection confidence threshold must be dynamically configurable (default: 0.35).

### FR-2: Trajectory & Multi-Object Tracking
- The system shall assign persistent track IDs using Kalman filtering and Hungarian/ByteTrack association.
- Trajectory histories must be retained for at least 300 frames per active vehicle.

### FR-3: Metric Speed & Homography Estimation
- The system shall calculate instantaneous and smoothed speeds ($km/h$) using calibrated homography matrices without ground sensor hardware.

### FR-4: Indian Passenger Car Unit (PCU) & Traffic Density Engine
- The system shall compute per-lane occupancy, vehicle counts, flow rates ($PCU/hr$), and traffic density using IRC (Indian Roads Congress) standards:
  - 2-Wheeler (Motorcycle/Scooter): 0.50 PCU
  - 3-Wheeler (Auto-Rickshaw): 1.00 PCU
  - Passenger Car: 1.00 PCU
  - Bus (BMTC): 3.00 PCU
  - Heavy Commercial Truck: 3.00 PCU
  - Bicycle: 0.40 PCU

### FR-5: Queue Length & Shockwave Analysis
- The system shall measure queue length in meters from stop-line backwards, count stopped vehicles ($v < 3\text{ km/h}$), and flag queue growth shockwaves.

### FR-6: Machine Learning Congestion Classifier & 5/15/30-min Predictor
- Traffic congestion states shall be classified into: `LOW` (Green), `MEDIUM` (Yellow), `HIGH` (Orange), `CRITICAL` (Red).
- The prediction module shall forecast traffic density and congestion levels at 5, 15, and 30-minute horizons.

### FR-7: Incident & Violation Detection
- Real-time alerts for:
  - Collisions / sudden decelerations ($\Delta v > 25\text{ km/h}$ in $< 0.5\text{ s}$)
  - Stalled vehicle / lane obstruction ($> 45\text{ s}$ motionless in moving phase)
  - Red-light violation & stop-line encroachment
  - Wrong-way driving trajectory anomalies

### FR-8: Emergency Vehicle Detection & Green Corridor Automation
- Detect sirens/ambulances, trigger instantaneous green light phase on approaching arm, hold conflicting movements in all-red state.

### FR-9: Signal Optimization & SUMO Integration
- Calculate Webster's minimum-delay cycle time:
  $$C_0 = \frac{1.5L + 5}{1 - Y}$$
  where $L$ is total lost time and $Y = \sum y_i$ is critical flow ratio.
- Adapt phase durations dynamically based on real-time PCU demand.

### FR-10: Central API & Command-Center Dashboard
- FastAPI backend serving REST APIs and low-latency WebSockets.
- Responsive, dark-themed real-time web command center for CCTV operators.

## 3. Non-Functional Requirements
- **Latency:** Real-time CV inference $\le 35\text{ ms}$, API response $\le 50\text{ ms}$, WebSocket push $\le 100\text{ ms}$.
- **Reliability:** 99.9% uptime with automatic failover to fixed-time fallback signal cycles if camera feed drops.
- **Portability:** Containerized via Docker & Docker Compose for deployment on edge (NVIDIA Jetson AGX) or AWS EC2 (g5.xlarge).
