# M1-03: High-Level System Architecture — BGTMS

## 1. End-to-End System Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                 BENGALURU CCTV / RTSP VIDEO FEEDS                                 |
|                     (Silk Board | Sony World | Marathahalli | Hebbal Overheads)                   |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                      DATA INGESTION & PIPELINE                                    |
|              - Frame Grabber / Video Stream Decoder (OpenCV / PyAV)                               |
|              - Dynamic Resize & Normalization (640x640)                                           |
|              - ROI Cropping & Perspective Calibration Mapping                                     |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
+---------------------------------------------------+     +-----------------------------------------+
|             COMPUTER VISION ENGINE                |     |            INCIDENT AI ENGINE           |
|  - Detection: YOLO Indian Vehicle Detector        |     |  - Accident / Crash Detection           |
|    (Car, Bus, Truck, Bike, Auto, Pedestrian)      |     |  - Sudden Stop / Stalled Vehicle Alarm  |
|  - Multi-Object Tracker: ByteTrack / Kalman       |     |  - Traffic Violations (Red-light/Wrong) |
|  - Trajectory Smoothing & Velocity Estimation     |     |  - Emergency Vehicle Preemption         |
|  - Polygonal Lane Detection & Vehicle Mapping     |     |    (Ambulance Green Corridor Trigger)   |
+------------------------+--------------------------+     +--------------------+--------------------+
                         |                                                     |
                         v                                                     |
+---------------------------------------------------+                          |
|             TRAFFIC INTELLIGENCE ENGINE           |                          |
|  - Per-Lane Vehicle Count, Density & Occupancy    |                          |
|  - Indian Roads Congress PCU Equivalency Calc     |                          |
|  - Queue Length & Approach Shockwave Detection    |                          |
|  - Real-Time Congestion Classifier                |                          |
|    (Low / Medium / High / Critical)               |                          |
+------------------------+--------------------------+                          |
                         |                                                     |
                         v                                                     |
+---------------------------------------------------+                          |
|               PREDICTIVE AI ENGINE                |                          |
|  - Temporal Forecast Models (5 / 15 / 30 min)     |                          |
|  - Bottleneck Propagation Forecasting             |                          |
+------------------------+--------------------------+                          |
                         |                                                     |
                         +------------------------+----------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                               SIGNAL INTELLIGENCE & OPTIMIZATION                                  |
|  - Webster Formula Optimal Cycle & Phase Duration Allocator                                       |
|  - Reinforcement Learning (Q-Learning / DQN) Adaptive Signal Timing Controller                    |
|  - SUMO (Simulation of Urban MObility) Closed-Loop Micro-Simulation Environment                   |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                    PRODUCTION BACKEND & API                                       |
|  - FastAPI Service (Async REST endpoints & WebSocket Real-time Broadcast)                        |
|  - Storage: In-memory cache + Redis + SQLite/PostgreSQL historical telemetry                      |
|  - Alert Dispatcher: Webhook / SMS / BTP Control Room Push                                       |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                             COMMAND & CONTROL REAL-TIME DASHBOARD                                 |
|  - Modern Dark-Themed Web Interface (HTML5 Canvas + WebSockets + Chart.js)                        |
|  - Live Annotated CCTV Visualizer with Bounding Boxes, Lanes & Speed Vectors                       |
|  - Real-time Phase Dial, Emergency Corridor Override, Junction Congestion Heatmap                 |
+---------------------------------------------------------------------------------------------------+
```

## 2. Component Interactions & Data Flow
1. **Frames:** CCTV stream decoded at ~30 FPS -> processed through `bgtms.vision.detector` and `bgtms.vision.tracker`.
2. **Metrics:** Bounding boxes and velocity vectors mapped against lane geometry in `bgtms.traffic.density_engine` to derive Passenger Car Units (PCU) and queue lengths.
3. **Safety & Incidents:** Concurrently, trajectory changes and bounding box intersections evaluated in `bgtms.incidents`.
4. **Adaptive Signals:** Dynamic green durations calculated via `bgtms.signal.webster_optimizer` and validated against SUMO traffic light controller.
5. **Broadcasting:** Fast telemetry push to operators via WebSocket at 10 Hz.
