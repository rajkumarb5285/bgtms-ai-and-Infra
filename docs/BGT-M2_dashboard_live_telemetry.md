# BGT-M2: Live Command Center Dashboard & Real-Time Telemetry

**Milestone:** M2 | **Track:** Operator Interface & Telemetry | **Status:** `COMPLETED`

## 1. Overview
The BGTMS Command & Control Dashboard provides CCTV operators, traffic police dispatchers, and city planners with an ultra-responsive, real-time command interface. It streams live synthetic/camera detections at 10 Hz over WebSockets, renders live perspective tracks, presents microscopic 3-lane queues and PCU densities, displays AI congestion forecasting horizons, and alerts operators to accidents and emergency vehicles with one-click green corridor overrides.

---

## 2. Interface Features & Architecture

### 2.1 Live CCTV Stream & Dynamic Canvas
- **Resolution:** 1280x720 60 FPS HTML5 Canvas.
- **Perspective Road Overlay:** Dynamic 3-lane trapezoid geometry calibrated for Silk Board Hosur Road approach.
- **Vehicle Rendering:** Bounding boxes, class tags, speed badges ($km/h$), and flashing visual beacons for ambulances/emergency units.
- **Interactive Camera Selector:** Switches between Hosur Road Approach (North), Electronic City Flyover (South), HSR Layout ORR (East), and BTM Layout (West).

### 2.2 Microscopic Lane Telemetry
- Three lane cards displaying:
  - Total Vehicle Count and IRC Passenger Car Units (PCU)
  - Color-coded percentage occupancy progress tracks
  - Stop-line queue length in meters
  - Average lane speed ($km/h$)

### 2.3 Adaptive Signal Timing Dial & Countdown
- Real-time animated cycle clock showing remaining seconds in active phase.
- Color shifts dynamically matching signal state:
  - Phase 1 (North-South): Emerald Green
  - Phase 2 (East-West): Amber Yellow
- Summarizes Webster cycle length (94.0s) and overall junction delay reduction (-34.8%).

### 2.4 Incident & Emergency Dispatch Feed
- Real-time event log with color-coded severity tiers (`CRITICAL`, `WARNING`, `INFO`).
- Interactive simulation buttons:
  - `+ Accident`: Simulates two-vehicle collision with abrupt speed drop.
  - `+ Ambulance`: Simulates inbound emergency vehicle and activates corridor alerts.
- Top navigation **🚨 TRIGGER GREEN CORRIDOR** toggle: forces emergency preemption across all intersection arms.

---

## 3. Communication Protocol
- Primary Channel: WebSocket at `ws://localhost:8000/ws/live-telemetry` broadcasting 10 Hz JSON telemetry payloads.
- Automatic Fallback: Polls REST API at `/api/v1/traffic/state` and `/api/v1/incidents` if WebSocket is temporarily interrupted.
