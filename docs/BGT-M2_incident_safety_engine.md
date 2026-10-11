# BGT-M2: Incident Detection & Safety Intelligence Engine

**Milestone:** M2 | **Track:** Safety & Enforcement | **Status:** `COMPLETED`

## 1. Overview
The BGTMS Incident & Safety Intelligence Engine delivers real-time situational awareness across high-density urban intersections. Operating concurrently with multi-object tracking (ByteTrack) and metric homography calibration, it detects collisions, sudden decelerations, stalled vehicles, emergency vehicles, and traffic law violations without ground loop sensors.

---

## 2. Core Detection Modules

### 2.1 Accident & Collision Detector (`bgtms/incidents/accident_detector.py`)
- **Kinematic Anomaly Detection:** Tracks rolling speed history (15 frames). Triggers `SUDDEN_DECELERATION` if $\Delta v > 25.0\text{ km/h}$ into near standstill ($v < 10\text{ km/h}$) within 0.5 seconds.
- **Bounding Box Impact Overlap:** Computes spatial intersection between interacting vehicle bounding boxes. Overlap area ratio $\ge 0.40$ combined with post-impact velocity $< 5.0\text{ km/h}$ flags `COLLISION`.
- **Deduplication Cooldown:** Uses a bidirectional track pair dictionary `(v1, v2)` with a 10.0-second cooldown timer, eliminating alert spamming while preserving true new incidents.
- **Multi-Vehicle Road Blockage:** Aggregates vehicles stalled $> 30\text{ s}$ per approach. When $\ge 2$ vehicles are stalled simultaneously on the same carriageway, raises a `ROAD_BLOCKAGE` alert.

### 2.2 Emergency Vehicle Preemption (`bgtms/incidents/emergency_detector.py`)
- **Visual Classification:** Detects `emergency_vehicle`, ambulances, fire engines, and police units (class ID 7 or semantic class tag).
- **Trajectory & Arrival Estimation:** Projects vehicle center towards the approach stop line:
  $$\text{ETA} = \frac{\text{Distance to Stop Line}}{\max(3.0, v / 3.6)}$$
- **Preemptive Green Corridor Plan:** Automatically generates an override `SignalPlan` with immediate green phase allocated to the approaching arm (default 30–35s green) while holding all conflicting movements in all-red.
- **Corridor Clearance Detection:** Monitors vehicle exit polygon ($y > y_{\text{exit}}$). Once all emergency vehicles traverse the junction, seamlessly returns signal control to Webster adaptive mode.

### 2.3 Traffic Violation Detector (`bgtms/incidents/violation_detector.py`)
- **Red-Light Jump Detection:** Flags vehicles crossing the stop-line ($y_2 > y_{\text{stop}} + 40\text{px}$) with forward velocity $> 5.0\text{ km/h}$ during active Red phases.
- **Stop-Line / Zebra Encroachment:** Distinguishes vehicles stopping forward over the stop line ($y_{\text{stop}} < y_2 \le y_{\text{stop}} + 40\text{px}$, $v < 5\text{ km/h}$) occupying pedestrian crossing space.
- **Wrong-Way Driving:** Compares trajectory vector displacement $\Delta y$ over 6 frames against approach flow direction. $\Delta y < -35\text{px}$ with $v > 7\text{ km/h}$ raises immediate critical alert.
- **Urban Speeding:** Flags vehicles exceeding corridor speed limits ($v > 50\text{ km/h}$).

---

## 3. Automated Validation Suite Results
All incident detectors are verified via `tests/test_incidents.py`:
- `test_sudden_deceleration_detection`: **PASS** (Speed drop 45 $\to$ 3 km/h triggers `WARNING`)
- `test_collision_detection`: **PASS** (Car-motorcycle overlap triggers `CRITICAL`)
- `test_stalled_vehicle_detection`: **PASS** (Zero velocity for 25 frames triggers `STALLED_VEHICLE`)
- `test_emergency_vehicle_detection_and_preemption`: **PASS** (Ambulance inbound triggers green corridor plan)
- `test_emergency_corridor_clearance`: **PASS** (Exit boundary clears corridor state)
- `test_red_light_running_violation`: **PASS** (Stop-line cross in red triggers `RED_LIGHT_VIOLATION`)
- `test_stop_line_encroachment`: **PASS** (Zebra encroachment triggers `STOP_LINE_VIOLATION`)
- `test_wrong_way_violation`: **PASS** (Inverted trajectory vector triggers `WRONG_WAY`)

**Overall Incident Suite: 8 / 8 Tests Passed (100%)**
