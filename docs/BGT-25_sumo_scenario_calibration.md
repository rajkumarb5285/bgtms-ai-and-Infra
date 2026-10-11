# BGT-25: SUMO Intersection Scenario & Demand Calibration

**Track:** G Signal Simulation | **Epic:** E08 | **Task:** BGT-25

## 1. Network Geometry & Topography
- **Junction:** Central Silk Board Junction (4-Arm Junction: North Hosur Rd, South Electronic City, East HSR Layout, West BTM Layout).
- **Files:** `configs/sumo/junction.nod.xml`, `junction.edg.xml`, `junction.con.xml`, `junction.rou.xml`, `junction.sumocfg`.

## 2. Traffic Demand Calibration
- Calibrated heterogeneous Indian traffic composition:
  - Two-Wheelers: Acceleration 2.8 m/s², Length 1.8 m
  - Auto-Rickshaws: Acceleration 1.8 m/s², Length 2.6 m
  - Passenger Cars: Acceleration 2.2 m/s², Length 4.2 m
  - City Buses & Trucks: Acceleration 1.0 m/s², Length 11.5 m
  - Emergency Ambulances: Priority route preemptions
