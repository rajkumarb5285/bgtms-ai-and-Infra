# M1-01: Problem Definition — Bengaluru Traffic Management System (BGTMS)

## 1. Executive Summary
Bengaluru suffers from severe urban traffic congestion, resulting in an estimated ₹19,725 crore ($2.4B) annual economic loss from lost productivity, fuel wastage, and air pollution. Key junctions such as Silk Board, Hebbal Flyover, Tin Factory, and Sony World Signal experience peak-hour queue delays exceeding 30–45 minutes with average corridor speeds dropping below 8 km/h.

Traditional fixed-time and loop-detector based traffic signals fail in Bengaluru due to:
1. **High vehicle heterogeneity:** Unsegregated mix of 2-wheelers (motorcycles, scooters: ~70% of vehicle population), 3-wheelers (auto-rickshaws), passenger cars, BMTC buses, light commercial vehicles (LCVs), and heavy trucks.
2. **Lack of lane discipline:** Continuous lane filtering and non-standard vehicle spacing violate conventional Western traffic assumptions.
3. **Static phase splits:** Fixed timer signals cannot adapt dynamically to sudden congestion shockwaves, peak surges, rain-induced slowdowns, or emergency vehicles.
4. **Delayed incident detection:** Stalled vehicles, minor collisions, and breakdown obstructions take 15–30 minutes to detect manually, cascading into gridlock.

## 2. Target Corridor and Junctions
- **Primary Demonstration Junction:** Central Silk Board Junction (Hosur Road / Outer Ring Road convergence)
- **Secondary Nodes:**
  - Sony World Signal (Koramangala 80ft Road / 100ft Road)
  - Marathahalli Bridge (ORR)
  - Hebbal Junction (Airport Corridor / Bellary Road)

## 3. Project Objectives
- **Accurate Real-Time Detection:** Detect and classify heterogeneous vehicles (cars, buses, trucks, motorcycles, autos, bicycles, pedestrians, ambulances) at ≥ 25 FPS under varying lighting and weather conditions.
- **Microscopic Traffic Analytics:** Real-time computation of per-lane density, Passenger Car Units (PCU), vehicle count, flow rates, queue lengths, and waiting time.
- **Incident & Emergency Preemption:** Detect accidents, breakdowns, wrong-way violations, and emergency vehicles within 3 seconds, triggering priority "Green Corridors".
- **Dynamic Signal Optimization:** Reduce average junction delay by 25–40% compared to fixed-time signals using Webster's optimal cycle calculation and reinforcement learning adaptation in SUMO.
- **Central Command & Control:** Deliver a sub-second live telemetry dashboard and REST/WebSocket API suite for Bengaluru Traffic Police (BTP) operations.

## 4. Key Performance Indicators (KPIs)
| KPI Category | Metric | Baseline (Fixed Time) | BGTMS Target |
|---|---|---|---|
| **Detection Quality** | mAP@50 (heterogeneous Indian classes) | — | ≥ 88.0% |
| **Inference Latency** | Edge / Server pipeline per frame | — | ≤ 35 ms (≥ 28 FPS) |
| **Queue Length** | Average peak queue length per approach | 420 m | ≤ 260 m (-38%) |
| **Junction Delay** | Average stopped delay per vehicle | 94 sec | ≤ 58 sec (-38%) |
| **Throughput** | Junction PCU throughput per hour | 4,200 PCU/h | ≥ 5,450 PCU/h (+30%) |
| **Emergency Clearance** | Priority corridor clearance time | 4.8 min | ≤ 1.5 min (-68%) |
| **Incident Alert Time** | Mean time to detect incident/stoppage | 12–18 min | ≤ 10 sec |
