# BGT-32: Live Traffic Command & Control Dashboard Demo

**Track:** H Deployment | **Epic:** E09 | **Task:** BGT-32

## 1. Overview
Constructed live browser command-centre dashboard displaying real-time video overlay, ByteTrack tracking trajectories, lane-by-lane microscopic density metrics, 5/15/30-minute congestion forecasting, incident & accident alerts, and active Webster adaptive signal phase controllers.

## 2. Dashboard Components & Modules

| Module | Interface Element | Description |
|---|---|---|
| **CCTV Vision Stream** | Canvas Bounding Boxes | Live vehicle bounding boxes, class badges, velocities, and tracking trails on recorded traffic footage. |
| **Microscopic Lane Metrics** | Lane Progress Gauges | PCU count, speed (km/h), queue length (m), and occupancy percentage per lane. |
| **Adaptive Signal Controller** | Traffic Light Signal Visualizer | Active green phase indicator, countdown timer, cycle time, and phase duration allocations. |
| **AI Congestion Forecaster** | Horizon Trend Cards | 5-minute, 15-minute, and 30-minute predicted PCU flow and congestion risk tiers. |
| **Incident Response Center** | Real-Time Alert Feed | Red-light violation alerts, stalled vehicle notifications, and accident warnings. |
| **Emergency Green Corridor** | Priority Preemption Action | One-click emergency ambulance priority trigger routing green lights across Silk Board North/South. |

## 3. Deployment & Access
- Hosted directly via FastAPI Static Mount: `http://localhost:8000/dashboard/`
- WebSocket Telemetry Link: `ws://localhost:8000/ws/live-telemetry` (10 Hz low-latency updates)
