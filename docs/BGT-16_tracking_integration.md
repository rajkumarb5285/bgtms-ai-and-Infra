# BGT-16: ByteTrack Multi-Object Tracking Integration

**Track:** D Tracking | **Epic:** E05 | **Task:** BGT-16

## 1. Overview
Integrated two-stage ByteTrack algorithm with SSDLite vehicle detector outputs. Vehicle bounding boxes and classes are matched across continuous traffic video sequences.

## 2. Key Features
- **Two-Stage Bipartite Association:** High-score detections matched first; low-score detections matched in second pass for occlusion recovery.
- **Kalman Motion Predictor:** Constant velocity model with alpha-beta trajectory estimation.
- **Stable Track Lifecycles:** Active, probationary, and pruned states with configurable `max_age` and `min_hits`.
