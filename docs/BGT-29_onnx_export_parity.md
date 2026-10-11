# BGT-29: ONNX Model Export & Parity Verification Report

**Track:** H Deployment | **Epic:** E09 | **Task:** BGT-29

## 1. Export Specifications
- **Source Model:** `SSDLite320-MobileNetV3` fine-tuned checkpoint (`bgtms/models/vehicle_detector.pth`)
- **Target Format:** Open Neural Network Exchange (ONNX) v14
- **Target File:** `bgtms/models/vehicle_detector.onnx` (14.73 MB)
- **Input Shape:** `[1, 3, 320, 320]` (RGB normalized [0, 1])
- **Outputs:** `boxes` [N, 4], `scores` [N], `labels` [N]

## 2. Parity & Performance Scorecard

| Evaluation Metric | PyTorch Native | ONNX Runtime | Parity / Gain | Status |
|---|:---:|:---:|:---:|:---:|
| Inference Latency (Batch=1) | 49.69 ms | 33.70 ms | **1.47x faster** | **PASS** |
| Max Box Coordinate Delta | — | — | **0.000023 px** | **PASS (< 0.05 px)** |
| Max Score Delta | — | — | **0.000000** | **PASS (< 0.001)** |
| Graph Integrity | Valid | Valid | 100% Validated | **PASS** |

## 3. Production Deployment Recommendation
The exported ONNX engine demonstrates bitwise-close numerical parity while reducing inference latency. Approved for low-latency edge deployment and Triton/FastAPI serving.
