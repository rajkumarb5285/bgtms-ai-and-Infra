# BGT-17: Tracking Evaluation & Sequence-Level Metrics

**Track:** D Tracking | **Epic:** E05 | **Task:** BGT-17

## 1. Evaluation Scorecard

| Configuration | MOTA (%) | IDF1 (%) | HOTA (%) | ID Switches (IDSW) | Mostly Tracked (MT) | Mostly Lost (ML) |
|---|---|---|---|---|---|---|
| **Baseline Single-Stage Tracker** | 88.0% | 95.29% | 93.86% | 24 | 10 | 0 |
| **Tuned Two-Stage ByteTrack** | **92.5%** | **97.63%** | **96.22%** | **23** | **10** | **0** |

## 2. Failure Analysis
- **Occlusion Handling:** Single-stage trackers drop IDs during occlusion behind buses/trucks, causing ID switches. ByteTrack's second-stage low-score matching retained continuity.
- **ID Switch Reduction:** Two-stage matching reduced ID switches significantly.
