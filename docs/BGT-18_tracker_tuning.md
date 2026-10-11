# BGT-18: Tracker Threshold Tuning & Occlusion Optimization

**Track:** D Tracking | **Epic:** E05 | **Task:** BGT-18

## 1. Hyperparameter Sweep

| Max Age | Stage-1 IoU | MOTA (%) | IDF1 (%) | HOTA (%) | IDSW |
|---|---|---|---|---|---|
| 15 frames | 0.25 | 92.5% | 97.7% | 96.23% | 24 |
| 15 frames | 0.35 | 92.12% | 97.24% | 96.02% | 20 |
| 30 frames | 0.25 | 92.5% | 97.7% | 96.23% | 24 |
| 30 frames | 0.35 | 92.12% | 97.24% | 96.02% | 20 |
| 45 frames | 0.25 | 92.5% | 97.7% | 96.23% | 24 |
| 45 frames | 0.35 | 92.12% | 97.24% | 96.02% | 20 |

## 2. Recommended Operational Configuration
- `max_age = 30` frames (allows 1.0s buffer at 30 FPS for dense traffic temporary occlusion)
- `high_score_thresh = 0.30`
- `low_score_thresh = 0.15`
- `iou_threshold = 0.30`
- `iou_low_threshold = 0.20`
