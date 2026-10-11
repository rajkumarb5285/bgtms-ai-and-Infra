# BGT-35: Active Learning Loop, Retraining & Model Registry Gate

**Track:** I Acceptance | **Epic:** E10 | **Task:** BGT-35

## 1. Active Learning Sampling Policy
To continuously improve detection of rare Indian vehicle variants (tempo travelers, overloaded trucks, non-standard auto-rickshaws):
1. **Uncertainty Mining:** Frames with detections exhibiting confidence $0.20 \le \text{conf} \le 0.35$ are flagged for human annotation.
2. **High-Loss Event Mining:** Frames causing tracker ID switches or sudden trajectory drop-offs are sampled into the retraining candidate pool.
3. **Hard-Negative Ingestion:** False alarms identified by traffic operators in the shadow review queue are automatically labeled as negative background patches.

## 2. Model Registry Promotion Gate
Before promoting any new retrained model checkpoint (`model_v{N+1}`) to production:
- **Rule 1 (Frozen Test Regression Gate):** Must match or exceed current mAP/Precision on the frozen test set (`data/splits/test.txt`).
- **Rule 2 (Slice Invariance):** Must maintain zero false positive rate on `slice_hard_negatives`.
- **Rule 3 (Latency SLA):** P95 inference latency must remain $\le 50.0\text{ ms}$ on edge hardware.
- **Rule 4 (Automated Rollback):** Previous model checkpoint weights (`.pth` and `.onnx`) are versioned in `s3://bgtms-ai-datasets/models/archive/` allowing 1-click fallback.
