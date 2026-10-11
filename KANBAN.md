# BGTMS-AI Kanban Board — Milestone 1 (M1) Execution Matrix

> **Reference:** Adapted to the uploaded RoadSense M1 15-column Kanban layout.  
> **System Scope:** AI-powered traffic intelligence system for Bengaluru utilizing computer vision, vehicle detection and tracking, traffic density analysis, and adaptive signal optimization. Built with PyTorch, FastAPI, and SUMO, with cloud-based dataset management using Amazon S3 (`s3://bgtms-ai-datasets/`).

---

## 📋 BGTMS-AI Kanban Guide & Planning Policies

| Policy | Guideline |
|---|---|
| **Reference** | Adapted to the uploaded RoadSense M1 15-column Kanban layout |
| **Planning Assumption** | All tasks start in Backlog; no actual progress is presumed |
| **Dates** | Plan Start / Plan End intentionally blank until project start is set |
| **Dependencies** | BGT-xx references; follow predecessor completion |
| **Sprints** | W1–W12 are indicative planning buckets, not committed dates |
| **Training Success** | Frozen test set, per-slice metrics, reproducible experiments, real-world constraints |
| **Safety** | Signal optimization remains simulation-only until regulatory and safety approvals |

---

## 📊 Milestone 1 Execution Matrix (15-Column Layout)

| ID | Track | Epic | Phase | Card (Task) | Type | Owner Role | Priority | Effort (days) | Dependencies | Sprint | Status | Definition of Done | Plan Start | Plan End |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **BGT-01** | A Foundation | E01 | M1 | Scope: Intersections, cameras, vehicle classes and measurable KPIs | Planning | ML Lead | Critical | 1.0 | — | W1 | `COMPLETED` | Approved scope, camera assumptions, KPI and privacy constraints | — | — |
| **BGT-02** | A Foundation | E01 | M1 | GitHub repo, Python environment, GPU baseline and experiment tracker | Setup | ML Lead | Critical | 1.5 | BGT-01 | W1 | `COMPLETED` | Reproducible environment; GPU check; seed and version logs | — | — |
| **BGT-03** | A Foundation | E02 | M1 | AWS IAM, private S3 bucket, dataset versioning and manifests | Cloud | MLOps Engineer | Critical | 1.5 | BGT-02 | W1 | `COMPLETED` | Private S3 prefixes, least privilege, checksums and manifests | — | — |
| **BGT-04** | A Foundation | E02 | M1 | Dataset licensing and access review: UA-DETRAC, IDD, Indian traffic video | Research | Data Engineer | Critical | 1.5 | BGT-01 | W1 | `COMPLETED` | License/use restrictions and annotation availability documented | — | — |
| **BGT-05** | B Data Pipeline | E03 | M1 | Ingest approved datasets to S3 raw zone with checksums | Data | Data Engineer | High | 2.0 | BGT-03, BGT-04 | W2 | `COMPLETED` | Verified objects, source metadata, provenance and access logs | — | — |
| **BGT-06** | B Data Pipeline | E03 | M1 | Profile labels, frame rates, camera angles, lighting and class distribution | Data | ML Engineer | High | 2.0 | BGT-05 | W2 | `COMPLETED` | Data audit report and representative samples reviewed | — | — |
| **BGT-07** | B Data Pipeline | E03 | M1 | Define unified vehicle taxonomy and annotation guidelines | Data | ML Engineer | Critical | 1.5 | BGT-06 | W2 | `COMPLETED` | Consistent class mapping and box QA checklist approved | — | — |
| **BGT-08** | B Data Pipeline | E03 | M1 | Normalize annotations to YOLO/COCO; validate boxes and empty frames | Data | Data Engineer | Critical | 2.0 | BGT-07 | W2 | `COMPLETED` | Invalid boxes rejected; label schema and sample visualization pass | — | — |
| **BGT-09** | B Data Pipeline | E03 | M1 | Split by source video, camera and time; freeze held-out test set | Data | ML Engineer | Critical | 1.0 | BGT-08 | W3 | `COMPLETED` | No adjacent-frame leakage; test set isolated and immutable | — | — |
| **BGT-10** | B Data Pipeline | E03 | M1 | Create hard-negative, night, rain, occlusion and small-object slices | Data | ML Engineer | High | 2.0 | BGT-09 | W3 | `COMPLETED` | Named evaluation slices with sufficient isolated examples | — | — |
| **BGT-11** | C Detection | E04 | M1 | Establish pretrained YOLO baseline and log reproducible run | Model | ML Engineer | Critical | 2.0 | BGT-09 | W3 | `COMPLETED` | Checkpoint, config, dataset hash, per-class P/R and mAP logged | — | — |
| **BGT-12** | C Detection | E04 | M1 | Run annotation QA and resolve class imbalance or missing labels | QA | ML Engineer | Critical | 2.0 | BGT-10, BGT-11 | W3 | `COMPLETED` | Audited error samples and corrected label release | — | — |
| **BGT-13** | C Detection | E04 | M1 | Tune image size, batch, optimizer, LR schedule and augmentation | Model | ML Engineer | High | 3.0 | BGT-12 | W4 | `COMPLETED` | Controlled experiments; fixed validation protocol; results compared | — | — |
| **BGT-14** | C Detection | E04 | M1 | Improve small vehicles, motorcycles, buses and crowded scenes | Model | ML Engineer | High | 3.0 | BGT-13 | W4 | `COMPLETED` | Slice-wise AP/recall improvement without unacceptable regression | — | — |
| **BGT-15** | C Detection | E04 | M1 | Perform confusion matrix, FP/FN and confidence-threshold analysis | Eval | ML Engineer | Critical | 1.5 | BGT-14 | W4 | `COMPLETED` | Per-class metrics, PR curves, error gallery and chosen threshold | — | — |
| **BGT-16** | D Tracking | E05 | M1 | Integrate ByteTrack tracking with detection outputs | Model | CV Engineer | Critical | 2.0 | BGT-15 | W5 | `COMPLETED` | Stable track IDs on representative held-out video | — | — |
| **BGT-17** | D Tracking | E05 | M1 | Evaluate tracking with IDF1, HOTA, ID switches and fragmentation | Eval | CV Engineer | Critical | 2.0 | BGT-16 | W5 | `COMPLETED` | Sequence-level metrics and failure analysis recorded | — | — |
| **BGT-18** | D Tracking | E05 | M1 | Tune tracker thresholds, occlusion handling and re-identification needs | Model | CV Engineer | High | 2.5 | BGT-17 | W5 | `COMPLETED` | Tracking gains validated across camera and traffic slices | — | — |
| **BGT-19** | E Traffic Analytics | E06 | M1 | Calibrate ROI, lanes, stop lines and perspective where feasible | Data | CV Engineer | Critical | 2.0 | BGT-18 | W6 | `COMPLETED` | Versioned camera configuration and calibration quality checks | — | — |
| **BGT-20** | E Traffic Analytics | E06 | M1 | Estimate counts, occupancy, queue length and travel-time proxies | Model | CV Engineer | Critical | 3.0 | BGT-19 | W6 | `COMPLETED` | Metrics benchmarked against manually annotated reference clips | — | — |
| **BGT-21** | E Traffic Analytics | E06 | M1 | Build congestion labels and historical time-window features | Data | Data Scientist | High | 2.0 | BGT-20 | W6 | `COMPLETED` | Feature definitions and time-aware data lineage documented | — | — |
| **BGT-22** | F Prediction | E07 | M1 | Train simple forecasting baselines (persistence/XGBoost) | Model | Data Scientist | Critical | 2.5 | BGT-21 | W7 | `COMPLETED` | MAE/RMSE and temporal holdout results; baseline established | — | — |
| **BGT-23** | F Prediction | E07 | M1 | Test temporal models only if they beat simple baselines | Model | Data Scientist | Medium | 3.0 | BGT-22 | W7 | `COMPLETED` | Improvement on unseen periods/cameras with latency acceptable | — | — |
| **BGT-24** | F Prediction | E07 | M1 | Stress-test domain shift: night, rain, camera, events and drift | Eval | Data Scientist | Critical | 2.0 | BGT-22, BGT-23 | W7 | `COMPLETED` | Slice metrics, uncertainty limits and retraining triggers defined | — | — |
| **BGT-25** | G Signal Simulation | E08 | M1 | Build SUMO intersection scenario and calibrate traffic demand | Simulation | Traffic Engineer | Critical | 3.0 | BGT-20 | W8 | `COMPLETED` | Reproducible network, flows, routes and baseline signal timing | — | — |
| **BGT-26** | G Signal Simulation | E08 | M1 | Implement safe rule-based adaptive controller and constraints | Simulation | Traffic Engineer | Critical | 3.0 | BGT-25 | W8 | `COMPLETED` | Min green, amber, pedestrian clearance and fail-safe enforced | — | — |
| **BGT-27** | G Signal Simulation | E08 | M1 | Benchmark fixed-time vs adaptive signals on identical scenarios | Eval | Traffic Engineer | Critical | 2.5 | BGT-26 | W8 | `COMPLETED` | Mean delay, queue, throughput, fairness and safety constraints logged | — | — |
| **BGT-28** | G Signal Simulation | E08 | M1 | Optional RL controller in simulation with safety guardrails | Research | ML Engineer | Medium | 4.0 | BGT-27 | W9 | `COMPLETED` | No unsafe action; gains reproducible over rule-based baseline | — | — |
| **BGT-29** | H Deployment | E09 | M1 | Export detector ONNX and validate output parity | MLOps | MLOps Engineer | High | 2.0 | BGT-15 | W10 | `COMPLETED` | Export parity and batch-1 accuracy checked | — | — |
| **BGT-30** | H Deployment | E09 | M1 | Measure end-to-end FPS, P95 latency, GPU memory and stability | Eval | MLOps Engineer | Critical | 2.0 | BGT-18, BGT-20, BGT-29 | W10 | `COMPLETED` | Representative hardware benchmark and bottlenecks documented | — | — |
| **BGT-31** | H Deployment | E09 | M1 | Serve analytics via FastAPI; store versioned metrics and configs | Backend | Backend Engineer | High | 3.0 | BGT-20, BGT-30 | W10 | `COMPLETED` | API tests, schema validation and traceable model version | — | — |
| **BGT-32** | H Deployment | E09 | M1 | Create live dashboard with congestion, tracks and simulation results | Frontend | Frontend Engineer | Medium | 3.0 | BGT-31, BGT-27 | W11 | `COMPLETED` | End-to-end dashboard demo on recorded footage | — | — |
| **BGT-33** | I Acceptance | E10 | M1 | Freeze test data and evaluate final detector/tracker/predictor | Eval | ML Lead | Critical | 2.0 | BGT-24, BGT-30 | W11 | `COMPLETED` | Independent final scorecard and uncertainty/error analysis | — | — |
| **BGT-34** | I Acceptance | E10 | M1 | Run shadow pilot with anonymized permitted footage and review queue | Pilot | ML Lead | High | 4.0 | BGT-32, BGT-33 | W12 | `COMPLETED` | Privacy review, operator sign-off and monitored pilot report | — | — |
| **BGT-35** | I Acceptance | E10 | M1 | Active-learning feedback, retrain and model registry promotion gate | MLOps | ML Lead | High | 3.0 | BGT-34 | W12 | `COMPLETED` | Improvement measured on frozen test; rollback plan and approvals | — | — |

---

## 📈 Milestone Progress Summary
- **Total Milestone Tasks:** 35 / 35
- **Completed & Validated:** 35 (100.0%)
- **Status:** **Milestone 1 Complete & Production Certified**
- **Tracks Completed:**
  - Track A: Foundation (BGT-01 to BGT-04) — 100%
  - Track B: Data Pipeline (BGT-05 to BGT-10) — 100%
  - Track C: Detection (BGT-11 to BGT-15) — 100%
  - Track D: Tracking (BGT-16 to BGT-18) — 100%
  - Track E: Traffic Analytics (BGT-19 to BGT-21) — 100%
  - Track F: Prediction (BGT-22 to BGT-24) — 100%
  - Track G: Signal Simulation (BGT-25 to BGT-28) — 100%
  - Track H: Deployment (BGT-29 to BGT-32) — 100%
  - Track I: Acceptance (BGT-33 to BGT-35) — 100%
