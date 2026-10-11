# BGT-22: Traffic Forecasting Baseline Benchmark

**Track:** F Prediction | **Epic:** E07 | **Task:** BGT-22

## 1. Model Scorecard (15-Minute Forecast Horizon)

| Model | MAE (PCU) | RMSE | R² Score | Baseline Beat? |
|---|---|---|---|---|
| **Naive Persistence Baseline** | 5.43 | 6.75 | 0.856 | Baseline |
| **Gradient Boosting Regressor** | **3.36** | **4.27** | **0.942** | **YES (+38.2%)** |
| **Random Forest Regressor** | **3.50** | **4.53** | **0.935** | **YES (+35.5%)** |

## 2. Classifier Benchmark
- **Random Forest Congestion Classifier Accuracy:** **80.72%**
