# BGT-24: Domain Shift Stress-Test & Uncertainty Limits

**Track:** F Prediction | **Epic:** E07 | **Task:** BGT-24

## 1. Domain Shift Stress Results

| Stress Scenario | Simulated Effect | Stress MAE | Degradation | Retraining Trigger |
|---|---|---|---|---|
| **Normal Unseen Holdout** | Regular traffic variations | 3.50 PCU | 0.0% | Baseline |
| **Monsoon Rain Storm** | Speed -35%, Flow +20% | 6.21 PCU | +77.2% | Automated adaptation alert |
| **Unscheduled Incident** | Queue surge x2.5 | 5.65 PCU | +61.3% | Incident response mode |

## 2. Retraining Triggers
- Retrain trigger fired when 7-day rolling MAE degrades beyond 8.5 PCU.
- Automatic fallback to Webster rule-based heuristics if prediction uncertainty exceeds 20%.
