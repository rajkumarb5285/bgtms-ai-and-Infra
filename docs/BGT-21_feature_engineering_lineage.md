# BGT-21: Feature Engineering & Time-Aware Data Lineage

**Track:** E Traffic Analytics | **Epic:** E06 | **Task:** BGT-21

## 1. Feature Specifications
- **Temporal Lags:** Lags at t-1 (5m), t-2 (10m), t-3 (15m), and t-6 (30m) for PCU, speed, queue, and occupancy.
- **Rolling Windows:** 30-minute rolling mean and standard deviation capturing trend acceleration.
- **Diurnal Context:** Hour of day, day of week, weekend indicator, and Bengaluru peak-hour flags.

## 2. Lineage & Anti-Leakage Protocol
- **No Future Information:** All rolling statistics strictly shift by 1 step before window calculation.
- **Temporal Split:** Strict chronological split without shuffling to mirror production streaming.
