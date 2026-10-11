# BGT-34: Shadow Pilot Deployment & Review Protocol

**Track:** I Acceptance | **Epic:** E10 | **Task:** BGT-34

## 1. Pilot Scope & Governance
- **Deployment Site:** Central Silk Board Junction, Hosur Road approach (Bengaluru).
- **Mode:** Passive Shadow Pilot (AI analyzes live stream, logs telemetry, predicts signal plans, and flags incidents without actuating hardware physical signal relays).
- **Safety Interlock:** Physical signal relay control remains strictly isolated behind manual operator authorization.

## 2. Privacy & Compliance Controls
1. **Automated Anonymization:** Face detection and license plate areas are Gaussian-blurred at edge ingestion prior to cloud transmission.
2. **Access Control:** AWS IAM least-privilege role policies enforce encryption at rest (KMS) and in-transit (TLS 1.3).
3. **Data Retention:** Raw video feeds are purged after 24 hours; only aggregated anonymized telemetry is stored in S3 analytical manifests.

## 3. Human-in-the-Loop Review Queue
- Incident alerts (accidents, collisions, red light violations) are routed to the Traffic Police Command Review Queue.
- Operator verification interface allows accepting, rejecting, or escalating detected events with audit logging.
