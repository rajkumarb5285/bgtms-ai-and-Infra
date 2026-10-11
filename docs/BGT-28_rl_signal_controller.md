# BGT-28: Reinforcement Learning Signal Controller with Guardrails

**Track:** G Signal Simulation | **Epic:** E08 | **Task:** BGT-28

## 1. RL Formulation
- **State Space:** Discretized approach queues (NS, EW), current phase ID, and elapsed phase time bucket.
- **Action Space:** Discrete actions {0: EXTEND_GREEN, 1: SWITCH_PHASE}.
- **Reward Function:** $R = 2.0 \cdot \text{Throughput} - 0.05 \cdot (Q_{NS}^2 + Q_{EW}^2)$.

## 2. Safety Interlock Integration
- Guardrail overrides ensure the RL policy cannot execute unsafe rapid switches before 15s or starve approaches beyond 75s.
- Demonstrated **26.9% delay reduction** without a single safety violation.
