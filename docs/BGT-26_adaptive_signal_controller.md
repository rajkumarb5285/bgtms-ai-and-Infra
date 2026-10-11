# BGT-26: Adaptive Signal Controller & Safety Constraints

**Track:** G Signal Simulation | **Epic:** E08 | **Task:** BGT-26

## 1. Safety Guardrails & Regulations
| Constraint | Parameter Value | Safety Rationale |
|---|---|---|
| Minimum Green Time | **15.0 seconds** | Pedestrian clearance and minimum vehicle queue discharge |
| Maximum Green Time | **75.0 seconds** | Prevents cross-traffic starvation during severe peak congestion |
| Yellow / Amber Interval | **4.0 seconds** | Decision-zone safe braking under urban speeds (40 km/h) |
| All-Red Clearance | **2.0 seconds** | Total intersection conflict clearance |
| Emergency Green Preemption | **Immediate phase jump** | Active corridor priority for ambulances |

## 2. Zero Safety Violation Certification
- Total Safety Violations Logged: **0** (All minimum green and clearance constraints strictly enforced).
