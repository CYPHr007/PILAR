# PILAR — Business Model v1

## Target Clients
Facilities with existing sensors on any industrial machine (or we advise on which sensors to buy — they handle installation).
Industries where machine failure = expensive downtime: chemical, food, water treatment, oil & gas, mining, manufacturing, robotics.

## Value Proposition
Affordable predictive maintenance tool for any industrial machine — easy enough for anyone on the team, ideally supervised by someone close to or part of the maintenance team. Prevents breakdowns before production stops.

PILAR is machine-agnostic: ships with a hydraulic/pump default model, and can be retrained on any equipment type using client sensor data.

We only provide the analysis/prediction tool. Clients own their hardware.

---

## Pricing

| Machines | Monthly |
|---|---|
| 1 machine | $750/month |
| 3 machines | $2,250/month |
| 10 machines | $7,500/month |

**Contract:** 6-month minimum
**Onboarding:** 2 weeks free (data collection + team familiarization)

---

## Guarantee
If PILAR makes a verified prediction error → **1 free month** credited.
Client must provide evidence. Builds trust, shows we stand behind the product.

---

## End-of-Contract
- Deliver a **6-month data summary report**: breakdowns avoided, estimated cost saved, machine health trends.
- Use report as lever to renew contract and expand to more machines.

---

## Growth Path Per Client
```
2 weeks free onboarding
→ 6 months × 3 machines  ($13,500)
→ renewal + expand        ($27,000+/6mo)
→ full factory coverage
```

---

## Unit Economics (targets)

| Clients | Avg machines | Monthly revenue |
|---|---|---|
| 5  | 3 | $11,250  |
| 10 | 5 | $37,500  |
| 20 | 5 | $75,000  |

---

## Next Steps
- [ ] Machine selection: define which machines/sensors PILAR supports out-of-the-box vs. retrain-required
- [ ] Onboarding process: sensor check, data connection, team training
- [ ] First 5 clients: outreach strategy (kit already built in /outreach)

---

## Product Roadmap — Embedded / Robot Direction

**PILAR Embedded** is the next product scale: the software runs inside an autonomous machine or robot rather than on a separate server.

How it works:
- PILAR runs on-device, continuously reading the machine's own sensor data
- It builds a behavioral baseline specific to that machine's normal operation
- When anomalous patterns are detected, it identifies the affected zone (mechanical, electrical, thermal, etc.)
- The machine itself — or its controller — automatically triggers a maintenance or inspection request for that zone, without waiting for a human to notice

Target: robot manufacturers, autonomous vehicle fleets, automated factory lines — any machine that needs to self-report its own health and request service when something drifts.

Pricing model for embedded: per-unit license or per-device SDK fee (to be defined once first factory partner is onboarded).
