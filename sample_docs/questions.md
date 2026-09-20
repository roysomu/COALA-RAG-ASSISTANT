# Demonstration questions and expected sources

This file is a test guide and is deliberately excluded from automatic sample ingestion.

1. How many operators and services does Beacon support? Expected: 25 operators, 50 services. Source: product_overview.md.
2. Compare incident retention across all three tiers. Expected: 30, 90, 180 days. Source: product_overview.md and security_policy.md.
3. What encryption is required at rest and in transit? Expected: AES-256; TLS 1.2 or newer, prefer 1.3. Source: security_policy.md.
4. When should a latency incident be declared SEV-2? Expected: p95 above 800 ms for 10 minutes. Source: incident_runbook.md.
5. When does latency escalate to SEV-1, and who is paged? Expected: p95 >2 s for 5 minutes or errors >5% for 5 minutes; commander and database on-call. Source: incident_runbook.md.
6. What are the recovery time and recovery point objectives? Expected: 60 and 15 minutes. Source: product_overview.md or incident_runbook.md.
7. Compare incident, audit, and backup retention for Beacon. Expected: 90 days, 365 days, 14 days. Sources: product_overview.md and security_policy.md.
8. What should happen when database connections exceed 85%? Expected: reduce concurrency 25%, reassess after 5 minutes. Source: incident_runbook.md.
9. Who approves emergency privileged access, and when does it expire? Expected: incident commander; 4 hours. Source: security_policy.md.
10. What signals are required before considering latency mitigated? Expected: p95 <400 ms and errors <1% for 15 minutes. Source: incident_runbook.md.
11. What is Lantern Harbor's annual price? Expected: insufficient evidence; none of the sources states pricing.
