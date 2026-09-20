# Lantern Harbor latency incident runbook

## Detection and initial action

Declare a SEV-2 latency incident when API p95 latency exceeds 800 ms for 10 consecutive minutes. The primary on-call acknowledges the alert within 5 minutes and opens an incident timeline. Check recent deployments, database connection saturation, and upstream error rates before choosing mitigation.

## Mitigation and escalation

If latency begins within 20 minutes of a deployment and rollback is safe, ask the incident commander to approve rollback. Never bypass approval. If database connections exceed 85% of the limit, reduce worker concurrency by 25% and reassess after 5 minutes.

Escalate to SEV-1 if p95 latency exceeds 2 seconds for 5 consecutive minutes, or the error rate exceeds 5% for 5 minutes. Page the incident commander and database on-call together. Send a customer status update every 15 minutes while SEV-1 is active.

## Recovery

Consider the incident mitigated when p95 latency stays below 400 ms and errors stay below 1% for 15 consecutive minutes. Restore worker concurrency in 10% increments, observing for 5 minutes after each increase.

The disaster recovery RTO is 60 minutes and RPO is 15 minutes. Record the mitigation timeline and assign a review owner before closing. Complete the incident review within 2 business days. Keep incident records according to the customer's tier; security audit logs follow their separate 365-day retention rule.
