# Lantern Harbor product overview

Lantern Harbor is a fictional incident coordination service for small operations teams. It offers incident timelines, searchable runbooks, service ownership, and status-page drafts. It never applies production changes automatically.

## Product tiers

The Cove tier supports 5 operators, 10 monitored services, and 30 days of incident-history retention. It includes runbook search and email notifications. Its availability target is 99.5% monthly.

The Beacon tier supports 25 operators, 50 monitored services, and 90 days of incident-history retention. It adds audit exports, role-based access, and on-call escalation schedules. Its availability target is 99.9% monthly.

The Lighthouse tier supports 100 operators and 200 monitored services. It retains incident history for 180 days and adds SAML single sign-on and dedicated support. Its availability target is 99.95% monthly. All tiers use the same encryption baseline described in security_policy.md.

## Limits and recovery

The maximum runbook attachment is 10 MB. Supported formats are UTF-8 text, Markdown, and text-based PDF. The service does not read scanned images. Status-page drafts need human approval before publication.

The service-wide recovery time objective (RTO) is 60 minutes. The recovery point objective (RPO) is 15 minutes. These objectives describe disaster recovery; they do not change the latency incident escalation rules.
