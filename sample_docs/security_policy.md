# Lantern Harbor security policy

This is an original fictional policy for demonstration, not a compliance certification.

## Encryption

Customer incident records, runbooks, and backups must use AES-256 encryption at rest. Data in transit must use TLS 1.2 or newer; TLS 1.3 is preferred. Operators must not send production API keys or passwords in incident comments. Store credentials only in the approved secret manager.

## Retention and deletion

Incident history follows the product tier: Cove 30 days, Beacon 90 days, and Lighthouse 180 days. Security audit events are retained for 365 days for every tier. Backups expire after 14 days. A customer deletion request removes primary incident records within 72 hours; backup copies expire on the normal 14-day schedule.

## Access controls

Privileged access requires multi-factor authentication. Review privileged access every 30 days. Temporary incident elevation expires after 4 hours. Only the incident commander may approve emergency elevation, and the approval must appear in the audit log.

## Reporting

Report suspected credential exposure to the security on-call immediately and revoke the affected credential within 15 minutes. Do not include the credential value in the escalation message. The security lead owns follow-up review within 2 business days.
