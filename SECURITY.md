# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |

Only the latest release is actively supported with security updates.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT open a public GitHub issue.**
2. Email **security@nqminh.dev** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
3. You will receive an acknowledgment within **48 hours**.
4. A fix will be developed and released as soon as possible, typically within **7 days**.

## Security Design

- All credentials are injected at runtime via environment variables (secrets manager + Modal Secrets).
- No secrets are hardcoded or committed to the repository.
- Worker endpoints are protected by bearer token authentication (`WORKER_API_KEY`) using constant-time comparison (`hmac.compare_digest`).
- CI/CD workflows use pinned action versions and hardened runners.
