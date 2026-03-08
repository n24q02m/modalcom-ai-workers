# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest  | Yes       |
| < latest | No       |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public issue
2. Email **n24q02m@gmail.com** with details
3. Include steps to reproduce if possible
4. Allow reasonable time for a fix before disclosure

## Security Measures

- Dependencies are regularly updated via Renovate
- All code changes require review before merging
- CodeQL analysis runs on every push
- No secrets are hardcoded in the repository
- Modal workers use per-app API keys (`<APP>_WORKER_API_KEY`) for endpoint authentication, providing credential isolation per consumer app
- Key comparison uses `hmac.compare_digest` to prevent timing attacks
- LiteLLM virtual keys provide additional per-app credential isolation at the proxy layer
