# Security Policy

## Supported Versions

Only the latest release is supported at this time.

## Reporting a Vulnerability

Please report security issues privately by emailing the maintainer. Do not open
public GitHub issues for security problems.

This release fixes a remote code execution vulnerability in the `Dimension`
formula evaluation by replacing `eval()` with a safe parser.
