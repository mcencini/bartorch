# Security policy

## Reporting a vulnerability

Report a vulnerability privately through GitHub's private vulnerability
reporting: the **Report a vulnerability** button under the repository's
[Security](https://github.com/mcencini/bartorch/security) tab.  Do not open a
public issue for it.  If the button is not available, open an issue asking the
maintainers for a private contact, without describing the vulnerability.

A report states the affected version or commit, the operating system,
architecture and device, the smallest input that reproduces the behaviour, its
observed impact, and any known mitigation.  The maintainers acknowledge a
report within seven days and communicate their assessment and the remediation
through the private advisory.  Disclosure follows the release of a fix or an
agreed date, and reporters are credited unless they ask not to be.

Vulnerabilities in the embedded BART, FINUFFT, PyTorch or another dependency
are reported to that project as well; a report here then concerns how
bartorch is affected.

## Supported versions

bartorch is pre-alpha.  Fixes are made on the default branch and released in
the next version; earlier releases are not maintained.

| Version | Supported |
| --- | --- |
| Default branch and latest release | Yes |
| Earlier releases | No |
