# Contributing to bartorch

The [developer guide](https://mcencini.github.io/bartorch/latest/guides/developer/index.html)
documents the toolchain, the editable installation, the repository layout, the
coding and documentation conventions, the pre-commit hooks and the pull-request
procedure.  A development setup is:

```bash
git clone --recurse-submodules https://github.com/YOUR-USERNAME/bartorch.git
cd bartorch
CC=clang CXX=clang++ python -m pip install -e '.[dev]'
pre-commit install
```

Before opening a pull request:

```bash
./scripts/lint.sh
./scripts/run_tests.sh
./scripts/build_docs.sh          # for documentation changes
```

Open the pull request against
[`mcencini/bartorch:main`](https://github.com/mcencini/bartorch/compare).
Participation is governed by the [code of conduct](CODE_OF_CONDUCT.md);
security vulnerabilities are reported as described in [SECURITY.md](SECURITY.md).
