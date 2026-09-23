# Pre-commit hooks

`.pre-commit-config.yaml` runs, on the files a commit changes:

| Hook | Check | CI counterpart |
| --- | --- | --- |
| `nbstripout` | Removes outputs from Jupyter notebooks, except the archived ones under `attic/` | None |
| `codespell` | Spelling, configured by `[tool.codespell]` in `pyproject.toml` | The Typos workflow |
| `ruff` | Lint rules, applying automatic fixes, on `src/`, `tests/` and `docs/examples/` | The Lint workflow (`./scripts/lint.sh`) |
| `ruff-format` | Formatting, on the same paths | The Lint workflow |

Install the hooks once per clone, and run them over the whole repository to
check it independently of a commit:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

A hook that rewrites a file fails the commit; stage the rewritten file and
commit again.  A deliberate exception is made for one commit, or for one hook
by its id:

```bash
git commit --no-verify
SKIP=codespell git commit
```

Bypassing a local hook does not bypass CI: the Lint and Typos workflows run
the same checks on every pull request.  The hook versions are pinned in the
configuration and CI installs the latest releases of ruff and codespell, so a
difference between the two is resolved by updating the pinned `rev`.
