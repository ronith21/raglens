# Instructions for coding agents (Jules and others)

RAGLens is a portfolio-grade Retrieval-Augmented Generation system with a built-in evaluation
harness. Python 3.11. The quality bar is production-style code a senior AI engineer would approve.
The full design is in [SPEC.md](SPEC.md). Read it before every task and keep it accurate.

## Non-negotiable rules

1. **Never fabricate results.** Every number, table, chart or screenshot in the repo must come from a
   command actually run in that session. Save raw outputs under `docs/proof/` (for example
   `docs/proof/pytest.txt`, `docs/proof/coverage.txt`, `docs/proof/eval_report.md`).
2. **No faking when something can't run** (no GPU, network blocked, model download fails). Use the
   documented fallback and state the limitation in the PR description and the README. Never commit
   placeholder images.
3. **Tests never need the network or a real LLM.** Use `FakeLLM` and `HashEmbedder` (see SPEC.md).
   Real models are optional extras, lazily loaded.
4. **Before finishing any task**, run all of these and make them pass:
   `ruff check .`, `ruff format --check .`, `mypy src`, `pytest --cov=raglens`.
   Fix every error and warning. Paste the final output of each in the PR description.
5. Type hints everywhere, docstrings on public functions, no dead code, no leftover TODOs,
   no secrets committed. Configuration through pydantic-settings with a `.env.example`.
6. **Git workflow:** always start from the latest `main`. Work on a branch named `jules/<task-slug>`,
   use clear conventional commit messages (`feat:`, `fix:`, `test:`, `docs:`, `ci:`), and open a
   pull request into `main` whose description lists what changed, how it was verified, and any
   limitations.
7. Keep CI green. If you change dependencies, update `pyproject.toml` and the lock/requirements file.
8. Do not delete or rewrite work from earlier tasks unless it is a bug fix; explain any such change
   in the PR.
