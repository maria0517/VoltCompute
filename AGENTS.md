# Project engineering rules

- Use Romanian for status updates; keep code, variable names and commits in the repository's existing conventions.
- Before any implementation, clarify the task's observable acceptance criteria, impacted components and dependencies.
- Never invent test results, command output, or file locations.
- One task branch and worktree per parallel implementation session. Confirm the branch/path before changes.
- Never run push, force push, merge, rebase, reset or clean automatically. Integrate only after human review.
- Never access, print, commit or share secrets, tokens, production databases or `.env` contents.
- Minimize changes: no unrelated refactors, dependency upgrades or architecture changes without approval.
- Backend/frontend agents must agree on API contracts BEFORE parallel implementation.
- Include tests for changed behavior; run project-defined lint, typecheck and tests when available.
- For each deliverable report: files changed, rationale, commands actually run and outcomes, risks, next steps.
- Code review must be independent of the author; reject claims of completion without objective checks.
- Treat README and project docs as sources of project-specific build/test commands; never assume framework.
- Do not deploy to production, change access controls or run migrations against live systems without explicit human approval.
