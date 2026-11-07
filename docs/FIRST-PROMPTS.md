# Prompts to paste in OpenCode GUI

## Session 1: Orchestrator (select a verified GPT model in GUI)
Analyze this Git repository and the requirements I provide. Delegate architecture to @architect. Do not change code. First return a 1-page implementation plan, acceptance criteria, API contracts, independent frontend/backend/QA tasks, and exactly which worktree/branch should own each task. Flag all risks and unknowns. Wait for my explicit approval before any implementation.

## Backend worktree: Developer (select cloud model)
You own only the BACKEND task from the approved plan. Confirm the current Git branch and working directory before changes. Implement only the agreed API contract, add unit/integration tests where practical, run available checks, and report actual results. Do not push or merge.

## Frontend worktree: Developer (select Gemini)
You own only the FRONTEND task from the approved plan. Confirm current branch and directory. Implement against the agreed API contract without silently changing the API. Add component tests when available and report actual checks. Do not push or merge.

## QA worktree: QA-local (select ollama/qwen2.5-coder:3b)
Inspect ONE small component and its specified acceptance criteria. Propose or create only focused unit tests. Avoid large context, repo-wide refactors or changing production code. Report any failure; do not claim success without running tests.

## Reviewer (via Orchestrator)
Use @reviewer to inspect the visible diff and acceptance criteria. Prioritize bugs/security/regressions. If another worktree's code is not visible, say so and ask for its diff or PR. Do not approve based only on the author's summary.

## Final integration (human)
List completed task branches, CI status, unresolved issues, migration or deployment risks, and proposed merge order. Do not merge or push without my explicit approval, even if checks pass.
