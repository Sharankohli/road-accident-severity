# AI use disclosure

All four members used AI tools during this project. This file lists them; the important prompts are in `PROMPT_LOG.md` and the AI mistakes we caught are in `AI_AUDIT.md`.

| Member | Tool(s) | Used for | Governance entries |
|---|---|---|---|
| Sharan Raj JANAKIRAMAN | Claude (Claude Code in the Claude desktop app), with the project instruction file `CLAUDE.md` | Understanding the brief and labs, planning and task split, explanations of methods, code for M1 and M3, reviewing teammates' pull requests, first drafts of the weekly reports and teammate guides | P1, A1–A4, A12 |
| Xinyu ZHU | GitHub Copilot in VS Code | Code for the M2 split, features, preprocessing and feature selection | P2, A5–A6 |
| SONG Haoyu | OpenAI Codex | Code and notebooks for the M4 ensembles, bootstrap, CatBoost and final evaluation | P3, A7–A9 |
| FOTSING TANKOU DAN KATEL | Claude (Claude Code in VS Code) | Code and notebook for the independent CatBoost check and its interpretation | P4, A10–A11 |

## How we controlled AI output
- Every AI suggestion was checked against the official ONISR data dictionary, the course material and our own measurements before it was kept.
- Every change reached `main` through a pull request reviewed by another member (Sharan reviewed and merged).
- Leakage rules were checked in every review: preprocessing and selection inside the pipeline, folds grouped by accident, test years 2024–2025 used once.
- Each member can explain and defend their own module without the AI.
