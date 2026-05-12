# AI Agent Configurations — Full Inventory

## 1. `AGENTS.md` (always-loaded instructions)

**File:** `AGENTS.md` (46 lines, repo root)

**What it does:** Provides project identity, architecture table, commands, key conventions, and gotchas to every AI coding agent session automatically. Supported natively by OpenCode, Gemini CLI, and GitHub Copilot (since Aug 2025).

**How to use:** No action needed — loaded automatically by all three tools on every session in this repo.

**Benefit:** Keeps the agent aligned with project structure without the user repeating themselves. The architecture table tells the agent which module does what (e.g., "recon.py has solvers, forward.py has the imaging model"). The gotchas prevent the agent from replicating known problems (hardcoded paths, import-time I/O, growing recon.py).

**How to extend:** Add new sections (new gotchas, new modules in the architecture table, new conventions). Keep it under ~60 lines — put detailed instructions in skills instead.

**How to disable:** Delete the file, or rename it (e.g., `mv AGENTS.md _AGENTS.md.disabled`).

## 2. `slitless-conventions` skill (on-demand coding rules)

**File:** `.agents/skills/slitless-conventions/SKILL.md` (53 lines)

**What it does:** Activated on-demand when editing/reviewing Python source. Enforces: `from slitless.X` imports (never relative), no hardcoded paths, no type hints, no async, commit format, 3-channel data format, no growing recon.py.

**How to use:** The agent loads it when the task matches its description — editing `python/slitless/` or `python/scripts/`, or before committing. You can also explicitly ask the agent to "load the slitless-conventions skill".

**Benefit:** Prevents the agent from introducing patterns that don't exist in the codebase (type hints, async, poetry, relative imports) and guards against replicating existing anti-patterns (hardcoded paths, import-time `np.load()`, growing recon.py).

**How to extend:** Add new conventions or anti-patterns to the body. Follow the YAML frontmatter format (name, description, license, compatibility).

**How to disable:** `rm -r .agents/skills/slitless-conventions/`

## 3. `find-docs` skill (documentation fetching)

**File:** `.agents/skills/find-docs/SKILL.md` (72 lines)

**What it does:** Instructs the agent to fetch current library documentation before writing code. Two methods: (A) `ctx7 library` + `ctx7 docs` for mainstream libraries, (B) direct web fetch to explicit URLs for niche/unsupported libraries (eispac, denoising-diffusion-pytorch, mas).

**How to use:** Activated when the agent writes code using numpy, scipy, torch, matplotlib, scikit-image, eispac, joblib, tqdm, seaborn, PIL, or opencv. `ctx7` CLI must be installed (`npm install -g ctx7@latest`).

**Benefit:** Agents have zero training data on eispac and outdated data on fast-moving libraries like torch. This skill forces the agent to verify API signatures against live docs.

**How to extend:** Add new library URLs to the table, or add library-specific notes following the eispac/torch patterns.

**How to disable:** `rm -r .agents/skills/find-docs/`

Alternatively, disable per-skill but keep the file: remove `.agents/skills/find-docs/SKILL.md` references from `AGENTS.md` line 46.

## 4. `.editorconfig` (editor normalization)

**File:** `.editorconfig` (17 lines, repo root)

**What it does:** Ensures consistent LF line endings, UTF-8 charset, trailing whitespace trimming, and 4-space indentation for Python across all editors that support EditorConfig (VS Code, IntelliJ, etc.).

**Benefit:** Prevents mixed line endings and tabs-vs-spaces noise in diffs.

**How to disable:** Delete the file.

## 5. `.gitattributes` (git normalization)

**File:** `.gitattributes` (25 lines, repo root)

**What it does:** Forces LF line endings for all text files (`*.py`, `*.md`, `*.tex`, etc.) and marks binary formats as binary (`*.npy`, `*.png`, `*.pth`, etc.) so git won't try to diff them or mangle line endings.

**Benefit:** Prevents CRLF corruption on Windows checkouts and avoids noisy binary diffs.

**How to disable:** Delete the file, then `git rm --cached -r . && git add -A && git commit -m "..."` to re-normalize.

## 6. `.gitignore` (artifact exclusion + credential safeguard)

**File:** `.gitignore` (28 lines, repo root)

**What it does:** Excludes Python artifacts (`__pycache__`, `*.pyc`, `egg-info/`), data files (`*.npy`, `*.h5`, `*.pth`), figures (`*.png`, `*.jpg`, `*.pdf`), and credential files (`.env`, `.env.*`, `.envrc`). The `.env` entries are the credential leak guardrail — if someone puts `CONTEXT7_API_KEY=...` in a repo-level env file, git ignores it.

**How to disable:** Delete lines 26–28 to remove `.env` safeguards.

## 7. `ctx7` CLI + API key (documentation tooling)

**Where it lives:** Global npm install (`~/.nvm/versions/node/v24.14.0/lib/node_modules/ctx7/`). API key stored outside repo (OS keyring or in-memory env var from `ctx7 setup --cli --api-key <key>`).

**What it does:** Powers the `find-docs` skill. Resolves library names to context7 IDs and fetches current API documentation.

**How to use:** `ctx7 library numpy "how to stack arrays"` → `ctx7 docs /numpy/numpy "np.stack"`.

**Benefit:** Agents get current docs instead of hallucinating outdated API signatures.

**How to disable:** `ctx7 remove --opencode --gemini` to undo `setup`, or `npm uninstall -g ctx7` to remove entirely.

## Relationship diagram

```
┌─────────────────────────────────────────────┐
│ AGENTS.md (always loaded)                    │
│   "Load skills when tasks match..."          │
│   ├─ slitless-conventions ◄── edit .py       │
│   └─ find-docs ◄── uses external lib         │
│         └─ ctx7 CLI (API key outside repo)   │
└─────────────────────────────────────────────┘
Supporting: .editorconfig, .gitattributes, .gitignore
```

## Quick disable everything

```bash
mv AGENTS.md _AGENTS.md.disabled
rm -r .agents/
rm .editorconfig .gitattributes
# .gitignore: remove lines 26-28 only if you want .env files tracked
ctx7 remove --opencode --gemini   # optional: undo ctx7 agent setup
```
