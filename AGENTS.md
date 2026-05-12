# Slitless Spectral Imaging

Simulation and image reconstruction for slitless spectral imaging (GPL-3.0).

## Commands

- **Setup**: `cd python && pip install -e .`
- **Tests**: none
- **Lint**: none
- **Typecheck**: none

## Architecture

| Module | Lines | Purpose |
|--------|-------|---------|
| `forward.py` | 896 | Forward imaging model, noise, Source/Imager classes |
| `recon.py` | 1448 | Solver functions: smart, scipy, nn, prior, diffusion, tomoinv, smart2, scipy_solver_parallel2 |
| `eistools.py` | 605 | EIS spectral fitting, velocity calc, data conversion |
| `data_loader.py` | 204 | Dataset classes, normalization transforms |
| `train.py` | 455 | U-Net training loop |
| `measure.py` | 156 | Loss functions, SSIM, NRMSE, cycle loss |
| `evaluate.py` | 520 | Model loading, prediction, plotting |
| `plotting.py` | 175 | Bar plots, image generation |
| `common.py` | 25 | Output channel adjustment |
| `networks/unet.py` | 86 | U-Net architecture |
| `scripts/` | — | Experiment scripts, plotters, dataset generators |

## Conventions

- Commit format: `type(scope): message` — `fix(forward):`, `refactor(recon):`, `feat(scripts):`, `chore:`, `docs:`
- Imports: `from slitless.forward import forward_op` (NOT relative)
- Data format: 3-channel `[intensity, velocity, linewidth]`, measurements = spectral orders [0, -1, 1]
- Parallelism: `from joblib import Parallel, delayed`

## Gotchas

1. 102 hardcoded `/home/kamo/` paths exist — never add more. Use `os.path.dirname(slitless.__file__)`.
2. `data_loader.py` calls `np.load()` at import time — blocks import if file missing. Don't replicate this pattern.
3. `recon.py` is monolithic (1448 lines). Add new solvers as separate modules, not inside it.

## Agent Skills

Load these when tasks match their purpose:

- `.agents/skills/slitless-conventions/SKILL.md` — Full convention list, import style, anti-pattern guardrails
- `.agents/skills/find-docs/SKILL.md` — Fetch current docs before using library APIs (especially eispac, torch)
