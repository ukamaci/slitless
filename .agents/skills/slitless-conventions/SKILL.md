---
name: slitless-conventions
description: Enforce slitless codebase conventions and prevent adding anti-patterns. Activate when editing or reviewing Python source in python/slitless/ or python/scripts/, or before committing changes. Covers import style, path handling, commit format, docstrings, 3-channel data format, and guardrails against introducing absent patterns like type hints, async, poetry, or relative imports.
license: GPL-3.0
compatibility: Requires Python 3 with numpy, scipy, torch, matplotlib, scikit-image, eispac, joblib, tqdm, seaborn. Install via `cd python && pip install -e .`
---

## Project conventions (preserve these)

### Imports
- Intra-package: `from slitless.forward import forward_op, Imager` (NOT relative `from .forward`)
- External: comma-separated single-line for small groups is the existing style. Match the surrounding file.

### Constants
- Physical constants in `UPPER_CASE` at module level: `SPEED_OF_LIGHT`, `WAVELENGTH`, `DISPERSION_SCALE`

### Docstrings and headers
- New-module header: `# YYYY-MM-DD` followed by `# Ulas Kamaci`
- Public functions: include a one-line docstring summary. NumPy-style Args/Returns for multi-argument functions, following examples in `measure.py` and `eistools.py`.

### Commit format
- `type(scope): message` — `fix(forward):`, `refactor(recon):`, `feat(scripts):`, `chore:`, `docs:`
- One concern per commit. Never commit `.pyc`, `.pth`, `.h5`, `.npy`, `.npz`, `.pickle`, images, or `egg-info/`.

### Parallelism
- Use `from joblib import Parallel, delayed`

### Device management
- `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`

### Data format
- Parameters are 3-channel arrays: `[intensity, velocity/Doppler, linewidth]`
- Measurements are multi-detector: first axis = spectral orders [0, -1, 1]

## Path handling (critical — do NOT introduce new hardcoded paths)
- NEVER write `/home/kamo/` or any absolute home-directory prefix
- Accept paths as function parameters with defaults derived from `os.path.dirname(slitless.__file__)`
- Use `os.path.join(data_dir, subdir, file)` — no string concatenation for paths

## Do NOT introduce (patterns absent from codebase)
- **Type hints** — don't add any unless converting the entire project
- **Relative imports** — always `from slitless.X`
- **Async/await** — not used in this scientific codebase
- **pyproject.toml, poetry, flit** — `setup.py` is the only packaging mechanism
- **New dependencies** — add them only to `install_requires` in `python/setup.py`
- **Abstract classes or decorator frameworks**
- **CI/CD or pre-commit hooks** — don't add

## Anti-patterns to avoid (these exist — don't replicate)
- **Module-level file I/O** — avoid `np.load()` at import time. Use lazy loading or accept paths as parameters.
- **No input validation** — new functions that accept arrays should check shapes and reject NaN with a `ValueError`.
- **Growing recon.py** — already 1448 lines. Add new solvers in `python/slitless/` as separate modules.
- **Empty `__init__.py`** — keep them empty (consistent with existing files).
