---
name: find-docs
description: Fetch current documentation for libraries used in the slitless project — numpy, scipy, torch, matplotlib, scikit-image, eispac, joblib, tqdm, seaborn, denoising-diffusion-pytorch, PIL, opencv. Activate when writing or reviewing code that uses these libraries, especially when uncertain about API signatures, when the library is rapidly changing (torch), or when the library is niche with zero training data (eispac). Never rely on training data for API details — always verify.
compatibility: Requires Python 3, ctx7 CLI (npm install -g ctx7@latest), and web access.
---

## Principle
Never rely on training data for library API details, signatures, or configuration options. Training data is frequently outdated or missing for niche libraries. Always fetch current documentation before using unfamiliar APIs.

## How to fetch docs

### Option A: ctx7 (preferred for mainstream libraries)
Two-step workflow:
```bash
ctx7 library <name> "<query>"          # Step 1: resolve library ID
ctx7 docs <libraryId> "<query>"        # Step 2: fetch documentation
```
Library IDs require a `/` prefix (e.g., `/numpy/numpy`, `/pytorch/pytorch`). Always run `ctx7 library` first. Do not exceed 3 attempts per question.

Works without authentication. For higher rate limits: `ctx7 login` or `export CONTEXT7_API_KEY=<key>`.

### Option B: Web fetch (fallback, always available)
Use the agent's built-in web fetch capability to retrieve documentation from the URLs listed below. Form URLs using the table. For specific pages, append the relevant path.

## Explicit documentation URLs

| Library | Documentation root |
|---------|-------------------|
| Python 3 stdlib | https://docs.python.org/3/library/ |
| numpy | https://numpy.org/doc/stable/reference/ |
| scipy | https://docs.scipy.org/doc/scipy/reference/ |
| torch (PyTorch) | https://pytorch.org/docs/stable/ |
| matplotlib | https://matplotlib.org/stable/api/ |
| scikit-image | https://scikit-image.org/docs/stable/api/ |
| eispac | https://eispac.readthedocs.io/en/latest/ |
| joblib | https://joblib.readthedocs.io/en/latest/ |
| tqdm | https://tqdm.github.io/ |
| seaborn | https://seaborn.pydata.org/api.html |
| PIL (Pillow) | https://pillow.readthedocs.io/en/stable/reference/ |
| opencv (cv2) | https://docs.opencv.org/4.x/ |
| denoising-diffusion-pytorch | https://github.com/lucidrains/denoising-diffusion-pytorch |
| mas | No public docs — read the local source at the import location |

## When to fetch docs
- BEFORE using any eispac, torch, denoising-diffusion-pytorch, or opencv API
- When writing code with an unfamiliar function or class from any library above
- When encountering an error that suggests an API mismatch
- When the user asks "how do I..." about any of these libraries
- When the code being written or reviewed uses imports from these libraries

## Library-specific notes

### eispac (CRITICAL)
Agents have zero training data on eispac. ALWAYS fetch docs before writing any eispac code. Key modules used in this project:
- `eispac.core.eiscube.EISCube` — EIS data cube container
- `eispac.core.fitting_functions` — spectral fitting functions
- `eispac.instr.calc_velocity` — velocity calculation from wavelength shift
- `eispac.core.scale_guess` — initial guess scaling for fit parameters
- `eispac.extern.mpfit.mpfit` — MPFIT least-squares fitting engine
User guide: https://eispac.readthedocs.io/en/latest/guide/

### torch (PyTorch)
PyTorch API changes frequently. Always verify with current docs. Key patterns in this project:
- `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- `torch.nn` modules (UNet, Conv2d, BatchNorm2d, ReLU)
- `torch.optim` for training
- `torch.utils.data.Dataset` and `DataLoader`

### denoising-diffusion-pytorch
Small third-party library, only README docs available. Key classes used:
- `denoising_diffusion_pytorch.Unet`
- `denoising_diffusion_pytorch.GaussianDiffusion`
