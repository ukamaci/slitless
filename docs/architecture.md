# Architecture Plan — Context-Aware Refactoring

> **Status: Proposed.** This document describes a target architecture to work
> toward. Nothing described here has been implemented yet. Use it as a
> reference when planning structural changes to the codebase.

## Purpose

A well-factored module structure enables reliable, machine-parsable mapping
between the mathematical models in `python/scripts/apj25.tex` and their
software implementations. This document diagnoses the current architecture,
presents a target design, and explains how the target solves the context
management problems identified in `AGENTS.md`.

## Relationship to Other Context Files

| File | Role | When Loaded |
|------|------|-------------|
| `AGENTS.md` | Operational commands, conventions, gotchas | Always |
| `docs/architecture.md` | This file — structural context | On task match |
| `docs/dead-code-inventory.md` | List of superseded scripts | Pre-cleanup |
| `.agents/context/` | Generated context artifacts (future) | On task match |

**When to update this file:** After any module split, solver addition, or paper
revision that changes the relationship between equations and code.

---

## 1. Current Architecture Diagnosis

| Problem | Context Impact |
|---------|---------------|
| `recon.py` is one 1,448-line file with 7 solver algorithms | No natural boundary for `@paper:` annotations — an annotation on `smart` lives 500 lines from one on `scipy_solver` |
| `forward.py` has 7 `forward_op_tomo_3d*` variants, 4 of them dead | Dead functions could accidentally get `@paper:` tags, polluting the domain map |
| Physical constants defined in 5+ places with slightly different values | Generated context can reference the wrong constant |
| `comparison_test_multi` (187 lines) mixes I/O, solver dispatch, evaluation, and file writing in one function | No clean surface to annotate — is it "the experiment function" or "the evaluation function"? |
| 34 dead script files (52% of `scripts/`) | Clutter obscures which scripts are canonical entry points |
| Zero tests | No executable verification that `forward_op` actually computes Eq. 1 |

---

## 2. Target Architecture

```
python/slitless/
├── __init__.py
├── constants.py                # Single home for SPEED_OF_LIGHT, WAVELENGTH, DISPERSION_SCALE
├── config.py                   # Dataclasses: ImagerConfig, MARTConfig, ScipySolverConfig, UNetConfig
├── forward/
│   ├── __init__.py
│   ├── gauss.py                # gauss, gauss_pix, gauss_torch, gauss_pix_torch
│   ├── projection.py           # tomomtx_gen, forward_op_tomo_3d, forward_op_tomo_3d_transpose
│   ├── parametric.py           # datacube_generator, forward_op (H∘Ψ), forward_op_torch
│   ├── noise.py                # add_noise
│   └── instrument.py           # Source, Imager classes
├── solvers/
│   ├── __init__.py
│   ├── mart.py                 # smart, smart2, gauss_pmf_fitter, smart_fit_spectra_joblib
│   ├── scipy1d.py              # scipy_solver, scipy_solver_parallel, scipy_solver_parallel2
│   ├── grad_descent.py         # grad_descent_solver
│   ├── nn.py                   # nn_solver
│   ├── tomoinv.py              # tomoinv
│   ├── prior.py                # prior_solver
│   └── diffusion.py            # diffusion_solver
├── learning/
│   ├── __init__.py
│   ├── dataset.py              # BasicDataset, OntheflyDataset, transforms
│   ├── unet.py                 # UNet, UNet_fixed, unet_parts
│   ├── training.py             # train_net
│   └── inference.py            # predict, net_loader
├── metrics/
│   ├── __init__.py
│   └── metrics.py              # compare_ssim, nrmse, nmse_torch, tv_loss, cycle_loss
├── eis/
│   ├── __init__.py
│   ├── fitting.py              # fit_spectra_joblib, _worker_fit_chunk
│   ├── interpolation.py        # eis_to_ssi_interpolator
│   └── download.py             # download_eis
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py             # Recon, Reconstructor, comparison_test_multi
└── plotting/
    ├── __init__.py
    └── plotting.py             # barplot_group, plot_recons, plot_val_stats

python/scripts/
├── dataset/
│   ├── generate_dataset.py     # was: generate_dset_v5.py + dataset_generation.py
│   ├── generate_testbed_set.py # unchanged
│   ├── eis_reader.py           # was: eis_reader_v3.py
│   └── full_fov_generator.py   # was: eis_full_fov_generator.py
├── experiments/
│   ├── comparison_testbed.py   # was: comparison_testbed_multi.py
│   ├── auto_param_searcher.py  # unchanged
│   ├── final_result_runner.py  # unchanged
│   └── optimal_init_search.py  # unchanged
├── analysis/
│   ├── datacube_fitting_comparison.py
│   ├── plot_fov_spectra.py
│   └── forward_1d.py           # was: forward_1d_exp.py
├── figures/
│   └── apj_plotter.py          # was: apj_plotter2.py
└── tests/
    ├── test_forward.py         # Eq. 1, Eq. 7, Eq. 2-3 verification
    ├── test_solvers_mart.py    # MART initialization, correction factors, flux conservation
    ├── test_solvers_scipy.py   # 1D MAP objective, gradient consistency
    └── test_eis_fitting.py     # Gaussian fit convergence, parameter bounds
```

---

## 3. How This Structure Solves the Context Problems

### 3.1 `@paper:` annotations become scoped and discoverable

Each solver module has one concern. The `@paper:` tag lives in the module docstring:

```python
# solvers/mart.py
"""MART reconstruction: Multiplicative Algebraic Reconstruction Technique.

@paper: Section 4.3 (MART algorithm, steps 1-8)
@role: iterative-tomographic-solver
@config: MARTConfig
"""

def smart(meas, imager, config: MARTConfig):
    """Step 1-8 of the MART algorithm as described in apj25.tex Section 4.3."""
    ...

def smart2(meas, imager, config: MARTConfig):
    """Extended MART with multi-Gaussian + background spectral model."""
    ...
```

The generated domain map now has natural section headers matching the module
boundaries. An agent reading `solvers/mart.py` immediately knows it's about MART.

### 3.2 Configuration objects replace parameter lists and document semantics

```python
# config.py
@dataclass
class MARTConfig:
    """Configuration for the MART/SMART2 iterative solver.

    @paper: Section 4.3, MART algorithm
    """
    # Outer loop
    max_outer: int = 10              # N_outer in paper
    max_inner: int = 5               # N_inner in paper

    # Contrast enhancement (Step 2)
    psi: float = 0.2                 # alpha in paper, contrast exponent

    # Spectral model (Step 1 initialization)
    frac_primary: float = 0.7        # fraction of flux in primary Gaussian
    frac_secondary: float = 0.2      # fraction of flux in secondary blend
    frac_background: float = 0.1     # fraction of flux in background
    center_primary: float = 0.0      # initial velocity of primary [pix]
    width_primary: float = 1.38      # initial line width of primary [pix]

    # Prior (EIS average spectrum constraint)
    prior_weight: float = 1.0        # weight of the 90-degree projection prior
    use_inf_prior: bool = True       # enable the EIS-average line profile prior
```

This solves the "19-parameter function" problem AND serves as self-documenting
context. An agent reading `MARTConfig` sees the paper mapping, parameter
semantics, and defaults all in one place. The config object IS the documentation.

### 3.3 Tests become executable domain contracts

```python
# tests/test_forward.py
def test_forward_op_matches_continuous_integral():
    """Verify Eq. 1 (eq:fwd_cont): forward_op = H∘Psi matches analytical integral."""
    ...

def test_datacube_generator_matches_parametric_voxel():
    """Verify Eq. 7 (eq:param_voxel): r_ijk = Gaussian + b + w."""
    ...

def test_forward_op_tomo_3d_matches_tomomtx_gen():
    """Verify Eq. 2-3: tomo forward matches matrix form H @ r."""
    ...

def test_add_noise_preserves_snr():
    """Verify Poisson noise model: SNR = sqrt(N_avg)."""
    ...

# tests/test_solvers_mart.py
def test_mart_initialization():
    """Verify Step 1: v = Psi([y^(0), 0, avg_width])."""
    ...

def test_mart_correction_factors():
    """Verify Step 5: c^(a) = [y^(a)/~y^(a)]^(2/K)."""
    ...

def test_mart_conserves_flux():
    """Verify multiplicative update preserves total flux within tolerance."""
    ...
```

These tests are the semantic integrity guarantee that pure annotation
verification cannot provide. They catch: someone changing the einsum in
`forward_op` and breaking Eq. 1; someone changing the correction factor exponent
from `2/K` to `1/K` in MART; someone changing the SNR model from Poisson to
Gaussian; someone changing the erf bounds in `gauss_pix`.

Each test comment includes the paper equation label — creating a bidirectional,
executable link between domain knowledge and implementation.

### 3.4 Dead code removal eliminates false annotation targets

With 34 dead scripts and 7 dead tomo variants removed:
- `gen_context.py` scans fewer files, produces smaller context
- No risk of annotating `forward_op_tomo_3d_old` instead of `forward_op_tomo_3d`
- Script `scripts/` directory has 13 alive files instead of 47 — any script is
  now automatically a "canonical" version

### 3.5 `scripts/` becomes the integration layer, not a dumping ground

Each script has one clear purpose. The directory structure IS the documentation
of what exists. An agent can navigate `scripts/experiments/` to find all
comparison testbeds, `scripts/dataset/` to find all dataset builders, and
`scripts/tests/` to find all test suites.

---

## 4. Refactoring Robustness

### Risk Mitigation

| Step | Risk | Mitigation |
|------|------|------------|
| Split a function into a new file | Import chain breaks | Automated import updater (`sed`/`autoflake` pattern) + smoke test after each move |
| Extract config from 19-parameter function | Default values change | Diff the defaults before/after. Extract defaults from current call sites via AST, not manual copy |
| Consolidate constants | Values conflict | Run existing scripts with both old and new constants, compare outputs. Flag any divergence |
| Remove dead code | Remove something still used | `git grep` for every call site of every function before deletion |
| Rename modules | Import statements break | `git mv` preserves history. Run existing alive scripts as smoke tests |
| Add tests | Tests misrepresent domain math | Validate test assertions against paper equations. Have domain expert review |

### Golden-Output Test Strategy

For every refactoring step, run the alive scripts
(`comparison_testbed_multi.py`, `final_result_runner.py`) and assert the output
is bitwise-identical — or within floating-point tolerance for
order-of-operations changes. If the tests don't exist yet to cover this, the
refactoring itself writes them as the first step.

**Critical fragility:** `forward.py` has `forward_op` (the H∘Psi composite) and
`forward_op_tomo_3d` (the H-only projection). These are called by 6+ different
solvers. A mistake in splitting these into separate files could silently break
MART, tomoinv, and U-Net training simultaneously. The mitigation is a golden
output test: run the full pipeline on one canonical input before and after the
split, and assert bitwise-identical output.

---

## 5. Effort/Benefit Analysis

| Task | Effort (hours) | Benefit |
|------|---------------|---------|
| Delete 34 dead scripts | 0.5 | Clean workspace, clear what's canonical |
| Remove dead functions from `forward.py` (7 tomo variants) | 1.0 | Clean domain surface for annotations |
| Consolidate constants into `constants.py` | 1.0 | Single source of truth, eliminates 5-way duplication |
| Split `forward.py` into 4 submodules | 2.0 | Natural boundaries for `@paper:` annotations |
| Split `recon.py` into 7 solver modules | 3.0 | One annotation per domain concern |
| Extract config dataclasses | 2.0 | Self-documenting parameters, machine-parseable semantics |
| Split `data_loader.py`/`eistools.py`/`evaluate.py` | 2.0 | Clean imports, no side effects |
| Organize `scripts/` into subdirectories | 1.0 | Canonical entry points are obvious |
| Write golden-output smoke tests | 3.0 | Catch semantic drift during refactoring AND ongoing development |
| Write domain-equation unit tests | 8.0 | Executable domain knowledge, catches semantic drift permanently |
| Write `gen_context.py` (generation + verification) | 3.0 | Auto-generated context that stays fresh |
| Annotate with `@paper:` tags | 1.0 | Links code to equations |
| **Total** | **27.5** | — |

**Convexity argument:** The first 12 hours (dead code removal, splits, constants,
configs) deliver approximately 60% of the structural improvement with near-zero
risk. The next 15 hours (tests, generation script, annotations) deliver the
self-healing context management system. The curve is front-loaded: early work
pays disproportionately.

---

## 6. What Refactoring Does NOT Solve

- **Paper prose changes** (MART algorithm description rewritten without new
  labels) still require human detection
- **New domain concepts** (a new solver algorithm, a new instrument model) still
  require someone to add `@paper:` tags and tests
- **Library dependency changes** (eispac API change, PyTorch deprecation) are
  orthogonal to code structure
- **The verification script itself** needs maintenance when LaTeX format or AST
  structure changes

These require the annotation + generation + test system described in this
document. Refactoring makes that system easier to build and more reliable, but
it doesn't replace it.
