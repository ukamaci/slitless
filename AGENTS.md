# Slitless Spectral Imaging

Simulation and image reconstruction for slitless EUV spectral imaging (GPL-3.0).

## Commands

- **Setup**: `cd python && pip install -e .`
- **Tests**: none
- **Lint**: none
- **Typecheck**: none

## Architecture

| Module | Lines | Purpose |
|--------|-------|---------|
| `forward.py` | 896 | Forward imaging model, noise, Source/Imager classes |
| `recon.py` | 1448 | Solver functions and Reconstructor classes |
| `eistools.py` | 605 | EIS spectral fitting, velocity calc, data conversion |
| `data_loader.py` | 204 | Dataset classes, normalization transforms |
| `train.py` | 455 | U-Net training loop |
| `measure.py` | 156 | Loss functions, SSIM, NRMSE |
| `evaluate.py` | 520 | Model loading, prediction, plotting |
| `plotting.py` | 175 | Bar plots, image generation |
| `networks/unet.py` | 86 | U-Net architecture |
| `scripts/` | — | Experiment scripts, plotters, dataset generators |

---

## Dataset

**Instrument**: Hinode/EIS (Extreme Ultraviolet Imaging Spectrometer). Spectral range 170–210 Å and 250–290 Å, dispersion 22 mÅ/pixel, plate scale 1 arcsec/pixel.

**Target line**: Fe XII 195.12 Å. Rest wavelength λ_rest = 195.117937907451 Å.

**Version in use**: `dset_v5` — 64×64 pixel patches, 3 physical parameter channels:
- Channel 0: Intensity [erg cm⁻² s⁻¹ sr⁻¹]
- Channel 1: Doppler velocity [pixels → km/s]
- Channel 2: Line width [pixels → km/s]

**Unit conversion**: 1 pixel = 22 mÅ ≈ 34 km/s at 195.12 Å.
`v_kms = v_px * 0.022275 * 299792.458 / 195.117937907451`

**Spectral model per pixel**: two-Gaussian + flat background
(primary Fe XII + secondary blend + continuum). Ground-truth parameters are from the primary Gaussian only.

**Dataset construction**:
1. Query Hinode SDC for AR observations, 1" slit, FOV ≥ 64×64", capturing 195.12 Å window → 2383 scans.
2. Read with EISPAC, interpolate onto instrument spectral grid.
3. Fit two Gaussians + background per pixel (Levenberg–Marquardt) to extract ground-truth `x`.
4. Simulate measurements `y = H * r` at 5 diffraction orders (0, ±1, ±2).
5. Mask pixels where fit failed, intensity outside [100, 25000], |velocity| > 68.5 km/s, or width < 0.022 Å.
6. Extract 15 independent 64×64 patches per scan.

**Split**: 80/10/10 train/val/test, split by parent raster (no patch-level leakage).
- Train: 23036 patches | Val: 2880 patches | Test: held-out pool, 100 patches used for evaluation.

**Normalization** (applied before U-Net training):
- Intensity and measurements: log z-score (log then subtract mean / divide std, computed on training set).
- Velocity and line width: z-score (zero mean, unit variance).
- `param_inv_transform` in `data_loader.py` reverses this at inference.

**Files** (in `data/datasets/baseline/`):
- `eis_train_5_dsetv5.npy` — 5-image train subset (quick tests, hyperparameter search)
- `eis_test_100_dsetv5.npy` — 100-image official test set
- Loaded as `.item()` dicts with keys: `param3d` (N,3,64,64), `meas`, `meas_damped` (N,K,64,64).

**Outlier**: `idx_im=45` in the test set has peak intensity ~18,375 (vs ~7,000 typical). All final paper results exclude this image (99-image set).

---

## Methods

### U-Net (`nn_solver`)

Supervised convolutional encoder–decoder trained end-to-end to map K measurement images → 3 parameter maps.

**Architecture**: U-Net with skip connections. Input: (K, 64, 64) stacked detector images. Output: (3, 64, 64) parameter maps. Each encoder/decoder block has two sequential 3×3 conv layers + batch norm + ReLU. 2×2 max pooling for downsampling, bilinear interpolation for upsampling. NF=64 base feature channels (doubles each encoder stage). A separate network is trained for each (K, γ) pair.

**Training**: MSE loss, Adam (lr=2e-4), batch size 32, 200 epochs, ~2.5 hours on one NVIDIA TITAN Xp.

**Inference**: single forward pass, <1 second. No per-image optimization.

**Model path** (paper results, K=3, noiseless):
`python/results/models/2026_05_17__18_55_57_NF_64_BS_32_LR_0.0002_EP_200_KSIZE_(3, 3)_MSE_LOSS_ADAM_all_dbsnr_100_None_K_3_eis_v5_logzscale`

---

### 1D MAP (`scipy_solver_parallel2`)

Column-wise Maximum A Posteriori estimation. Inverts each spatial column independently via regularized nonlinear least-squares (L-BFGS-B):

```
ξ̂_i = argmin ||y_i - A_i(ξ_i)||² + R(ξ_i)
```

Each column optimizes 7M parameters: primary Gaussian (f, μ, σ), secondary blend (f₁, μ₁, σ₁), and background (b) per pixel. Only the primary Gaussian is returned as output.

**Regularizer**: per-parameter 1D smoothness (squared differences between adjacent pixels along the column). Weights `lam_i`, `lam_v`, `lam_w` are separately tuned per (K, γ).

**Spectral template** (fixed): `frac1`, `frac2`, `frac_bg` set fractional amplitudes; `cent1`, `cent2`, `wid1`, `wid2` fix the two-Gaussian centroids and widths; `bg_shape_norm` is a 21-element normalized background profile. All derived from EIS training data statistics.

**Solver**: `scipy.optimize.minimize` with `L-BFGS-B` (replaces the Nelder-Mead used by Davila 2019).

**Hyperparameter tuning**: Optuna over `lam_v` and `lam_w` (log-uniform [1e1, 1e10]), minimizing `rmse_vel + rmse_wid` on the 5-image train subset. See `auto_param_searcher.py`.

**Cost**: ~minutes per image; parallelized over columns with joblib.

---

### MART (`smart2_twostage`)

Multiplicative Algebraic Reconstruction Technique for limited-angle tomography. Recovers the full spectral data cube r ∈ ℝ^(P×M²) iteratively, then fits Gaussians to extract line parameters.

**Two-stage procedure** (to decouple velocity and line-width recovery):

*Stage 1*: MART with spectral prior disabled (`w_prior=0`), doubled iteration counts. Velocities evolve freely → produces spatially-varying Doppler map μ̂⁽¹⁾.

*Stage 2*: Constructs velocity-adapted initial cube using μ̂⁽¹⁾, then re-runs MART with `w_prior=1`. The spectral prior (average two-component template from EIS training data, scaled to match zeroth-order flux) constrains line widths and intensities while keeping velocities near the Stage 1 solution.

**MART update** (each inner iteration):
1. Compute projections: ỹ = H·r
2. Correction factors: c⁽ᵃ⁾ = y⁽ᵃ⁾ / ỹ⁽ᵃ⁾
3. Back-project: C⁽ᵃ⁾ = Hᵀ⁽ᵃ⁾ c⁽ᵃ⁾
4. Multiplicative update: r ← r · ∏ (C⁽ᵃ⁾)^(wₐ/W) over non-converged orders

**Outer loop** additionally applies: contrast enhancement r ← (r + r^(1+α)) / norm, and spatial+spectral smoothing with a 3D kernel.

**Post-processing**: spectral fitting (same two-Gaussian + background Levenberg–Marquardt fit as dataset preparation) extracts primary line parameters from the reconstructed cube.

**Key parameters**: `psi` (relaxation/alpha), `maxouter`, `maxinner`, `prior_weight`, `cent1`, `wid1`/`wid2` (line centroids/widths in Å), `frac1`, `frac2`, `frac_bg`.

**Cost**: ~minutes per image; parallelized over pixels with `n_jobs=-1`.

---

## Regularizers

| Parameter | Solver | Meaning |
|-----------|--------|---------|
| `lam_i` | 1D MAP | Weight on intensity 1D smoothness |
| `lam_v` | 1D MAP | Weight on velocity 1D smoothness |
| `lam_w` | 1D MAP | Weight on line-width 1D smoothness |
| `lam_sum` | 1D MAP | Sum-to-one constraint (keep 0) |
| `prior_weight` | MART | Weight of spectral prior projection |
| `psi` / alpha | MART | Contrast enhancement exponent |
| MSE loss | U-Net | Implicit regularization via dataset prior |

1D MAP regularizer (Eq. per column i):
`R(ξ_i) = Σ_j Σ_p  w_p * (ξ_{i,j}^(p) - ξ_{i,j-1}^(p))²`
where p=1..3 → primary params, p=4..6 → secondary blend, p=7 → background.

---

## Evaluation Metrics

**Reported metric**: mean of per-image RMSEs (not global/pooled RMSE).

For image i, parameter p:
```
RMSE_i_p = sqrt( mean_{x,y}( (recon_i_p(x,y) - truth_i_p(x,y))² ) )
```
Reported value: `mean over i of RMSE_i_p` = `Rec.rmse_phy.mean(axis=(0,1))`

`Rec.rmse_phy` shape: `(num_realizations, num_images, 3)`. Same for `Rec.bias_phy`.

**Do not use** global/pooled RMSE (`sqrt(mean of all pixel errors²)`); it inflates the metric due to high-intensity outliers.

**Units**: Intensity (arbitrary), Velocity (km/s), Line Width (km/s).

Physical conversion: `Imager.frompix(..., width_unit='km/s')`. For scatter plots or manual computation:
```python
val_kms = val_px * 0.022275 * 299792.458 / 195.117937907451
```

**Test set**: 99 images (idx_im=45 excluded as outlier). All paper tables report 99-image per-image mean RMSE and Bias.

**Experiment configurations** (K-sweep: noiseless; SNR-sweep: K=3):
- K-sweep: K ∈ {2, 3, 4, 5}, γ=∞ (no added noise)
- SNR-sweep: K=3, γ ∈ {10, 20, 30} dB, plus γ=∞

---

## Workflow

### Quick test
```
python python/scripts/comparison_testbed_multi.py
```
Edit solver block and parameters at top. Prints RMSE/Bias. Does not save unless `save=True`.

### Hyperparameter search (1D MAP)
```
python python/scripts/auto_param_searcher.py
```
Optuna over `lam_v`, `lam_w` for each K in [2,3,4,5], minimizes `rmse_vel + rmse_wid`. Saves `optimal_scipy2_params_k.json`.

### Full evaluation
```
python python/scripts/final_result_runner.py
```
Set `METHOD = 'unet' | 'mart' | '1dmap'`. Iterates 7 configs. Saves `Rec.pickle` + `results_summary.txt` per config under `python/results/recons/<timestamp>_final_runner_<METHOD>/`.

### Plotting
```
python python/scripts/apj_plotter.py
```
- `get_bar_chart_figure()` — reads `figures/apj26_post_revision/bar_data.pickle`
- `get_joint_scatter_figure(config='K_3_dbsnr_None')` — hexbin true-vs-estimated scatter with Pearson r
- `get_recon_comp_figures_final()` — per-image reconstruction comparisons

`bar_data.pickle` must be regenerated manually when results change. Structure: `data['K_sweep'|'Gamma_sweep']['U-Net'|'MART'|'1D MAP']['RMSE'|'Bias'][param_idx][config_idx]`.

---

## Conventions

- Commit format: `type(scope): message` — e.g. `fix(forward):`, `feat(scripts):`, `docs:`
- Imports: `from slitless.forward import forward_op` (NOT relative)
- Spectral orders: `[0, -1, 1, -2, 2][:K]`
- Parallelism: `from joblib import Parallel, delayed`

## Gotchas

1. 102 hardcoded `/home/kamo/` paths exist — never add more. Use `os.path.dirname(slitless.__file__)`.
2. `data_loader.py` calls `np.load()` at import time — blocks import if file missing. Don't replicate.
3. `recon.py` is monolithic (1448 lines). Add new solvers as separate modules, not inside it.
4. `meas4dar` must be sliced to `[:, :K]` before passing to `Reconstructor_Multi`; not sliced internally.
5. `intenscaling=True` locks the optimizer to a fixed intensity scale; keep `False` for standard evaluation.
6. MART spectral prior widths (`wid1`, `wid2`) are in Å, not km/s.
7. `idx_im=45` in `eis_test_100_dsetv5.npy` is an outlier (peak ~18,375). Exclude for fair comparison.

## Agent Skills

- `.agents/skills/slitless-conventions/SKILL.md` — conventions, import style, anti-pattern guardrails
- `.agents/skills/find-docs/SKILL.md` — fetch current docs before using library APIs (eispac, torch)