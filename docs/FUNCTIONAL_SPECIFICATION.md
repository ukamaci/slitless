# Slitless Spectral Imaging — Functional Requirement Specification

## Table of Contents

1.  [Domain Model](#1-domain-model)
    1.1 [Fundamental Quantities](#11-fundamental-quantities)
    1.2 [Source Entity](#12-source-entity)
    1.3 [Instrument (Imager) Entity](#13-instrument-imager-entity)
    1.4 [Unit Conversion](#14-unit-conversion)
2.  [Forward Measurement Model](#2-forward-measurement-model)
    2.1 [Gaussian Line Profile Functions](#21-gaussian-line-profile-functions)
    2.2 [Spectral Forward Model (2D)](#22-spectral-forward-model-2d)
    2.3 [Tomographic (Data Cube) Forward Model](#23-tomographic-data-cube-forward-model)
    2.4 [Tomographic Matrix Operator](#24-tomographic-matrix-operator)
    2.5 [Tomographic Transpose (Adjoint) Operator](#25-tomographic-transpose-adjoint-operator)
    2.6 [Data Cube from Parameters](#26-data-cube-from-parameters)
    2.7 [Simplified Blur-Only Forward Model (Educational)](#27-simplified-blur-only-forward-model-educational)
3.  [Noise Models](#3-noise-models)
    3.1 [Gaussian Noise](#31-gaussian-noise)
    3.2 [Poisson (Shot) Noise](#32-poisson-shot-noise)
    3.3 [No-Noise Mode](#33-no-noise-mode)
4.  [Source Parameter Reconstruction (Inverse Problem Solvers)](#4-source-parameter-reconstruction-inverse-problem-solvers)
    4.1 [SMART](#41-smart)
    4.2 [SMART2 — Two-Component + Background](#42-smart2--two-component--background)
    4.3 [Column-Wise Regularized Optimization (Scipy Solver)](#43-column-wise-regularized-optimization-scipy-solver)
    4.4 [Two-Component Column-Wise Solver (Scipy2)](#44-two-component-column-wise-solver-scipy2)
    4.5 [Gradient Descent Solver (Global, Differentiable)](#45-gradient-descent-solver-global-differentiable)
    4.6 [Neural Network Solver (U-Net)](#46-neural-network-solver-u-net)
    4.7 [Diffusion-Based Solver (DPS)](#47-diffusion-based-solver-dps)
    4.8 [Tomographic Inversion (Tomoinv)](#48-tomographic-inversion-tomoinv)
    4.9 [Prior-Based Solver](#49-prior-based-solver)
    4.10 [Gaussian Fitting from Data Cubes (Method of Moments)](#410-gaussian-fitting-from-data-cubes-method-of-moments)
    4.11 [Nonlinear Curve Fitting from Data Cubes](#411-nonlinear-curve-fitting-from-data-cubes)
    4.12 [Pseudoinverse + Gaussian Fitting (Linear Recon)](#412-pseudoinverse--gaussian-fitting-linear-recon)
    4.13 [FFT-Based Line Width Estimation](#413-fft-based-line-width-estimation)
5.  [Error and Quality Metrics](#5-error-and-quality-metrics)
    5.1 [Structural Similarity (SSIM)](#51-structural-similarity-ssim)
    5.2 [Normalized Root Mean Square Error (NRMSE)](#52-normalized-root-mean-square-error-nrmse)
    5.3 [Peak Signal-to-Noise Ratio (PSNR)](#53-peak-signal-to-noise-ratio-psnr)
    5.4 [Normalized Mean Squared Error (NMSE, batch)](#54-normalized-mean-squared-error-nmse-batch)
    5.5 [Cycle Loss (Measurement-Domain Consistency)](#55-cycle-loss-measurement-domain-consistency)
    5.6 [Total Variation (TV) Loss](#56-total-variation-tv-loss)
    5.7 [Gradient Residual Loss](#57-gradient-residual-loss)
    5.8 [Combined Loss Function](#58-combined-loss-function)
    5.9 [Error Statistics (RMSE, Bias)](#59-error-statistics-rmse-bias)
    5.10 [Per-Pixel Gaussian Fitting Accuracy](#510-per-pixel-gaussian-fitting-accuracy)
6.  [Data Management](#6-data-management)
    6.1 [Pre-Computed Dataset](#61-pre-computed-dataset)
    6.2 [On-the-Fly Dataset](#62-on-the-fly-dataset)
    6.3 [Data Normalization Transforms](#63-data-normalization-transforms)
    6.4 [Dataset Statistics Computation](#64-dataset-statistics-computation)
    6.5 [On-Demand Noise Injection](#65-on-demand-noise-injection)
    6.6 [Date-Aware Dataset Splitting](#66-date-aware-dataset-splitting)
7.  [Learning-Based Reconstruction (Neural Network Training)](#7-learning-based-reconstruction-neural-network-training)
    7.1 [Training Pipeline](#71-training-pipeline)
    7.2 [Training Metrics](#72-training-metrics)
    7.3 [Model Checkpointing](#73-model-checkpointing)
    7.4 [Training Diagnostics Output](#74-training-diagnostics-output)
    7.5 [Model Loading from Checkpoint](#75-model-loading-from-checkpoint)
    7.6 [Inference Prediction](#76-inference-prediction)
    7.7 [Multi-SNR Evaluation](#77-multi-snr-evaluation)
8.  [EIS Spectral Data Processing Subsystem](#8-eis-spectral-data-processing-subsystem)
    8.1 [Parallel EIS Spectral Fitting via MPFIT](#81-parallel-eis-spectral-fitting-via-mpfit)
    8.2 [Parameter Extraction from Fit Results](#82-parameter-extraction-from-fit-results)
    8.3 [Model-Based Spectrum Inpainting](#83-model-based-spectrum-inpainting)
    8.4 [Single-Pixel Spectral Fitting with Component Decomposition](#84-single-pixel-spectral-fitting-with-component-decomposition)
    8.5 [EIS to Slitless Spectral Imager (SSI) Interpolation](#85-eis-to-slitless-spectral-imager-ssi-interpolation)
    8.6 [Synthetic Spectrum from 3-Parameter Representation](#86-synthetic-spectrum-from-3-parameter-representation)
    8.7 [PMF-to-Physical Unit Conversion](#87-pmf-to-physical-unit-conversion)
    8.8 [Random Spatial Patch Extraction](#88-random-spatial-patch-extraction)
    8.9 [EIS Data Download](#89-eis-data-download)
    8.10 [EIS Data Cube Outlier Detection and Correction](#810-eis-data-cube-outlier-detection-and-correction)
    8.11 [EIS Parameter Clipping to Physical Bounds](#811-eis-parameter-clipping-to-physical-bounds)
    8.12 [Physics-Based Dataset Quality Filtering](#812-physics-based-dataset-quality-filtering)
    8.13 [Background Offset Calibration via Grid Search](#813-background-offset-calibration-via-grid-search)
    8.14 [EIS Detector Noise Characterization via Parametric Model](#814-eis-detector-noise-characterization-via-parametric-model)
    8.15 [EIS Template Manipulation](#815-eis-template-manipulation)
    8.16 [Global Template Fitting for Solver Initialization](#816-global-template-fitting-for-solver-initialization)
    8.17 [Dataset Statistical Profiling with Physics Filtering](#817-dataset-statistical-profiling-with-physics-filtering)
    8.18 [Full FOV Dataset Generation (No Cropping)](#818-full-fov-dataset-generation-no-cropping)
9.  [Visualization](#9-visualization)
    9.1 [Source Parameter Plot](#91-source-parameter-plot)
    9.2 [Measurement Plot](#92-measurement-plot)
    9.3 [Truth vs Reconstruction Comparison](#93-truth-vs-reconstruction-comparison)
    9.4 [Reconstruction Results (Neural Network)](#94-reconstruction-results-neural-network)
    9.5 [Loss Convergence Plot](#95-loss-convergence-plot)
    9.6 [Loss vs Iteration Plot](#96-loss-vs-iteration-plot)
    9.7 [Error Histograms](#97-error-histograms)
    9.8 [Joint Error Distribution Plots](#98-joint-error-distribution-plots)
    9.9 [Grouped Bar Plots](#99-grouped-bar-plots)
    9.10 [Comparison Sweep Lines](#910-comparison-sweep-lines)
    9.11 [FOV Spectra Plot (6-Panel Summary)](#911-fov-spectra-plot-6-panel-summary)
    9.12 [Multi-Position FOV Spectral Visualization](#912-multi-position-fov-spectral-visualization)
    9.13 [Spectral Fit Diagnostic Plot (Multi-Method)](#913-spectral-fit-diagnostic-plot-multi-method)
    9.14 [2D Parameter Map Comparison Grid](#914-2d-parameter-map-comparison-grid)
    9.15 [Patch Location Markers](#915-patch-location-markers)
    9.16 [Full FOV with Patch Overlay](#916-full-fov-with-patch-overlay)
    9.17 [Animated Map Display](#917-animated-map-display)
10. [Orchestration and Comparison](#10-orchestration-and-comparison)
    10.1 [Single-Source Reconstruction Pipeline (Reconstructor)](#101-single-source-reconstruction-pipeline-reconstructor)
    10.2 [Multi-Source Reconstruction Pipeline (Reconstructor Multi)](#102-multi-source-reconstruction-pipeline-reconstructor-multi)
    10.3 [Standardized Solver Comparison Testbed](#103-standardized-solver-comparison-testbed)
    10.4 [Results Persistence](#104-results-persistence)
    10.5 [Training Results Persistence](#105-training-results-persistence)
11. [Output Channel Adjustment](#11-output-channel-adjustment)
    11.1 [Channel Selection and Extension](#111-channel-selection-and-extension)
12. [Synthetic Data Generation](#12-synthetic-data-generation)
    12.1 [Patch Extraction from Images](#121-patch-extraction-from-images)
    12.2 [EIS-Derived Dataset Generation](#122-eis-derived-dataset-generation)
    12.3 [ImageNet-Derived Dataset Generation](#123-imagenet-derived-dataset-generation)
    12.4 [On-the-Fly Training Data from Images](#124-on-the-fly-training-data-from-images)
    12.5 [Dataset Tomographic Measurement Regeneration](#125-dataset-tomographic-measurement-regeneration)
    12.6 [Correlated Random Field Generation (Gaussian Smoothing)](#126-correlated-random-field-generation-gaussian-smoothing)
    12.7 [Worley (Cellular) Noise Texture Generation](#127-worley-cellular-noise-texture-generation)
    12.8 [Synthetic Test Pattern Generation](#128-synthetic-test-pattern-generation)
    12.9 [Plus/Cross Test Pattern Generation](#129-pluscross-test-pattern-generation)
    12.10 [Horizontal Lines Test Pattern](#1210-horizontal-lines-test-pattern)
13. [Appendix A: Physical Constants and Defaults](#13-appendix-a-physical-constants-and-defaults)
14. [Appendix B: Spectral Order Semantics](#14-appendix-b-spectral-order-semantics)
15. [Appendix C: Solver Capability Matrix](#15-appendix-c-solver-capability-matrix)
16. [Appendix D: Data Flow Diagram](#16-appendix-d-data-flow-diagram)
17. [Appendix E: External Dependencies (Interface Contracts)](#17-appendix-e-external-dependencies-interface-contracts)

---

## 1. Domain Model

### 1.1 Fundamental Quantities
The system models a spectroscopic imaging problem. At each spatial location (row, column) in a 2D field of view, the following three scalar values together describe the state:

| Parameter | Physical Unit | Pixel Unit | Description |
|-----------|--------------|------------|-------------|
| Intensity | raw counts (or erg/cm²/s/sr) | dimensionless (or scaled counts) | Emitted brightness |
| Velocity | km/s | pixels of shift | Line-of-sight Doppler shift |
| Line width | Angstroms (Å) | pixels of width | Spectral line broadening |

### 1.2 Source Entity
A **Source** is an aggregate of three equal-size 2D arrays (intensity, velocity, linewidth), a rest wavelength (Å), and a unit-system flag (physical or pixel). A Source supports:
- Construction from separate 2D arrays or from a 3-channel stacked array
- Construction from an existing 3-channel array (param3d)
- Visualization: 3-panel color image (hot/seismic/plasma colormaps) with optional SSIM/RMSE/PSNR annotations in the title

### 1.3 Instrument (Imager) Entity
An **Imager** describes the measurement hardware with:
- Pixel size (µm)
- Dispersion (µm/mÅ) or equivalently dispersion scale (mÅ/pixel or Å/pixel) — the two are reciprocally related
- Mid-wavelength (Å) — central wavelength of the detector array
- Spectral orders — list of integers representing which diffraction orders are measured (default [0, -1, 1])
- Pixelated flag — whether the forward model integrates Gaussians over pixel widths (True) or samples at midpoints (False)
- Spatial mask — 2D binary array of active/unmasked detector pixels
- Noise model parameters (SNR, max count, avg count, noise model type)
- Intensity scale factor for numerical conditioning

### 1.4 Unit Conversion
Physical-to-pixel and pixel-to-physical conversion uses:
- **Doppler formula**: `observed_wavelength = rest_wavelength × (1 + velocity / speed_of_light)`
- **Speed of light** = 299,792.458 km/s
- **Pixel shift**: `(observed_wavelength − mid_wavelength) / dispersion_scale`
- **Line width**: multiply or divide by dispersion_scale
- **Width to km/s**: multiply by `speed_of_light / rest_wavelength`
- **Intensity scaling**: multiply or divide by intensity scale factor
- Conversions operate on full 3-channel parameter arrays or on Source objects

---

## 2. Forward Measurement Model

### 2.1 Gaussian Line Profile Functions
Two evaluation modes for a Gaussian spectral line:
- **Continuous**: `G(x) = (1/σ√2π) × exp(−½((x−μ)/σ)²)` — evaluated at pixel midpoints
- **Pixelated**: `G_pix(x) = ½[erf((x−μ+0.5)/(√2σ)) − erf((x−μ−0.5)/(√2σ))]` — integral over one pixel width

Both are available in array-compute and differentiable (autodiff-compatible) forms.

### 2.2 Spectral Forward Model (2D)
Given three 2D arrays (intensity, velocity_pixels, linewidth_pixels) and a list of spectral orders:

For **order 0**: output = intensity × mask (direct mapping, no dispersion)

For **non-zero order k**: each spatial pixel emits a Gaussian line centered at `k × velocity` with width `|k| × linewidth`. The measurement at each detector row is the sum of contributions from all spatial pixels, weighted by their intensity. The operation is:

```
measurement[order_k, row, col] = Σ_{src_row} mask[src_row,col] × intensity[src_row,col] × G(row, k×velocity[src_row,col], |k|×linewidth[src_row,col])
```

Implemented via tensor contraction for efficiency. Supports batch processing (3D input with leading batch dimension).

### 2.3 Tomographic (Data Cube) Forward Model
Given a 3D spectral data cube of shape `(spectral_dim, height, width)` and a set of orders:

For **order 0**: sum the cube along the spectral axis

For **non-zero order k**: shear each spectral slice (at index λ) by `k × (λ − center_λ)` along the spatial (row) axis, then sum along the spectral axis. The shear uses periodic boundary conditions (roll).

For **special order "inf"**: sum along the spatial (row) axis instead, producing a spectral-only projection.

For **even orders** (±2): apply a boxcar averaging kernel along the spectral axis before shearing (to model spectral continuity over sub-pixel dispersion), with a half-pixel symmetry correction via [0.5, 0.5] convolution.

Supported orders: any combination from {0, −1, +1, −2, +2, inf}.

### 2.4 Tomographic Matrix Operator
Generate an explicit linear operator matrix for the tomographic model. For a cube shape and order list, produce a matrix mapping flattened cube to flattened measurements. Each non-zero order row uses a shift map; order 0 uses identity columns; order "inf" uses row-sum projections. For order ±2, the matrix uses linear-interpolation weights (±0.5 weights for 2-pixel partial assignments).

### 2.5 Tomographic Transpose (Adjoint) Operator
The adjoint of the tomographic forward model: back-projects measurement values along their dispersion paths to the data cube. For each order, measurement values are distributed uniformly across the spectral dimension at the sheared positions. Unmapped rows in the back-projection are filled with ones (for multiplicative reconstruction) or zeros (for additive reconstruction). Supports 2D (no batch) and 3D (with spatial batch) variants.

### 2.6 Data Cube from Parameters
Generate a 3D data cube from 3-channel parameters: at each spatial pixel, construct a Gaussian (pixelated or continuous) centered at `velocity + spectral_dim/2` with width `linewidth`, scaled by intensity. The cube spans a configurable number of wavelength bins.

### 2.7 Simplified Blur-Only Forward Model (Educational)
A simplified 1D-convolution forward model for tutorial use: each row of a scene is convolved with a Gaussian kernel whose width varies per pixel (provided by a blur-width map). Uses tensor contraction for batch efficiency. This is a different forward model from the spectral/tomographic ones above.

---

## 3. Noise Models

### 3.1 Gaussian Noise
Additive white Gaussian noise parameterized by SNR in dB:
- Compute signal variance per measurement channel
- Noise standard deviation = `sqrt(signal_variance / 10^(SNR_dB/10))`
- Add independent normal samples

### 3.2 Poisson (Shot) Noise
Photon-counting noise with three parameterization modes:
1. **Max count**: scale signal so peak = max_count, apply Poisson draws, unscale
2. **Average count**: scale signal so mean = avg_count, apply Poisson draws, unscale
3. **SNR-based**: compute avg_brightness = `(10^(SNR_dB/20))^2`, scale appropriately, apply Poisson, unscale

### 3.3 No-Noise Mode
Return signal unchanged (for clean evaluation).

---

## 4. Source Parameter Reconstruction (Inverse Problem Solvers)

### 4.1 SMART
**Type**: Tomographic, multiplicative-iterative, data-cube-based.

**Initialization**: Create data cube from zero-order measurement with uniform velocity = 0, uniform width = 1.38 pixels.

**Algorithm** (for each outer iteration):
1. **Contrast enhancement**: Transform `cube → (cube + cube^(1+ψ)) × Σcube / Σ(cube + cube^(1+ψ))` (power-law with energy conservation, configurable exponent ψ)
2. **3D smoothing**: Convolve cube with separable kernel `[0.25, 0.5, 0.25] ⊗ [0.25, 0.5, 0.25] ⊗ [0.25, 0.5, 0.25]` in (λ, y, x)
3. **Inner iterations** (configurable count):
   - Forward-project cube to measurements via tomographic matrix
   - Compute chi-squared per order: `mean((meas − est)² / (meas + ε))`
   - Skip converged orders (chi < 1e-10)
   - Compute correction factors: `correction = (meas / (est + ε))^(2/3)`
   - Back-project corrections via transpose operator
   - Voxels unconstrained by a given order's projection → set correction to 1
   - Multiply all unconverged-order corrections (geometric mean, equal weights)
   - Apply multiplicative correction to cube
4. Report chi-squared

**Post-processing**: Extract parameters via:
- **PMF fitter** (method of moments): Normalize each pixel's spectrum to unit sum, compute weighted mean → velocity, weighted std → linewidth, total sum → intensity
- **MPFIT fitter**: Fit each pixel's spectrum against an EIS template using MPFIT, extract component-0 parameters, convert physical units back to pixel units

**Infinite-order prior** (optional): Append a spatial-sum projection (uniform spectral shape) to measurements with a prior line width, forcing the reconstruction to match the global spectral profile.

**Configurable**: psi, max outer iterations, max inner iterations, prior width, fitter choice, template.

### 4.2 SMART2 — Two-Component + Background
**Type**: Same as SMART but models two Gaussian lines plus a background continuum.

**Initialization**: Build cube as sum of:
- Component 1: `Gaussian(intensity × frac1, cent1, wid1)`
- Component 2: `Gaussian(intensity × frac2, cent2, wid2)`
- Background: `normalized_spectral_shape[λ] × intensity × frac_bg`
- All parameters (fractions, centers, widths, background shape) are configurable

**Key differences from SMART**:
- Infinite-order prior: uses the actual cube's spatial sum (per-wavelength profile), clipped and renormalized to match zero-order total
- Prior carries a configurable **weight** in the correction factor
- Correction factors per order are raised to their order weight before geometric averaging; unconverged-order-to-active-weight ratio normalizes
- **Core-copy boundary condition**: unmapped spectral rows in a back-projection use the central row's correction value (preserving spectral shape at boundaries)
- Final PMF fitting subtracts background minimum before fitting

**Configurable**: fractions (frac1, frac2, frac_bg), centers (cent1, cent2 in Å), widths (wid1, wid2 in Å), background shape (normalized array per wavelength), psi, max outer/inner iterations, prior weight, live plot flag, initial cube override, fitter choice.

### 4.3 Column-Wise Regularized Optimization (Scipy Solver)
**Type**: Per-column, optimization-based (no data cube).

For each detector column independently, solve:
```
minimize_(int_row, vel_row, wid_row)  data_fidelity(forward(int,vel,wid), meas) + λ_i×TV(int) + λ_v×TV(vel) + λ_w×TV(wid)
```
- **Data fidelity**: L2 (sum of squares) or L1 (sum of absolute values)
- **Total Variation penalty**: `Σ(diff(row_i)²)` along the column for each parameter
- **Initialization**: intensity = zero-order measurement; velocity and linewidth = uniform values computed from rest wavelength, mid-wavelength, and dispersion scale
- **Optimizer**: configurable method (default L-BFGS-B), configurable max iterations
- Output shape: (3, rows, columns)

**Parallel variant**: Distributes per-column optimization tasks across multiple workers.

### 4.4 Two-Component Column-Wise Solver (Scipy2)
**Type**: Extension of column-wise solver for two Gaussians + background.

**Decision variables** (7 per spatial row): intensity1, velocity1, width1, intensity2, velocity2, width2, background_intensity.

**Objective** (per column):
```
minimize  data_fidelity(forward(G1) + forward(G2) + project(background), meas) + regularization
```

- Background projection: uses a precomputed tomographic projection matrix for a uniform-cube background basis, masked by the spatial mask
- Regularization weights dynamically scaled: intensity2 penalty = `λ_i × (frac1/max(frac2, ε))²`; background penalty = `λ_i × (frac1/max(frac_bg, ε))² × 100`
- Returns only component 1 parameters

### 4.5 Gradient Descent Solver (Global, Differentiable)
**Type**: Full-field optimization using automatic differentiation.

**Decision variables**: three 2D arrays (init: intensity = zero-order meas, velocity/width = uniform)

**Objective**:
```
minimize  data_fidelity(forward(int,vel,wid), meas) + λ_i×TV(int) + λ_v×TV(vel) + λ_w×TV(wid)
```

- **Optimizers**: Adam (lr configurable) or SGD
- **Data fidelity**: L1 or L2
- **Regularization**: Total Variation on all three maps
- Tracks loss per iteration; optionally tracks relative velocity/width error against ground truth
- Can save intermediate reconstructions at specified intervals to disk

### 4.6 Neural Network Solver (U-Net)
**Type**: Learned direct inversion via trained convolutional network.

**Architecture**: U-Net with configurable:
- Input channels (number of spectral orders / detectors)
- Output channels (1 for single-parameter, 3 for all parameters)
- Number of down/up layers
- Starting filter count
- Kernel sizes per layer
- Bilinear upsampling or transposed convolution
- Residual mode (output difference from input)

**Output modes**: "all" (3 channels: intensity, velocity, linewidth), "int" (intensity only), "vel" (velocity only), "width" (linewidth only).

**Inference**: Load checkpoint → normalize input measurements → forward pass → inverse-transform output parameters.

### 4.7 Diffusion-Based Solver (DPS)
**Type**: Learned prior (DDPM) combined with measurement-consistency gradient guidance.

- Load pre-trained U-Net denoiser model from diffusion training checkpoint
- Configure diffusion: image_size, timesteps (1000), sampling_timesteps, cosine beta schedule, gradient_scale per parameter channel
- During each denoising step: model denoises, then measurement-consistency gradient pulls the sample toward the measurements
- Draw multiple samples; final estimate = sample mean

### 4.8 Tomographic Inversion (Tomoinv)
**Type**: Iterative tomographic with Gaussian projection constraint.

**Setup**: Uses fixed tomographic matrix for 21 spectral wavelengths, orders [0, −1, 1].

**Two modes**:
1. **Inverse mode**: Compute regularized pseudoinverse solution `Hᵀy` → `(HᵀH + λI)⁻¹Hᵀy`, then iterate: `r ← proj_gauss(pinv + λ × (HᵀH + λI)⁻¹ × r)`, decaying λ by 0.95 each iteration
2. **Gradient mode**: `r ← proj(r − step_size × (HᵀHr − Hᵀy))`

**Projection modes** (proj):
- **Gaussian**: reshape to cube → fit each pixel spectrum to Gaussian (method of moments) → regenerate cube
- **Positivity**: clip negatives to zero (optionally combined with Gaussian)
- **Gaussian+Positivity**: clip then fit Gaussian

Configurable: data_step, positivity, projection type, initial estimate, step size, regularization λ, iterations.

### 4.9 Prior-Based Solver
**Type**: Non-iterative, parametric-prior estimate.

Estimates: intensity = `frac1 × meas_order_0`; velocity = uniform value from (prior_center_wavelength − mid_wavelength) / dispersion_scale; linewidth = uniform value from prior_width / dispersion_scale.

### 4.10 Gaussian Fitting from Data Cubes (Method of Moments)

**Version 1**: Normalize each pixel's spectrum to unit sum (probability mass function). Compute intensity = total sum; velocity = weighted mean − spectral_center; linewidth = sqrt(weighted variance).

**Version 2**: Same but with edge-case handling: divide by intensity only where > 0; clip intensity to [0, 1], velocity to [−2, 2] pixels, linewidth to [0.5, 2.3] pixels; NaN → defaults.

### 4.11 Nonlinear Curve Fitting from Data Cubes
Fit each spatial pixel's spectrum to a pixel-integrated Gaussian via Levenberg-Marquardt (`curve_fit`): 3 parameters (intensity, centroid, width) bounded to [0, −2, 1] / [1, 2, 2.3], max 5000 function evaluations.

### 4.12 Pseudoinverse + Gaussian Fitting (Linear Recon)
Solve `H × x = y` via pseudoinverse, reshape the solution into a data cube, then refine per-column via parametric Gaussian fitting. Used for comparison/plotting.

### 4.13 FFT-Based Line Width Estimation
Estimate average line width by computing the ratio of Fourier transforms of spectral orders: `|FFT(order_1) / FFT(order_0)|` or `|FFT((order_1 + order_−1)/2) / FFT(order_0)|`. The magnitude ratio forms a Gaussian in frequency space; fit its width via `curve_fit` to recover the underlying line width. Exploits the property that Doppler shifts appear as phase modulation.

---

## 5. Error and Quality Metrics

### 5.1 Structural Similarity (SSIM)
Per-2D-channel SSIM between truth and estimate, with data_range = `max(truth) − min(truth)`. Vectorized over leading dimensions.

### 5.2 Normalized Root Mean Square Error (NRMSE)
`sqrt(mean((truth − estimate)²)) / normalization` where normalization is: `'sigma'` (std of truth), `'minmax'` (range of truth), or `None` (1 = raw RMSE). Vectorized.

### 5.3 Peak Signal-to-Noise Ratio (PSNR)
`10 × log₁₀(max(truth)² / mean((truth − estimate)²))`. Vectorized.

### 5.4 Normalized Mean Squared Error (NMSE, batch)
Per-channel NMSE: `mean(((truth − estimate) / (max(truth) − min(truth)))²)`, where max/min are per-channel per-sample, then averaged across batch.

### 5.5 Cycle Loss (Measurement-Domain Consistency)
Pass estimate parameters through forward model; compute L2 or L1 difference against original measurements: `mean((forward(estimate) − measurements)²)`.

### 5.6 Total Variation (TV) Loss
`mean(|img[row+1, col] − img[row, col]|) + mean(|img[row, col+1] − img[row, col]|)`

### 5.7 Gradient Residual Loss
First and second spatial derivatives of residual image, penalized by L1 or L2, with predefined weights: dx (0.5), dy (0.5), d²x (0.25), d²y (0.25), dxy (0.25).

### 5.8 Combined Loss Function
Weighted sum of parameter-space losses (MSE, L1, NMSE) and measurement-domain losses (cycle loss), with per-loss weights. For single-channel output modes, substitutes the network output into the appropriate channel of a cloned truth tensor.

### 5.9 Error Statistics (RMSE, Bias)
Per-channel: `RMSE = sqrt(mean((truth − estimate)², axis=spatial))`; `Bias = mean(estimate − truth, axis=spatial)`.

### 5.10 Per-Pixel Gaussian Fitting Accuracy
For comparing spectral fitting methods on a single pixel: compute fitted parameters (intensity, velocity, width) from the fit results, compare against target parameters using RMSE and Bias in physical units.

---

## 6. Data Management

### 6.1 Pre-Computed Dataset
Load measurement–parameter pairs from `.npy` dictionary files. Each file contains keys: `int`, `vel`, `width` (parameter maps), `meas_0`, `meas_-1`, `meas_1`, `meas_-2`, `meas_2` (measurement maps for up to 5 spectral orders). Dataset is organized into train/val/test subdirectories.

### 6.2 On-the-Fly Dataset
Generate training data dynamically from stored parameter maps:
- Load intensity, velocity, linewidth arrays from disk
- Randomize velocity: sample scaling from uniform [0, vel_max], map to [−scale, +scale]
- Randomize linewidth: sample min and max from two uniform distributions, map to [min, max]
- Compute measurements via the forward model in batches (GPU-accelerated)
- Supports multi-part data files for distributed training

### 6.3 Data Normalization Transforms
- **Measurement transform**: divide by max_intensity (6000)
- **Measurement inverse**: multiply by max_intensity
- **Parameter transform**: divide intensity by max_intensity; standardize velocity: `(vel − vel_mean) / vel_std`; standardize linewidth: `(width − width_mean) / width_std`
- **Parameter inverse transform**: reverse, with optional linewidth conversion from Å to km/s
- Statistics (max_intensity, means, stds) loaded from a pre-computed stats file

### 6.4 Dataset Statistics Computation
Given a dataset directory, compute: `int_max`, `int_mean`, `meas_mean`, `vel_mean`, `vel_std`, `width_mean`, `width_std` across all training samples.

### 6.5 On-Demand Noise Injection
Apply noise to measurements at dataset access time with configurable SNR and noise model.

### 6.6 Date-Aware Dataset Splitting
Split `.npy` patch files into train/val/test by extracting unique observation date strings from filenames. All patches from the same observation raster go to the same split (80/10/10). Uses random split at the raster level to prevent spatial-correlation leakage across splits.

---

## 7. Learning-Based Reconstruction (Neural Network Training)

### 7.1 Training Pipeline
Train a U-Net to map measurements → parameters:
- Configurable: input channels, output channels, output mode, starting filters, layers, kernel sizes, bilinear upsampling, residual mode
- Configurable: optimizer (Adam/SGD), loss function (MSE/L1/NMSE), learning rate, epochs, batch size
- Optional cycle loss with configurable weight
- On-the-fly dataset regeneration every N epochs (with rotation through 5 data parts)
- Validation every N epochs
- Best model saved by validation loss
- Keyboard-interrupt handling (saves interrupted state)

### 7.2 Training Metrics
Per epoch tracked: training loss, validation loss, per-channel SSIM (train/val), per-channel RMSE (train/val).

### 7.3 Model Checkpointing
- Best validation-loss model saved as `best_model.pth`
- Final model saved after training completion
- Interrupted model saved on keyboard interrupt

### 7.4 Training Diagnostics Output
- Loss convergence plot (semilog, train + validation)
- SSIM vs epoch plot (per channel, train + validation)
- RMSE vs epoch plot (per channel, train + validation)
- Oracle baseline: SSIM/RMSE/MAE if predicting the training mean (constant predictor)
- Summary text file with all parameters, baselines, and final metrics

### 7.5 Model Loading from Checkpoint
Parse training summary text file to recover: number of starting filters, input channels, output channel type, kernel sizes, bilinear flag, number of layers. Reconstruct U-Net architecture, load weight checkpoint, set to eval mode.

### 7.6 Inference Prediction
Run trained network on batched or single measurements → return estimated parameter maps.

### 7.7 Multi-SNR Evaluation
Evaluate trained model at multiple SNR levels (configurable list): rebuild dataset with each SNR, run inference, collect SSIM and RMSE per SNR.

---

## 8. EIS Spectral Data Processing Subsystem

### 8.1 Parallel EIS Spectral Fitting via MPFIT
Fit each spatial pixel's observed EUV spectrum against a multi-Gaussian + polynomial template:
- Prepare data: copy to safe arrays, apply mask (set masked→0), handle bad data (negative errors→0)
- For each spatial row, distribute pixels across workers
- Each pixel fit: select in-window wavelengths, compute initial guess via `scale_guess`, run MPFIT with configurable tolerances and max iterations
- Return: fitted parameters per pixel, parameter errors, fit status codes, chi-squared values
- Configurable: number of parallel workers, minimum data points required for fit, Gaussian component index to extract

### 8.2 Parameter Extraction from Fit Results
Extract intensity, velocity, and linewidth for a specified Gaussian component:
- Intensity = `√(2π) × peak × width`
- Velocity = `speed_of_light × (fitted_centroid − rest_wavelength) / rest_wavelength`
- Linewidth = `fitted_width`
- Error propagation for intensity from peak + width errors
- Bad fits (status ≤ 0) → zero all parameters

### 8.3 Model-Based Spectrum Inpainting
After fitting, for each spatial pixel with masked wavelength points:
- Evaluate the full fitted model (multi-Gaussian + polynomial) at all wavelengths
- Replace only the masked wavelength values with the model values
- Leaves unmasked data untouched

### 8.4 Single-Pixel Spectral Fitting with Component Decomposition
Fit a single pixel spectrum using MPFIT against a template:
- Support Poisson error weighting (`error = sqrt(clip(|signal|, min_count, ∞))`) or uniform weighting
- Return: raw parameters, dense wavelength grid, combined fit curve, per-Gaussian component curves, background curve
- Raises error on failed fit

### 8.5 EIS to Slitless Spectral Imager (SSI) Interpolation
Convert EIS native-wavelength data cube to uniform-wavelength SSI data cube:
- Target grid: reference wavelength ± dispersion_step × (index − center_index)
- For each spatial pixel: interpolate native wavelength→flux onto target grid
- Configurable interpolation method (e.g., cubic) and edge-padding mode (constant with edge values or zero)
- Handle non-monotonic or insufficient valid wavelength points

### 8.6 Synthetic Spectrum from 3-Parameter Representation
Generate a Gaussian spectral curve (as peak flux density) from 3-parameter tuple:
- `center = rest_wavelength × (1 + velocity / speed_of_light)`
- `peak = intensity / (√(2π) × width)`
- Return: `peak × exp(−½((wave − center) / width)²)`
- This is the reverse of the PMF fitter: parameters → spectrum

### 8.7 PMF-to-Physical Unit Conversion
Convert PMF fitter output (pixel units) to physical units:
- Intensity: multiply by dispersion_scale (Å/pixel)
- Velocity: multiply by dispersion_scale × speed_of_light / rest_wavelength
- Linewidth: multiply by dispersion_scale

### 8.8 Random Spatial Patch Extraction

**Intensity-weighted**: Sample random top-left coordinates, score by total intensity sum in the patch, return top-N patches by score with their coordinates. Pool size = min(32, 3 × N).

**Mask-aware**: Pre-compute integral image of the binary mask. For all valid patch positions, compute mask-sum in O(1); exclude any position with mask-sum > 0. Sample from remaining valid positions, score by intensity, return top-N patches with coordinates and measurement patches.

### 8.9 EIS Data Download
Download EIS HDF5 data/header files from a remote HTTP mirror for a given observation date:
- Construct URL from date components (YYYY/MM/DD)
- Use curl or wget with skip-if-exists
- Save to local directory

### 8.10 EIS Data Cube Outlier Detection and Correction

**Detection**:
- `zero_masker`: flag spectral slices where zero-pixel count is between 50 and `(total_pixels − 50)`
- `negative_masker`: flag pixels with value < −1000

**Correction**: per-pixel 1D cubic interpolation across flagged spectral points.

### 8.11 EIS Parameter Clipping to Physical Bounds
Clip parameter arrays to physically valid ranges:
- Intensity ∈ [0, 6000] (for EIS Fe XII data)
- Velocity ∈ [−68.5, +68.5] km/s
- Linewidth ≥ 0.022 Å

### 8.12 Physics-Based Dataset Quality Filtering
Iterate over `.npy` dataset files; compute a validity mask: intensity ∈ [100, 6000], |velocity| < 68.5 km/s, width > 0.022 Å. Delete any file where any single pixel violates these bounds. Optionally scale measurement arrays by dispersion_scale.

### 8.13 Background Offset Calibration via Grid Search
Perform brute-force 2D grid search for best affine background model: `meas_0 ≈ (α+1) × intensity + β`. Search α ∈ [0, 0.3] and β ∈ [0, 495] over a 100×100 grid, minimizing squared error.

### 8.14 EIS Detector Noise Characterization via Parametric Model
Fit a photon-transfer-curve noise model: `error = sqrt(a × |intensity| + b)` (where a = gain, b = read-noise²). Apply to multiple EIS observation dates to characterize detector noise scaling with intensity, evaluating fit quality via R².

### 8.15 EIS Template Manipulation
**Background stripping**: Deep-copy an EIS template, set `n_poly = 0`, truncate `parinfo` and `fit` arrays to include only Gaussian parameters. Produces a Gaussian-only template for comparison testing.

### 8.16 Global Template Fitting for Solver Initialization
Fit a 2-Gaussian + polynomial model to the global mean spectrum of a dataset to derive solver initialization parameters: `frac1`, `frac2`, `frac_bg`, `cent1`, `cent2`, `wid1`, `wid2`, `bg_shape_norm`. Compute separately per dataset.

### 8.17 Dataset Statistical Profiling with Physics Filtering
Load EIS patches, compute raw and physics-filtered statistics (percentiles at 0/1/5/50/95/99/100, mean, std) for intensity, velocity, linewidth. Generate distribution histograms and 2D histograms (e.g., intensity vs velocity with LogNorm).

### 8.18 Full FOV Dataset Generation (No Cropping)
For each EIS observation: download → fit spectra with joblib → interpolate to SSI cube → forward-project to measurements (orders 0, −1, +1) → save full-FOV data including parameter maps, 3 measurement maps, and data cube.

---

## 9. Visualization

### 9.1 Source Parameter Plot
3-panel figure: intensity (hot colormap), velocity (seismic colormap), linewidth (plasma colormap). With optional SSIM/RMSE/PSNR annotations per panel.

### 9.2 Measurement Plot
K-panel figure (one per spectral order), hot colormap, each titled by order number.

### 9.3 Truth vs Reconstruction Comparison
Multi-row grid: reconstructed parameters (top) and true parameters (bottom), each panel with colormap and colorbar.

### 9.4 Reconstruction Results (Neural Network)
3-row grid: measurements (row 1), true parameters (row 2), predicted parameters (row 3). Single-channel output modes show only the relevant panel. Annotated with SSIM and RMSE per channel.

### 9.5 Loss Convergence Plot
Line plot of loss vs iteration/epoch, with grid, optionally semilog y-scale.

### 9.6 Loss vs Iteration Plot
Line plot with grid showing loss over optimizer iterations.

### 9.7 Error Histograms
Per-channel histograms (20 bins, orange fill, black edges) of reconstruction errors, titled with RMS error and bias, for both physical and pixel units.

### 9.8 Joint Error Distribution Plots
Hexbin joint plots of error vs true value for each parameter channel, showing bias and error standard deviation. Plus cross-dependence plots (velocity error vs intensity, linewidth error vs intensity).

### 9.9 Grouped Bar Plots
Multi-group, multi-member bar chart with configurable group labels, member labels, axis labels; bar values annotated.

### 9.10 Comparison Sweep Lines
Line plot with markers showing a metric vs a swept parameter, one line per method with legend, dashed lines, grid.

### 9.11 FOV Spectra Plot (6-Panel Summary)
2×3 grid: row 1 = intensity/velocity/linewidth, row 2 = measurements (order 0/−1/+1). Uses WCS coordinates for annotation when available.

### 9.12 Multi-Position FOV Spectral Visualization
Read EIS data cube, extract spectra at specified grid positions (or custom index list), plot each with error bars in a multi-panel figure, within a configurable wavelength window.

### 9.13 Spectral Fit Diagnostic Plot (Multi-Method)
For a single pixel: plot observed spectrum (black dots), combined fit (red line), individual Gaussian components (blue/green/magenta dashed), background (cyan dash-dot), ideal param3d curve (black dotted). Show fitted parameter values in a text box.

### 9.14 2D Parameter Map Comparison Grid
Multi-column grid: true parameter maps (column 1) vs reconstructed maps from multiple fitting methods (columns 2+), 3 rows (intensity/velocity/linewidth), each annotated with RMSE and Bias.

### 9.15 Patch Location Markers
Overlay rectangle patches on a parameter plot (intensity map) to show where patches were extracted from the full FOV.

### 9.16 Full FOV with Patch Overlay
Plot full-FOV intensity map with blue rectangle overlays marking extracted patch locations.

### 9.17 Animated Map Display
Loop display of multiple 2D maps with updated colormap per frame (for exploring generated datasets).

---

## 10. Orchestration and Comparison

### 10.1 Single-Source Reconstruction Pipeline (Reconstructor)
1. Take Imager (with configured noise params) and solver function
2. Simulate measurements: forward model + add noise
3. For neural network solver: apply measurement normalization before solve, inverse-transform parameters after solve
4. Run solver → get reconstructed parameters and loss trace
5. Support multiple noise realizations (re-simulate noise, re-solve)
6. Package results as Recon object with evaluation

### 10.2 Multi-Source Reconstruction Pipeline (Reconstructor Multi)
1. Take array of parameter maps (param4dar) or pre-computed measurements (meas4dar)
2. For each source: optionally scale intensity, create Imager, optionally set measurements, create Reconstructor, solve
3. Collect per-source metrics: SSIM, RMSE (pixel), RMSE (physical), MAE (pixel), MAE (physical), Bias (pixel), Bias (physical)
4. Also collect mean-baseline metrics (predicting spatial mean per channel)
5. Return all results aggregated

### 10.3 Standardized Solver Comparison Testbed
Compare solvers on a dataset with uniform measurement configuration:
- Dispatch: `'scipy'` → column-wise optimization, `'tomoinv'` → tomographic inversion, `'gd'` → gradient descent, `'nn'` → U-Net, `'diffusion'` → diffusion, `'smart'` → SMART
- Configurable: data path, data file, single/multi source, save flag, number of detectors (spectral orders), SNR, noise model
- Configurable solver-specific keyword arguments
- Output: reconstruction time, RMSE/MAE/SSIM/Bias statistics, comparative plots, error histograms, pickle save

### 10.4 Results Persistence
Save to timestamped directory:
- Summary text: all parameters, metrics, solver name, times, errors
- Per-source reconstruction figure (truth vs estimate)
- Error histogram figures (pixel and physical units)
- Recon arrays (.npy)
- Full Reconstructor object (pickle)

### 10.5 Training Results Persistence
Save training outputs to timestamped directory:
- Summary text: network parameters, optimization parameters, data parameters, baselines, final metrics
- Convergence plot
- SSIM and RMSE vs epoch plots
- Per-SNR bar plots
- SSIM/RMSE arrays (.npy)
- Model checkpoint files (.pth)

---

## 11. Output Channel Adjustment

### 11.1 Channel Selection and Extension
For networks that output only a subset of channels:
- **Crop mode**: select the matching channels from ground truth
- **Extend mode**: insert the network output into a full 3-channel tensor (copying truth into the other channels)

---

## 12. Synthetic Data Generation

### 12.1 Patch Extraction from Images
Extract 5 patches of configurable size from the center and quadrant centers of input images. Save as grayscale images.

### 12.2 EIS-Derived Dataset Generation
Tile EIS parameter maps into patches of configurable size, apply forward model to generate measurements, save as `.npy` dictionary files with intensity/velocity/width and per-order measurements.

### 12.3 ImageNet-Derived Dataset Generation
Load ImageNet 64×64 batches, convert to grayscale, rotate 90°, normalize per-image to [0, 1], split into train/val/test partitions.

### 12.4 On-the-Fly Training Data from Images
For each data point: randomly select 3 grayscale image patches (for intensity, velocity, linewidth), normalize each to [0, 1], apply random velocity scale (uniform [0.1, 2.0] pixels) and random linewidth range (min ∈ [1.0, 1.3], max ∈ [1.3, 2.2] pixels, linearly increasing with min), generate measurements via forward model, save.

### 12.5 Dataset Tomographic Measurement Regeneration
For an existing dataset of patches with 3D data cubes: regenerate all 5 measurement channels (orders 0, −1, +1, −2, +2) from the data cube via the tomographic forward model, overwrite original measurements, recompute normalization statistics.

### 12.6 Correlated Random Field Generation (Gaussian Smoothing)
Generate smooth 2D random maps by filtering white noise with zero-mean Gaussian kernel `sigma=(0, 4, 4)`, clipping to [−0.5, 0.5], shifting to [0, 1], scaling to [vmin, vmax].

### 12.7 Worley (Cellular) Noise Texture Generation
Generate procedural textures using Voronoi/Worley noise: place random feature points, compute distance to nearest and second-nearest point per grid cell, output `√(f2) − √(f1)`. Normalize and optionally scale peak to [0.7, 1.0] per scene. Batch generation supported.

### 12.8 Synthetic Test Pattern Generation
- **Sinusoidal grid**: `cos(kx×x) × sin(ky×y)` over a square grid
- **Block letter "I"**: rendered via drawing primitives (rectangles for top bar, stem, bottom bar with border outline), normalized to [0, 1]
- **Combined phantom**: weighted sum of sinusoidal grid + block letter, split across 3 channels with different scalings and offsets for intensity/velocity/linewidth testing

### 12.9 Plus/Cross Test Pattern Generation
Generate image with superposition of plus (+) and cross (×) signs using anti-aliased line drawing with Gaussian blur smoothing.

### 12.10 Horizontal Lines Test Pattern
Generate image with equispaced horizontal lines using anti-aliased line drawing with Gaussian blur.

---

## 13. Appendix A: Physical Constants and Defaults

| Constant | Value | Units |
|----------|-------|-------|
| Speed of light | 299,792.458 | km/s |
| Default rest wavelength | 195.117937907451 | Å |
| Default mid-wavelength | 195.119 | Å |
| Default dispersion scale | 0.022275 | Å/pixel |
| Default pixel size | 13.5 | µm |
| Default dispersion | 1/1.65 | µm/mÅ |
| Default spectral dimension | 21 | pixels |
| Max intensity (normalization) | 6000 | counts |
| EIS intensity valid range | [100, 25000] | counts |
| EIS velocity valid range | [−68.5, +68.5] | km/s |
| EIS linewidth minimum | 0.022 | Å |

---

## 14. Appendix B: Spectral Order Semantics

| Order | Meaning |
|-------|---------|
| 0 | Undispersed (direct) image |
| −1 | First negative diffraction order — dispersed opposite direction |
| +1 | First positive diffraction order — dispersed in dispersion direction |
| −2 | Second negative diffraction order |
| +2 | Second positive diffraction order |
| inf | "Infinite order" — spatial sum across dispersion direction (column sum, used as prior) |

---

## 15. Appendix C: Solver Capability Matrix

| Solver | Data-Cube Based | Multi-Component | Background | Learned | Per-Column | Iterative |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| SMART | ✓ | | | | | ✓ |
| SMART2 | ✓ | ✓ | ✓ | | | ✓ |
| Scipy | | | | | ✓ | ✓ |
| Scipy2 | | ✓ | ✓ | | ✓ | ✓ |
| Gradient Descent | | | | | | ✓ |
| U-Net | | | | ✓ | | |
| Diffusion DPS | | | | ✓ | | ✓ |
| Tomoinv | ✓ | | | | | ✓ |
| Prior Solver | | | | | | |
| FFT (Line Width) | | | | | | |

---

## 16. Appendix D: Data Flow Diagram

```
Source (intensity, velocity, linewidth maps)
    │
    ├──→ 2D Spectral Forward Model ──→ Clean Measurements (per spectral order)
    │         or
    ├──→ Data Cube Generator ──→ 3D Spectral Cube ──→ Tomographic Forward ──→ Clean Measurements
    │                                                        │
    │                                                  Transpose (Adjoint)
    │                                                        │
    │                                              Iterative Solvers (SMART, Tomoinv)
    │
    ├──→ Noise Model (Gaussian / Poisson) ──→ Noisy Measurements
    │                                                   │
    │                          ┌──────────────┬─────────┼──────────┬──────────────┐
    │                          ▼              ▼         ▼          ▼              ▼
    │                      SMART/        Gradient    U-Net    Diffusion     Column-Wise
    │                      Scipy         Descent     (NN)     (DPS)        Optimization
    │                          │              │         │          │              │
    │                          └──────────────┴─────────┴──────────┴──────────────┘
    │                                                   │
    │                                     Reconstructed Parameters
    │                                                   │
    └──────────────────→ Error Metrics ←────────────────┘
                    (SSIM, NRMSE, PSNR, RMSE, MAE, Bias)
```

---

## 17. Appendix E: External Dependencies (Interface Contracts)

The system must interface with the following external capabilities:

| Capability | Purpose |
|-----------|---------|
| Multi-dimensional array computation | All numerical operations on 2D/3D/4D arrays |
| Automatic differentiation | Gradient computation for GD and diffusion solvers |
| GPU-accelerated tensor computation | Batch forward model and neural network operations |
| Nonlinear optimization (L-BFGS-B, etc.) | Column-wise scipy solvers |
| Nonlinear least-squares curve fitting (Levenberg-Marquardt) | Data cube spectrum fitting |
| MPFIT (Levenberg-Marquardt in IDL heritage) | EIS spectral template fitting |
| Convolutional neural network construction | U-Net architecture |
| Denoising diffusion probabilistic model | Diffusion-based solver |
| Image structural similarity (SSIM) computation | Quality metric |
| Gaussian filtering / convolution | Smoothing in SMART, synthetic data generation |
| 1D interpolation | EIS-to-SSI wavelength resampling |
| Parallel job execution | Parallel EIS fitting, parallel column optimization |
| Statistical plotting (histograms, joint plots, bar charts) | All visualization |
| Image I/O | Data loading and figure saving |
| File download (HTTP) | EIS data acquisition |
| Template-based spectral fitting (eispac) | EIS spectral analysis |
