# Dead Script Inventory

> **Status: Audit artifact.** Lists identified dead/unused Python scripts in
> `python/scripts/`. 34 of 51 files are superseded or one-off experiments.
> Delete or archive before undertaking structural refactoring — dead files
> obscure which scripts are canonical entry points and pollute automated
> code scanning.

---

## Summary

| Category | Count |
|----------|-------|
| Dead (superseded or one-off) | 34 |
| Alive (current, actively used) | 13 |
| Uncertain (needs manual review) | 4 |
| **Total** | **51** |

---

## Dead Scripts (34)

| File | Evidence | Superseded By |
|------|----------|---------------|
| `apj_plotter.py` | Manual data entry, hardcoded `/home/kamo/` path, references `apj2024` figures | `apj_plotter2.py` |
| `apj_tomo_plotter.py` | Synthetic toy data only, one-off tomographic experiment, content duplicated in `tomoinv_tester.py` | — |
| `background_inspection.py` | Hardcoded path to `dset_v4`, one-off visualization | — |
| `basp_figger.py` | References `dset_v0_12_scans/eistest256/` (v0), baseline figure generation | — |
| `check_v5_stats_full.py` | Empty file (0 lines), never implemented | — |
| `comparison_testbed.py` | Uses `dset_v1` + APJ2019 date, single-detector comparison | `comparison_testbed_multi.py` |
| `cyc_loss_testing.py` | Hardcoded 2022 timestamps in model folders, one-off cycle loss experiment | — |
| `dataset_analysis.py` | Hardcoded path to `dset_v4/data/train/`, v5 is current canonical dataset | — |
| `dataset_generation.py` | Creates v4 dataset using `mas.forward_model`, v5 handled by newer script | `generate_dset_v5.py` |
| `dset_v3_fixer.py` | Targets `dset_v3/data/` for fixing outdated format (2 versions behind v5) | — |
| `eis_fitting_test.py` | One-off comparison of 1c vs 2c template fits, single analytical question | — |
| `eis_preprocessor.py` | References `dset_v0_raw` and `dset_v0_trimmed`, preprocessing for v0 | — |
| `eis_reader.py` | Original v1 EIS reader, single-date processing | `eis_reader_v3.py` |
| `eis_reader_v2.py` | Dated 2025-12-24, saves to `dset_v3/`, manual fitting logic | `eis_reader_v3.py` |
| `eis_to_ssi.py` | Single-date processing, manual interpolation loop, now in `eistools.py` | `eis_reader_v3.py` |
| `figen_comparison.py` | References external `/home/kamo/resources/tip2014/` (MATLAB data from another group) | — |
| `forward_eis_exp.py` | References `dset0_2022_06_14/` and `dset_v1`, imports `mas.forward_model` | — |
| `gaussfit_testbed.py` | Uses `glob.glob('/.../eistest64*')`, FFT-based fitting test, experimental one-off | — |
| `grad_descent.py` | Dated 2022-08-09, manual GD loop | `recon.py:grad_descent_solver` |
| `grad_param_search.py` | Dated 2022-10-20, imports `expsweep`, references `dset6*` | `auto_param_searcher.py` |
| `param_searcher.py` | Uses `dset_v1` + APJ2019 date, grid search with old `Reconstructor` | `auto_param_searcher.py` |
| `param_searcher_multi.py` | Uses `eis_train_5_64x64.npy` (older format), old `scipy_solver` | `auto_param_searcher.py` |
| `rmse_testbed.py` | Dated 2022-10-18, hardcoded 2022 model folder, references `dset6*` | — |
| `sandbox_1dmap_test.py` | "sandbox" in name, debugging/development script | — |
| `sandbox_error_analysis.py` | "sandbox" in name, one-off error characterization | — |
| `sandbox_fit_debug.py` | "sandbox" in name, 30-line debug script for `smart2` fitting | — |
| `scipy_solver.py` | Dated 2022-10-31, references `dset_v1` + APJ2019, prototype solver | `recon.py` |
| `slitlessfig.py` | References `dset0_2022_06_14/` and `dset_v1`, imports `mas.forward_model` | — |
| `test_fitters.py` | Uses `dsetv4` data, compares PMF vs EIS fitters on single sample | `datacube_fitting_comparison.py` |
| `test_smart.py` | Runs `smart2` on 50 v5 samples for diagnostic stats, single-purpose test | — |
| `tomo_fwd_tester.py` | Compares multiple tomo forward variants on synthetic data | `tomo_fwd_tester_v2.py` |
| `tomoinv_tester.py` | Synthetic data tomoinversion test, content nearly identical to `apj_tomo_plotter.py` | — |
| `tutorial_simplified.py` | Jupyter notebook converted to `.py`, generic ML tutorial unrelated to slitless | — |

---

## Alive Scripts (13)

| File | Purpose |
|------|---------|
| `apj_plotter2.py` | Latest publication figure generator (K_sweep + Gamma_sweep data) |
| `auto_param_searcher.py` | Optuna-based hyperparameter search for solvers |
| `check_v5_stats.py` | Verifies `dset_v5/norm_stats.npy` against actual training data |
| `comparison_testbed_multi.py` | Current comprehensive multi-solver testbed |
| `datacube_fitting_comparison.py` | Compares PMF vs EIS fitters with structured metrics |
| `eis_full_fov_generator.py` | Generates full FOV datasets from EIS observations |
| `eis_reader_v3.py` | Latest EIS reader with `fit_spectra_joblib` and outlier correction |
| `final_result_runner.py` | Primary result runner for current paper (7 SNR/configurations) |
| `forward_1d_exp.py` | Clean 1D forward experiment, reusable educational script |
| `generate_dset_v5.py` | Generates v5 tomographic measurements from datacubes |
| `generate_testbed_set.py` | Creates consolidated test/train/val `.npy` sets |
| `optimal_init_search.py` | Searches for optimal eispac fit initializations |
| `plot_fov_spectra.py` | Reusable FOV spectrum visualization utility |
| `tomo_fwd_tester_v2.py` | Latest tomographic forward verification with real EIS data |

---

## Uncertain Scripts (4 — Needs Manual Review)

| File | Reason | Question to Resolve |
|------|--------|---------------------|
| `eis_fitting.py` | Standalone eispac fitting demo with single date, no stale deps | Is this a useful reference/tutorial, or superseded by `eis_reader_v3.py`? |
| `eiscube_outlier_fixer.py` | Has reusable functions (`zero_masker`, `negative_masker`, `interpolator`) | Is this still used during dataset preparation, or was it a one-off fix? |
| `sample_size_analysis.py` | Theoretical margin-of-error analysis with hardcoded stats values | Is this a paper figure generator (keep) or ad-hoc analysis (delete)? |
| `test_interpolation_effect.py` | Well-structured with functions and `get_metrics` | Active analysis tool despite "test" label, or superseded? |
