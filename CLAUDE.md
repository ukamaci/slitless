# Slitless Codebase Guidelines

## Never delete working code without explicit confirmation
If introducing new functionality alongside existing code, make the new option selectable (e.g. a config flag) rather than replacing the old code. Only remove or replace existing code when the user has explicitly confirmed it is no longer needed.

## Single source of truth
Configuration values (dataset name, paths, hyperparameters) should be defined once and everything else derived from them. Avoid repeating the same literal value in multiple places — if it needs to change, it should change in exactly one location. For example, derive paths, run names, and transform stats from a single `DSET` variable rather than hardcoding each separately.

## Reuse before creating
Before writing a new function, class, or script, search the existing codebase for something that already does it or can be adapted with a small change. Prefer modifying an existing script over creating a new one. Only introduce new code when the existing options are genuinely insufficient.

## Script structure
**Imports at the very top.** All `import` statements go before everything else — before the config block, before constants. Never place imports in the middle of a script.

**Config at top, no argparse.** All parameters live in a clearly marked config block at the top of the script, immediately after imports. Never use `argparse` or `sys.argv`. The script itself is the record of what was run.

## Key existing entry points
- `python/slitless/forward.py` — all forward models (Gaussian, tomo, torch variants)
- `python/slitless/recon.py` — all solvers and reconstructor classes
- `python/slitless/data_loader.py` — dataset classes, normalization transforms (`param_transform`, `meas_transform`, `param_inv_transform`)
- `python/slitless/train.py` — supervised U-Net training loop
- `python/scripts/generate_testbed_set.py` — consolidated testbed set generation; supports dset_v5 and dset_v6 (set `DSET_VERSION` in config block)
- `python/scripts/generate_dset_v6.py` — dset_v6 generation (filtered, measurements included)
- `python/scripts/final_result_runner.py` — evaluation across all method/config combinations
