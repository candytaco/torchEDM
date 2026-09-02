# MDE verification against the reference package

Scripts replicating the findings in [`REPORT.md`](REPORT.md): torchEDM's MDE
(via the sklearn-like `MDEFitter` API) compared against the reference
implementation [pao-unit/MDE](https://github.com/pao-unit/MDE) (`dimx`,
pyEDM-based).

## Setup

```bash
pip install torch pyEDM scikit-learn scipy pandas tqdm pydiffmap
git clone https://github.com/pao-unit/MDE ~/MDE
pip install -e ~/MDE
pip install -e <this repo>
export MDE_REPO=~/MDE
```

Verified with torch 2.13.0 (CPU), pyEDM 2.5.7, numpy 2.4, Python 3.11.

## Scripts

| script | what it compares | runtime* |
|---|---|---|
| `01_crossmap_sweep.py` | dimension-1 candidate sweep (simplex core), tie attribution | ~1 min |
| `02_greedy_noccm.py` | greedy selection path, CCM disabled, per-dim spectra | ~2 min |
| `03_ccm_gate.py [--e-attrib]` | CCM convergence gate per candidate (E, slopes, decisions) | ~25 min |
| `04_ccm_engine_matched.py` | CCM engine under matched design (needs 03's pickle) | ~10 min |
| `05_full_run_fly_fwd.py` | full MDE run, matched parameters, post + pre modes | ~10 min |
| `06_lorenz_exclusion.py` | Lorenz5D (tau=-5, exclusionRadius=10), sample-mode radius demo | ~5 min |

*on 4 CPU cores. Each script prints its expected outcome in the docstring.
Run 03 before 04 (04 reuses `ref_ccm_all80.pkl`).
