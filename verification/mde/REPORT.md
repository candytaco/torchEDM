# MDE: functional divergences from the reference implementation

Compared: torchEDM MDE through the sklearn-like `MDEFitter` API vs
[pao-unit/MDE](https://github.com/pao-unit/MDE) (`dimx`, pyEDM 2.5.7; torch
2.13 CPU, float64). Data: the reference package's own test data
(`Fly80XY_norm_1061.csv`, target FWD, lib=[1,300]/pred=[301,600]; pyEDM
`Lorenz5D`). The reference reproduces its shipped validation CSVs here.
Parameters matched wherever both APIs expose them; each finding names the
script in this directory that replicates it. Accepted differences
(neighbor-tie resolution, RNG streams) are identified as such and excluded
from the divergence list.

## Window mapping (used by every comparison)

torchEDM windows are 0-based, half-open `[start, stop)` pairs — flat lists
and strings are rejected. Reference `lib=[a,b], pred=[c,d]` (1-offset
inclusive; the horizon trims the last library row) is reproduced by the class
API as `train=[(a-1, b-1)], test=[(c-1, d)]`, and through `MDEFitter.Fit` by
splitting the series at `c-1`: `XTrain = rows a-1..c-2`, `XTest = rows
c-1..d`, `Fit(..., TrainEnd=1, TestEnd=1)`. Verified: both sides use train
rows 0..298, test rows 300..599 on the Fly windows. [01]

## Components verified equivalent

- **Simplex cross-map core** (candidate scoring). 80-candidate dimension-1
  sweep: max |Δrho| = 3.9e-3, nonzero only for 4 candidates that each have one
  test row with an exact distance tie at the knn boundary. Recomputing with
  pyEDM's tie order (distance, |predRow−libRow|, libRow) reproduces pyEDM
  exactly; plain argsort order reproduces torchEDM exactly. Same distance
  metric (Euclidean), weights exp(−d/d_min), knn = |columns|+1, Tp target
  alignment. Residual = accepted tie handling. [01]
- **Greedy selection, CCM disabled** (`noCCM=True` vs `Convergent=False`):
  identical 8-variable sequence (TS33→TS76) and rho to 6 decimals;
  per-dimension candidate spectra for dims 2–8 agree to 5e-7 (reference stores
  float32). [02]
- **CCM engine under a matched design**: imposing the reference design on
  torchEDM's `ConvergentCrossMap` (full-data pool, per-candidate E, libSizes
  [106,159,901,954], slope on L/N) gives slope Pearson r = 0.998 over 80
  candidates, rms difference 0.0051 vs pyEDM's own seed-to-seed slope sd
  0.0036 (max |d| = 0.0137 ≈ 3.8 sd; 6 of 80 columns exceed 3 sd, consistent
  with the differing RNG streams). The CCM computation itself is equivalent;
  the gate divergence comes from the surrounding design, itemized below. [04]
- **Lorenz5D full run** (tau=−5, exclusionRadius=10, seed matched): identical
  selection and rho at every dimension (V3 0.398759, V4 0.806760, V2 0.946372,
  V1 0.976646) — slope values differ (V1: +0.084 vs +0.139) but no decision
  flips. [06]

## Divergences in the CCM convergence gate

Net effect on Fly/FWD (80 candidates, every shared parameter matched):
slope Pearson r = 0.18, Spearman = 0.31; decision agreement 62.5% against the
reference's slope gate, 52.5% against its full gate; passes: torchEDM 41/80,
reference 59/80 (slope gate) and 21/80 (full gate). [03]

1. **Embedding dimension.** Reference: per-candidate E = argmax-rho of pyEDM
   `EmbedDimension` (candidate embedding cross-predicting the target, E in
   1..maxE, optional `firstEMax` first-local-peak rule; Run.py:222-252).
   torchEDM: one E for all candidates = self-prediction optimum of the target
   series (`ConvergentCrossMap.py:141-157`, `Hyperparameters.py:248`); no
   `firstEMax` equivalent. Measured: E=4 for every candidate vs reference
   per-candidate median 11 (range 1–15). Forcing the reference E values while
   keeping the rest of torchEDM's design moves slope agreement to r = 0.79
   (Spearman 0.84) and decisions to 86.25% — the E criterion is the dominant
   factor. [03 --e-attrib]
2. **`embedDimRhoMin` gate has no torchEDM equivalent.** The reference skips a
   candidate whose E-sweep peak rho is below threshold before running CCM
   (Run.py:262-264); at the Fly test's 0.65 this alone cuts reference passes
   from 59 to 21 of 80. [03]
3. **Library-size basis.** Reference: `pLibSizes` percent of full data length
   (dimx MDE.py:386-394; [106,159,901,954] at N=1061). torchEDM: percent of
   the train window (`MDE.py:506,567`; [29,44,254,269] at nTrain=299).
4. **Sampling pool and prediction set.** Reference pyEDM CCM ignores lib/pred:
   libraries are drawn from all valid rows, rho computed over all valid rows.
   torchEDM `convergent='post'` (sample mode): both restricted to the train
   window (`ConvergentCrossMap.py:341,397-402`); `convergent='pre'`
   (variables mode): library from train, rho over the test window
   (`ConvergentCrossMap.py:290-296,316`). The two torchEDM modes also differ
   from each other, and selected different variables on Fly/FWD (see 9).
5. **Slope normalization.** Reference regresses rho on libSizes/N
   (Run.py:31-32); torchEDM on libSizes/max(libSizes) (`MDE.py:511-512,
   574-575`). The same 0.01 threshold therefore cuts at different effective
   convergence rates.
6. **exclusionRadius in sample-mode CCM.** `exclusionRadius==0` masks the
   self-match (diagonal); any radius > 0 applies no exclusion at all, so the
   self-match re-enters the neighbor set (`ConvergentCrossMap.py:381-383`).
   pyEDM CCM always removes the self-match and additionally excludes
   |t_i−t_j| ≤ radius. Measured (Lorenz, V5→V1, identical draws): rho by
   libSize [0.854, 0.883, 0.969, 0.970] at radius=0 vs
   [0.870, 0.901, 0.996, 0.998] at radius=10. [06]
7. **No caching.** Reference caches E and slope per column per run, freezing
   each candidate's verdict (Run.py:70-71,215-216,271-272); torchEDM re-runs
   the CCM check every dimension. Identical outcome when `CCMSeed` is set;
   unseeded, a rejected candidate is re-tried each dimension under fresh
   draws, which the reference's caching precludes.

## Divergences in the selection loop

8. **Termination.** Reference stops expansion at the first dimension where no
   candidate passes `crossMapRhoMin` or the CCM gate (Run.py:135-145,
   315-320). torchEDM selects nothing for that target and continues iterating
   to `MaxD` (`MDE.py:281-365` has no break), re-running every candidate's CCM
   check each remaining dimension: same selection when seeded, plus the
   unseeded re-try effect of (7).
9. **Full-run outcome, Fly/FWD** (matched parameters;
   `MinPredictionThreshold=0.2` for `crossMapRhoMin=0.2`; nothing available
   for `embedDimRhoMin=0.65`): agreement at dim 1 only, as the gate
   divergences (1–5) compound through the greedy path.

   | dim | reference | rho | post | rho | pre | rho |
   |---|---|---|---|---|---|---|
   | 1 | TS33 | 0.6528 | TS33 | 0.6528 | TS33 | 0.6528 |
   | 2 | TS4 | 0.7923 | TS5 | 0.7699 | TS21 | 0.7734 |
   | 3 | TS8 | 0.8190 | TS32 | 0.8269 | TS9 | 0.8204 |
   | 4 | TS9 | 0.8394 | TS72 | 0.8402 | TS69 | 0.8454 |
   | 5 | TS32 | 0.8591 | TS71 | 0.8511 | TS30 | 0.8595 |
   | 6 | TS24 | 0.8602 | TS73 | 0.8620 | TS17 | 0.8742 |
   | 7 | TS26 | 0.8690 | TS61 | 0.8631 | TS12 | 0.8874 |
   | 8 | TS71 | 0.8711 | TS57 | 0.8699 | TS48 | 0.8975 |

   The dim-2 split is the gate disagreement on TS4: reference slope +0.0349
   (pass), torchEDM −0.0025 (fail). Final-dimension rho is comparable
   (0.871 / 0.870 / 0.898). [05, 03]

## API-boundary semantics (sklearn adapter)

10. **Last data row unreachable as a test point.** The adapter emits the test
    pair end as `len(data)−1`, which the 1-offset core converts to row
    `len(data)−2` (`DataAdapter.py:222-224`, `EDM.py:827`). Reference
    `pred=[c,N]` therefore cannot be expressed; comparisons trim the reference
    window instead. Related: `ConvergentCrossMap` indexes `Y[test+Tp]` without
    trimming, so a test window touching the last row raises IndexError
    (`ConvergentCrossMap.py:166`, `utils.py:36-47`) where pyEDM drops such
    rows; its train window uses `min(stop, N−Tp)` where `Simplex` uses
    `stop−Tp`, a one-row pool difference when the train window ends mid-data.
11. **`TrainStart=0` produces row index −1** (wraps to the last row):
    `CreateIndices` subtracts 1 from the adapter's already-0-offset start
    (`EDM.py:774`); the guard at `EDM.py:748-749` assigns a dead local.
    The default `TrainStart=1` avoids it.

## Scoring edge cases

12. **NaN.** torchEDM `Correlation` has no NaN handling (`_core.py:92-111`):
    a NaN prediction yields a NaN score and the candidate sorts last, and NaN
    embedding rows are dropped once from the shared all-column embedding
    (`MDE.py:220-227`). Reference drops NaN pairs inside `ComputeError` and
    NaN rows per candidate combination. Identical on NaN-free data (both test
    sets); differs when data contain NaN.
13. **Low-variance filter exists only in torchEDM** (`MDE.py:239-240`,
    std < `stdThreshold` excluded from the pool); not binding on the test data
    (min column std 0.046 > default 1e-2).

## Parameter mapping and defaults that differ functionally

| reference (dimx) | MDEFitter | note |
|---|---|---|
| `D` | `MaxD` | — |
| `Tp` 1 | `PredictionHorizon` 1 | equal defaults |
| `tau` −1 | `Step` −1 | equal defaults |
| `crossMapRhoMin` 0.5 | `MinPredictionThreshold` 0.0 | torch default disables the filter; no termination either way (see 8) |
| `embedDimRhoMin` 0.5 | — | no equivalent (see 2) |
| `pLibSizes` [10,15,85,90] | `CCMLibraryPercentiles` linspace(10,90,5) | grids differ; basis differs regardless (see 3) |
| `sample` 20 | `CCMNumSamples` 10 | — |
| `ccmSlope` 0.01 | `CCMConvergenceThreshold` 0.01 | same value, different slope scale (see 5) |
| `ccmSeed` None | `CCMSeed` None | — |
| `maxE` 15 | `CCMMaxEmbeddingDimensions` 15 | different E criterion (see 1) |
| `firstEMax` | — | no equivalent |
| `noCCM=True` | `Convergent=False` | equivalent (verified, [02]) |
| lazy per-dimension gate | `Convergent` 'pre'/'post' | fitter default 'pre'; 'pre' and 'post' use different CCM prediction sets (see 4) |
| — | `stdThreshold` 1e-2 | torch-only filter (see 13); MDE-class default is 1e-3 |

## Decisions

Each divergence above was reviewed and resolved as follows.

**Align with the reference:**
- (1) Embedding dimension: per-candidate E, tuned on candidate→target
  prediction and consumed for the target→candidate reconstruction — the
  tune/consume pairing copied as deliberate reference design (obfuscated by
  its CCM API taking a single E). Implemented as one upfront batched sweep
  (`FindOptimalEmbeddingDimensionality(candidates, target, joint=False)`),
  argmax per candidate on raw scores (torchEDM does not round; the
  reference's 4-decimal tie-rounding is not copied). Two search modes: the
  default shares the most restrictive (maxDims) row set across all depths in
  one fully batched pass — a deliberate torch-parallel divergence that
  changes some chosen dimensions; `IterativeDimensionSearch=True` gives each
  depth its own valid rows, reproduces the reference, and is what these
  verification scripts use.
- (2) Solo-predictability pre-screen: adopted, as an upfront pass over all
  candidates (cheap under batching) — new threshold parameter mirroring the
  reference's `embedDimRhoMin`, fed by the same sweep's per-candidate peak.
- (6) Sample-mode exclusion: fixed — self-match always masked; radius > 0
  additionally masks |t_i−t_j| ≤ radius.
- (7) Growth-test verdicts cached per candidate per run.
- (8) Expansion terminates (per target) on a barren round.

**Deliberately diverge (train/test separation):**
- (3) Growth-test sample sizes remain percentages of the training window,
  not of the full recording.
- (4) The screen samples from and measures on the training window only;
  `'pre'` mode is fixed to match `'post'` (it measured on the test window).
  The reference's screen — which draws on and scores against all rows,
  including held-out data — is rejected.
- (5) The growth slope is regressed on sample size ÷ training-window length
  (grid-invariant units), not the reference's ÷ recording length.
- (12) NaN handling stays: upfront any-column row filtering; NaN score
  disqualifies a candidate.
- (13) The low-variance candidate filter stays (no reference counterpart);
  default unified to 1e-3 across the wrapper and the class.

**torchEDM-internal (not reference alignment):**
- (10, 11) 0-based indexing enforced end-to-end (the engine's 1-offset
  convention, inherited from pyEDM's mixed 0/1 indexing, is removed rather
  than compensated for). Every row passed is usable except rows whose
  target index (t+Tp) or history stack would be out of bounds — those are
  trimmed, never crash. Consequences accepted: `TrainStart=0` means row 0;
  `TestStart=0` means the first prediction input is XTest row 0 (the first
  Tp test targets are unpredicted unless overlap is passed explicitly); the
  training-window end uses the bounds-only clamp, keeping one row the
  reference's unconditional Tp-subtraction drops mid-data.
- (9) carries no decision — it is the measured compound effect, re-measured
  after implementation (see the re-verification section if present).

## Re-verification after implementation

The decisions above are implemented (commits following the ledger commit).
Re-running the scripts against the reference:

- Windows are now 0-based half-open `[start, stop)` pairs with bounds-only
  trimming; the
  mapping in `common.py` reproduces the reference windows exactly, and the
  simplex core [01] and CCM-off greedy path [02] results are unchanged.
- Candidate E search [03]: torchEDM matches the reference's per-candidate E
  80/80, peaks equal at 4 decimals; the solo-predictability gate
  (`MinCandidateCorrelation=0.65`) agrees 80/80 at the reference threshold.
- Full convergence gate [03]: decision agreement 52.5% → 97.5% (21 vs 21
  passes); slope Pearson r 0.18 → 0.79. The residual is two candidates
  (TS1, TS24) whose marginal slopes flip across the 0.01 threshold under
  the deliberately kept train/test-separation divergences (3–5).
- Full run [05]: selection matches the reference through dim 5 (TS33, TS4,
  TS8, TS9, TS32), splitting at dim 6 on TS24 (the marginal-slope case);
  `'pre'` and `'post'` now produce identical selections.
- Lorenz5D [06]: identical selection and rho at all 4 dims, identical
  per-candidate E; the fixed sample-mode exclusion makes radius=10 accuracy
  drop below radius=0 (pre-fix it rose above, from self-match leakage).
- torchEDM's own test suite: the 6 tests affected by the indexing change
  pass after window translation; the 5 embed-dimension tests pass at their
  original 1e-6 tolerance (all 11 crashed or failed beforehand); the
  remaining failures predate this work (value drift in the legacy
  simplex/smap/ccm tests and a CUDA-only test).

## Comparison-methodology notes

- Reference quirk avoided in [06]: `removeTime=True` drops the Time column in
  `Validate()`, then `PrepareNumericFrame` (noTime=False) drops the new first
  column, silently removing the first variable from the candidate pool — the
  shipped Lorenz validation stops at 3 of 4 selectable variables because of
  it. Passing the frame Time-less with `noTime=True` keeps all candidates.
- Accepted differences observed, excluded from the list above: neighbor-tie
  resolution (torch.topk vs pyEDM's deterministic lexsort; bounded at one
  swapped neighbor per exact tie [01]) and CCM library-sampling RNG stream
  identity at equal seeds (bounded by the seed-to-seed spread measured in
  [04]).
