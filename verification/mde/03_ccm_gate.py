"""CCM convergence gate per candidate (target FWD): reference pipeline vs
torchEDM (post-alignment).

Reference (dimx Run.py):
  E_c      = pyEDM EmbedDimension(columns=candidate, target=FWD, maxE=15) argmax
  gate 1   = E-sweep max rho >= embedDimRhoMin (0.65 here)
  libSizes = [10,15,85,90]% of full N=1061 -> [106,159,901,954]
  CCM      = pyEDM CCM(E=E_c, sample=20, seed): library sampled from all valid
             rows, prediction over all valid rows
  slope    = OLS of rho('FWD:candidate') on libSizes/N;  gate 2 = slope > 0.01

torchEDM (aligned): per-candidate E and peak from the same batched sweep
(matches reference E 80/80 and peaks to 4 decimals); gate 1 =
MinCandidateCorrelation; gate 2 = growth slope. Deliberately kept divergences:
the growth test samples from and measures on the training window, its sizes
are percentages of nTrain -> [29,44,254,269], and the slope is per fraction
of the training window.

Expected (torch 2.13 / pyEDM 2.5.7): E matches 80/80, peak diff 0.0000;
full-gate decision agreement 97.5% (21 vs 21 passes; residual = the kept
train/test-separation divergences flipping two marginal slopes, TS1 and TS24);
slope Pearson r = 0.79.

Writes ref_ccm_all80.pkl (reference E/slope/rho per candidate) for reuse by 04.
"""
import os
import pickle

import numpy as np
import torch

from common import load_fly, ts_columns, fly_split, FLY_FIT_KWARGS

SEED = 7777
PCTS = [10, 15, 85, 90]


def reference_side(df, ts_cols, out_pkl):
    from pyEDM import EmbedDimension, CCM
    from sklearn.linear_model import LinearRegression

    numericDF = df.drop(columns=['index'])
    N = df.shape[0]
    libSizes = [int(N * p / 100) for p in PCTS]
    x = (np.array(libSizes, dtype=float) / N).reshape(-1, 1)

    res = {}
    for c in ts_cols:
        edf = EmbedDimension(dataFrame=numericDF, columns=c, target='FWD',
                             maxE=15, lib=[1, 300], pred=[301, 600], Tp=1,
                             tau=-1, exclusionRadius=0, validLib=[], noTime=True,
                             numProcess=15, showPlot=False)
        iMax = edf['rho'].round(4).argmax()
        maxRhoE = float(edf['rho'].iloc[iMax].round(4))
        E = int(edf['E'].iloc[iMax])
        ccmDF = CCM(dataFrame=numericDF, columns=c, target='FWD',
                    libSizes=libSizes, sample=20, E=E, Tp=1, tau=-1,
                    exclusionRadius=0, seed=SEED, noTime=True)
        ccmVals = ccmDF[f'FWD:{c}'].to_numpy()
        slope = round(float(LinearRegression().fit(
            x, np.nan_to_num(ccmVals)).coef_[0]), 5)
        res[c] = dict(E=E, maxRhoE=maxRhoE, slope=slope, rhoVals=ccmVals)
        print(f'{c}: E={E} maxRhoE={maxRhoE:.4f} slope={slope:+.5f}', flush=True)

    with open(out_pkl, 'wb') as f:
        pickle.dump(dict(res=res, libSizes=libSizes), f)
    return res, libSizes


def main():
    from torchEDM.Fitters.MDEFitter import MDEFitter
    from scipy.stats import spearmanr

    df = load_fly()
    ts_cols = ts_columns(df)
    here = os.path.dirname(os.path.abspath(__file__))
    ref_pkl = os.path.join(here, 'ref_ccm_all80.pkl')
    if os.path.exists(ref_pkl):
        with open(ref_pkl, 'rb') as f:
            d = pickle.load(f)
        ref, ref_libsizes = d['res'], d['libSizes']
    else:
        ref, ref_libsizes = reference_side(df, ts_cols, ref_pkl)

    XTrain, YTrain, XTest, YTest = fly_split(df, ts_cols)
    fitter = MDEFitter(MaxD=1, Convergent='post', PredictionHorizon=1,
                       MinPredictionThreshold=0.2, MinCandidateCorrelation=0.65,
                       CCMLibraryPercentiles=np.array(PCTS),
                       CCMNumSamples=20, CCMConvergenceThreshold=0.01,
                       CCMSeed=SEED, CCMMaxEmbeddingDimensions=15,
                       dtype=torch.float64, progressBar=False)
    fitter.Fit(XTrain, YTrain, XTest, YTest, **FLY_FIT_KWARGS)
    mde = fitter.MDE

    E_match = sum(1 for i, c in enumerate(ts_cols)
                  if mde.candidateEmbedDimensions[0].get(i) == ref[c]['E'])
    peak_diff = max(abs(mde.candidatePeakCorrelations[0][i] - ref[c]['maxRhoE'])
                    for i, c in enumerate(ts_cols))
    print(f'E matches: {E_match}/80; max peak diff: {peak_diff:.4f}')

    ref_full, tor_full, ref_slopes, tor_slopes = [], [], [], []
    for i, c in enumerate(ts_cols):
        ok, slope = mde._check_single_candidate_convergence(i, mde.targets[0])
        peak_ok = mde.candidatePeakCorrelations[0][i] >= 0.65
        tor_full.append(bool(ok) and peak_ok)
        ref_full.append((ref[c]['slope'] > 0.01) and (ref[c]['maxRhoE'] >= 0.65))
        ref_slopes.append(ref[c]['slope'])
        tor_slopes.append(slope)
    ref_full, tor_full = np.array(ref_full), np.array(tor_full)
    print(f'full-gate decision agreement: {(ref_full == tor_full).mean():.2%} '
          f'(ref passes {ref_full.sum()}, torch passes {tor_full.sum()})')
    print(f'slope Pearson r = {np.corrcoef(ref_slopes, tor_slopes)[0, 1]:.3f} '
          f'Spearman = {spearmanr(ref_slopes, tor_slopes).statistic:.3f}')
    print('columns where full-gate decisions differ:')
    for i, c in enumerate(ts_cols):
        if ref_full[i] != tor_full[i]:
            print(f'  {c}: ref slope={ref_slopes[i]:+.4f} peak={ref[c]["maxRhoE"]:.3f} '
                  f'| torch slope={tor_slopes[i]:+.4f} '
                  f'peak={mde.candidatePeakCorrelations[0][i]:.3f}')


if __name__ == '__main__':
    main()
