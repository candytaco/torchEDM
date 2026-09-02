"""CCM convergence gate per candidate (target FWD): reference pipeline vs
torchEDM _check_single_candidate_convergence (convergent='post').

Reference (dimx Run.py):
  E_c      = pyEDM EmbedDimension(columns=candidate, target=FWD, maxE=15) argmax
  gate 1   = E-sweep max rho >= embedDimRhoMin
  libSizes = [10,15,85,90]% of full N=1061 -> [106,159,901,954]
  CCM      = pyEDM CCM(E=E_c, sample=20, seed): library sampled from all valid
             rows, prediction over all valid rows
  slope    = OLS of rho('FWD:candidate') on libSizes/N;  gate 2 = slope > ccmSlope

torchEDM 'post' path:
  E        = self-prediction E of the TARGET series (one E for every candidate)
  libSizes = [10,15,85,90]% of nTrain=299 -> [29,44,254,269]
  CCM      = sample mode: library sampled from train rows, prediction over train rows
  slope    = OLS of mean rho on libSizes/max(libSizes);  gate = slope > threshold
  (no gate 1 equivalent)

Optional stage (--e-attrib): rerun the torchEDM CCM with E forced to the
reference per-candidate value, keeping torch pool/axis, to attribute how much
disagreement the E choice alone causes.

Writes ref_ccm_all80.pkl (reference E/slope/rho per candidate) for reuse by 04.

Expected (torch 2.13 / pyEDM 2.5.7): slope Pearson r=0.18, Spearman 0.31;
decisions 62.5% vs slope-only gate, 52.5% vs full gate; passes 41 (torch) vs
59/21 (reference slope-only/full); torch E=4 for all candidates vs reference
per-candidate median 11. With --e-attrib: r=0.79, Spearman 0.84, decisions
86.25%.
"""
import os
import pickle
import sys

import numpy as np
import torch

from common import load_fly, ts_columns

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

    X = df[ts_cols].values
    y = df['FWD'].values
    fitter = MDEFitter(MaxD=1, Convergent=False, PredictionHorizon=1,
                       CCMLibraryPercentiles=np.array(PCTS),
                       CCMNumSamples=20, CCMConvergenceThreshold=0.01,
                       CCMSeed=SEED, CCMMaxEmbeddingDimensions=15,
                       dtype=torch.float64, progressBar=False)
    fitter.Fit(X[0:301], y[0:301], X[301:601], y[301:601],
               TrainStart=1, TestStart=0)
    mde = fitter.MDE

    tor = {}
    for i, c in enumerate(ts_cols):
        ok, slope = mde._check_single_candidate_convergence(i, mde.targets[0])
        tor[c] = dict(slope=slope, convergent=bool(ok))

    tor_libsizes = [int(p / 100 * mde.trainData.shape[0]) for p in PCTS]
    print(f'reference libSizes: {ref_libsizes} (basis: full N={df.shape[0]})')
    print(f'torchEDM  libSizes: {tor_libsizes} (basis: nTrain={mde.trainData.shape[0]})')

    ref_slope = np.array([ref[c]['slope'] for c in ts_cols])
    tor_slope = np.array([tor[c]['slope'] for c in ts_cols])
    ref_full = np.array([(ref[c]['slope'] > 0.01) and (ref[c]['maxRhoE'] >= 0.65)
                         for c in ts_cols])
    ref_so = ref_slope > 0.01
    tor_dec = np.array([tor[c]['convergent'] for c in ts_cols])

    print(f'slope Pearson r = {np.corrcoef(ref_slope, tor_slope)[0, 1]:.3f}  '
          f'Spearman = {spearmanr(ref_slope, tor_slope).statistic:.3f}')
    print(f'decision agreement, slope-only reference gate: {np.mean(ref_so == tor_dec):.2%}')
    print(f'decision agreement, full reference gate (+embedDimRhoMin=0.65): '
          f'{np.mean(ref_full == tor_dec):.2%}')
    print(f'passes/80: reference full gate {ref_full.sum()}, slope-only {ref_so.sum()}, '
          f'torchEDM {tor_dec.sum()}')

    from torchEDM.Hyperparameters import FindSelfPredictionEmbeddingDimension
    Etarget = FindSelfPredictionEmbeddingDimension(
        mde.data[:, [mde.targets[0]]], maxDims=15, train=mde.train,
        test=mde.test, predictionHorizon=1, step=-1, exclusionRadius=0,
        embedded=False, validLib=[], dtype=torch.float64, device=mde.device,
        batchSize=100, showProgress=False)
    ref_E = np.array([ref[c]['E'] for c in ts_cols])
    print(f'E: torchEDM target self-prediction (all candidates) = {Etarget[0]}; '
          f'reference per-candidate min={ref_E.min()} '
          f'median={np.median(ref_E):.0f} max={ref_E.max()}')

    if '--e-attrib' in sys.argv:
        from torchEDM.EDM.ConvergentCrossMap import ConvergentCrossMap
        x = np.array(tor_libsizes, dtype=float)
        x = x / x.max()
        var_s = []
        for c in ts_cols:
            ccm = ConvergentCrossMap(
                X=y, Y=df[c].values[:, None], trainSizes=tor_libsizes,
                repeats=20, embedDimensions=ref[c]['E'], predictionHorizon=1,
                step=-1, exclusionRadius=0, trainIndices=[(1, 300)],
                testIndices=[(301, 600)], device='cpu', batchMode='sample',
                dtype=torch.float64, seed=SEED, showProgress=False)
            rho = np.asarray(ccm.Run().forward_performance)
            xm, ym = x.mean(), rho.mean()
            var_s.append(float(((x * rho).mean() - xm * ym) /
                               ((x**2).mean() - xm**2)))
        var_s = np.array(var_s)
        var_dec = var_s > 0.01
        print(f'\nE forced to reference values (torch pool/axis unchanged): '
              f'slope Pearson r={np.corrcoef(ref_slope, var_s)[0,1]:.3f} '
              f'Spearman={spearmanr(ref_slope, var_s).statistic:.3f} '
              f'decision agreement={np.mean(ref_so == var_dec):.2%}')


if __name__ == '__main__':
    main()
