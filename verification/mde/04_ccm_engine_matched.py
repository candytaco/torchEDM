"""CCM engine equivalence under a MATCHED design.

Runs torchEDM's ConvergentCrossMap (sample mode — the engine behind
convergent='post') with the reference design imposed: full-data library pool
and prediction set, per-candidate E from the reference, libSizes
[106,159,901,954], repeats=20, slope on libSizes/N. Compares against the pyEDM
slopes stored by 03_ccm_gate.py, with library-sampling noise calibrated by
rerunning pyEDM CCM under seeds 1..5 on a subset of columns.

Windows are the 0-based half-open full span (0, N); rows without a valid
next-value target are trimmed by the engine.

Run 03_ccm_gate.py first to create ref_ccm_all80.pkl.
"""
import os
import pickle

import numpy as np
import torch

from common import load_fly, ts_columns

SEED = 7777
PCTS = [10, 15, 85, 90]


def main():
    from torchEDM.EDM.ConvergentCrossMap import ConvergentCrossMap
    from pyEDM import CCM
    from sklearn.linear_model import LinearRegression

    df = load_fly()
    ts_cols = ts_columns(df)
    numericDF = df.drop(columns=['index'])
    N = df.shape[0]
    libSizes = [int(N * p / 100) for p in PCTS]
    x = np.array(libSizes, dtype=float) / N

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, 'ref_ccm_all80.pkl'), 'rb') as f:
        ref = pickle.load(f)['res']

    fwd = df['FWD'].values
    tor_s = []
    for c in ts_cols:
        ccm = ConvergentCrossMap(
            X=fwd, Y=df[c].values[:, None], trainSizes=libSizes, repeats=20,
            embedDimensions=ref[c]['E'], predictionHorizon=1, step=-1,
            exclusionRadius=0, trainIndices=[(0, N)],
            testIndices=[(0, N)], device='cpu', batchMode='sample',
            dtype=torch.float64, seed=SEED, showProgress=False)
        rho = np.asarray(ccm.Run().forward_performance)
        tor_s.append(float(LinearRegression().fit(
            x.reshape(-1, 1), np.nan_to_num(rho)).coef_[0]))
        print(f'{c}: E={ref[c]["E"]} ref={ref[c]["slope"]:+.5f} '
              f'torch(matched)={tor_s[-1]:+.5f}', flush=True)

    print('\npyEDM slope spread across seeds 1..5 (sampling-noise calibration):')
    sds = []
    for c in ts_cols[::16]:
        slopes = []
        for seed in range(1, 6):
            ccmDF = CCM(dataFrame=numericDF, columns=c, target='FWD',
                        libSizes=libSizes, sample=20, E=ref[c]['E'], Tp=1,
                        tau=-1, exclusionRadius=0, seed=seed, noTime=True)
            v = ccmDF[f'FWD:{c}'].to_numpy()
            slopes.append(float(LinearRegression().fit(
                x.reshape(-1, 1), np.nan_to_num(v)).coef_[0]))
        sds.append(np.std(slopes))
        print(f'  {c}: mean={np.mean(slopes):+.5f} sd={np.std(slopes):.5f}')

    ref_s = np.array([ref[c]['slope'] for c in ts_cols])
    tor_s = np.array(tor_s)
    d = tor_s - ref_s
    sd_typ = float(np.mean(sds))
    print(f'\nmatched-design slope diff: max|d|={np.max(np.abs(d)):.5f} '
          f'rms={np.sqrt(np.mean(d ** 2)):.5f}; '
          f'typical pyEDM seed-to-seed sd={sd_typ:.5f}')
    print(f'columns with |d| > 3x sampling sd: '
          f'{[ts_cols[i] for i in np.where(np.abs(d) > 3 * sd_typ)[0]]}')
    print(f'slope Pearson r (matched design) = {np.corrcoef(ref_s, tor_s)[0, 1]:.4f}')
    ref_dec = ref_s > 0.01
    tor_dec = tor_s > 0.01
    print(f'decision agreement (slope > 0.01): {np.mean(ref_dec == tor_dec):.2%}')


if __name__ == '__main__':
    main()
