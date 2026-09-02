"""Dimension-1 cross-map sweep: per-candidate simplex rho, torchEDM vs pyEDM.

Reference side replicates dimx's SimplexWorker (pyEDM Simplex + ComputeError)
for every candidate column. torchEDM side is the sklearn-like MDEFitter;
per-candidate scores read from MDE.stepwise_performance.

Second part proves the residual differences are neighbor-tie resolution:
recomputing simplex manually with pyEDM's tie order (distance, |predRow-libRow|,
libRow) reproduces pyEDM exactly, and with plain stable argsort reproduces
torchEDM exactly; only rows with an exact distance tie at the knn boundary
differ.

Expected output (torch 2.13 / pyEDM 2.5.7):
  windows identical (train 0..298, test 300..599)
  max |diff| ~ 3.9e-03, confined to candidates with 1 tied test row
  (TS35, TS11, TS77, TS31); tie-order recomputation matches both sides exactly.
"""
import numpy as np
import torch

from common import load_fly, ts_columns


def manual_simplex_rho(x, y, tr, te, tiebreak, Tp=1, knn=2):
    """Univariate embedded simplex, pyEDM weight formula, selectable tie order."""
    D = np.abs(x[tr][:, None] - x[te][None, :])
    preds = np.empty(len(te))
    for j in range(len(te)):
        d = D[:, j]
        if tiebreak == 'pyEDM':
            order = np.lexsort((tr, np.abs(te[j] - tr), d))
        else:
            order = np.argsort(d, kind='stable')
        nn = order[:knn]
        dn = d[nn]
        w = np.exp(-dn / max(dn[0], 1e-6))
        preds[j] = np.sum(w * y[tr[nn] + Tp]) / np.sum(w)
    return float(np.corrcoef(y[te + Tp], preds)[0, 1]), preds


def main():
    from pyEDM import Simplex, ComputeError
    from torchEDM.Fitters.MDEFitter import MDEFitter

    df = load_fly()
    numericDF = df.drop(columns=['index'])
    ts_cols = ts_columns(df)

    ref_rho = {}
    for c in ts_cols:
        sdf = Simplex(dataFrame=numericDF, columns=[c], target='FWD',
                      lib=[1, 300], pred=[301, 600], E=0, Tp=1, tau=-1,
                      embedded=True, exclusionRadius=0, noTime=True, kdWorkers=1)
        ref_rho[c] = ComputeError(sdf['Observations'], sdf['Predictions'])['rho']

    X = df[ts_cols].values
    y = df['FWD'].values
    fitter = MDEFitter(MaxD=1, Convergent=False, PredictionHorizon=1,
                       MinPredictionThreshold=0.0, dtype=torch.float64,
                       progressBar=False)
    fitter.Fit(X[0:301], y[0:301], X[301:601], y[301:601],
               TrainStart=1, TestStart=0)
    mde = fitter.MDE
    tr, te = mde._selectionTrainIndices, mde._selectionTestIndices
    print(f'torchEDM train rows: {tr[0]}..{tr[-1]} (n={len(tr)}) | '
          f'test rows: {te[0]}..{te[-1]} (n={len(te)})')
    print('reference:           train 0..298 (n=299) | test 300..599 (n=300)')

    torch_rho = {c: mde.stepwise_performance[0, 0, i] for i, c in enumerate(ts_cols)}
    diffs = np.array([torch_rho[c] - ref_rho[c] for c in ts_cols])
    print(f'\nmax |diff| = {np.max(np.abs(diffs)):.2e}   '
          f'mean |diff| = {np.mean(np.abs(diffs)):.2e}')

    print('\ntie-resolution attribution for the 4 largest differences:')
    for i in np.argsort(-np.abs(diffs))[:4]:
        c = ts_cols[i]
        r_py, p_py = manual_simplex_rho(df[c].values, y, tr, te, 'pyEDM')
        r_tk, p_tk = manual_simplex_rho(df[c].values, y, tr, te, 'argsort')
        nrows = int(np.sum(~np.isclose(p_py, p_tk)))
        print(f'  {c}: pyEDM={ref_rho[c]:.6f} manual(pyEDM ties)={r_py:.6f} | '
              f'torch={torch_rho[c]:.6f} manual(argsort ties)={r_tk:.6f} | '
              f'tied rows={nrows}')


if __name__ == '__main__':
    main()
