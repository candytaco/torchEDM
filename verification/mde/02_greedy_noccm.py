"""Greedy variable-selection path with CCM disabled: dimx noCCM vs
torchEDM Convergent=False, plus per-dimension candidate-score spectra.

crossMapRhoMin=-999 keeps every candidate in the reference's rhoD so the full
spectra can be compared. Windows as in 01_crossmap_sweep.py.

Expected output (torch 2.13 / pyEDM 2.5.7): identical 8-variable sequence
(TS33, TS4, TS31, TS56, TS17, TS68, TS48, TS76) with rho equal to 6 decimals;
spectra for dims 2..8 agree to ~5e-7 (reference stores rho as float32);
dim 1 has 4 candidates off by up to 3.9e-3, all tie cases (see 01).
"""
import numpy as np
import torch

from common import load_fly, ts_columns, fly_split, FLY_FIT_KWARGS

D = 8


def main():
    import dimx as dx
    from torchEDM.Fitters.MDEFitter import MDEFitter

    df = load_fly()
    ts_cols = ts_columns(df)

    mde_ref = dx.MDE(df, removeColumns=['index', 'FWD', 'Left_Right'],
                     D=D, target='FWD', lib=[1, 300], pred=[301, 600],
                     noCCM=True, crossMapRhoMin=-999,
                     mpMethod='forkserver', consoleOut=False)
    mde_ref.Run()
    ref_vars = list(mde_ref.MDEOut['variables'])
    ref_rho = list(mde_ref.MDEOut['rho'])

    XTrain, YTrain, XTest, YTest = fly_split(df, ts_cols)
    fitter = MDEFitter(MaxD=D, Convergent=False, PredictionHorizon=1,
                       MinPredictionThreshold=0.0, dtype=torch.float64,
                       progressBar=False)
    result = fitter.Fit(XTrain, YTrain, XTest, YTest, **FLY_FIT_KWARGS)
    torch_vars = [ts_cols[i] for i in result.selected_variables[0] if i >= 0]
    torch_rho = [r for r in result.performance[0] if not np.isnan(r)]

    print(f'{"dim":>3} {"ref":>7} {"rho":>9}   {"torch":>7} {"rho":>9}  match')
    for d in range(D):
        print(f'{d+1:>3} {ref_vars[d]:>7} {ref_rho[d]:>9.6f}   '
              f'{torch_vars[d]:>7} {torch_rho[d]:>9.6f}  '
              f'{ref_vars[d] == torch_vars[d]}')

    print('\nper-dimension candidate rho spectra (torch - reference):')
    stepwise = fitter.MDE.stepwise_performance
    for d in range(1, D + 1):
        spec = mde_ref.rhoD[d]['rho']
        diffs = np.array([stepwise[0, d - 1, ts_cols.index(c)] - r
                          for c, r in spec.items()])
        print(f'  dim {d}: n={len(diffs)}  max|diff|={np.max(np.abs(diffs)):.2e}  '
              f'n>|1e-3|={int(np.sum(np.abs(diffs) > 1e-3))}')


if __name__ == '__main__':
    main()
