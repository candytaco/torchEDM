"""Full MDE run, parameters matched where the APIs allow.

Reference: the package's own Fly FWD validation config (D=8, lib=[1,300],
pred=[301,600], crossMapRhoMin=0.2, embedDimRhoMin=0.65, ccmSlope=0.01,
ccmSeed=7777) — reproduces MDE_Fly_2_Valid.csv.

torchEDM: MaxD=8, matched windows, MinPredictionThreshold=0.2
(~crossMapRhoMin), CCM threshold/seed/samples/percentile grid matched;
embedDimRhoMin has no torchEDM equivalent. Run with Convergent='post'
and Convergent='pre'.

Expected: agreement at dim 1 (TS33) only; reference continues TS4, TS8, TS9,
TS32, TS24, TS26, TS71; post picks TS5, TS32, TS72, TS71, TS73, TS61, TS57;
pre a third path. Dim-2 split = gate disagreement on TS4 (reference slope
+0.0349 pass, torch -0.0025 fail; see 03). Final rho comparable
(0.871 / 0.870 / 0.898).
"""
import numpy as np
import torch

from common import load_fly, ts_columns


def torch_run(df, ts_cols, convergent):
    from torchEDM.Fitters.MDEFitter import MDEFitter
    X = df[ts_cols].values
    y = df['FWD'].values
    fitter = MDEFitter(MaxD=8, Convergent=convergent, PredictionHorizon=1,
                       MinPredictionThreshold=0.2,
                       CCMLibraryPercentiles=np.array([10, 15, 85, 90]),
                       CCMNumSamples=20, CCMConvergenceThreshold=0.01,
                       CCMSeed=7777, CCMMaxEmbeddingDimensions=15,
                       dtype=torch.float64, progressBar=False)
    result = fitter.Fit(X[0:301], y[0:301], X[301:601], y[301:601],
                        TrainStart=1, TestStart=0)
    sel = [ts_cols[i] for i in result.selected_variables[0] if i >= 0]
    rho = [float(r) for r in result.performance[0] if not np.isnan(r)]
    return sel, rho


def main():
    import dimx as dx

    df = load_fly()
    ts_cols = ts_columns(df)

    mde_ref = dx.MDE(df, removeColumns=['index', 'FWD', 'Left_Right'],
                     D=8, target='FWD', lib=[1, 300], pred=[301, 600],
                     crossMapRhoMin=0.2, embedDimRhoMin=0.65, ccmSlope=0.01,
                     ccmSeed=7777, mpMethod='forkserver', consoleOut=False)
    mde_ref.Run()
    ref_vars = list(mde_ref.MDEOut['variables'])
    ref_rho = list(mde_ref.MDEOut['rho'])

    sel_post, rho_post = torch_run(df, ts_cols, 'post')
    sel_pre, rho_pre = torch_run(df, ts_cols, 'pre')

    print(f'{"dim":>3} | {"reference":>9} {"rho":>7} | {"torch post":>10} {"rho":>7} | {"torch pre":>9} {"rho":>7}')
    for d in range(max(len(ref_vars), len(sel_post), len(sel_pre))):
        f = lambda lst, i: lst[i] if i < len(lst) else '-'
        g = lambda lst, i: f'{lst[i]:.4f}' if i < len(lst) else '-'
        print(f'{d+1:>3} | {f(ref_vars,d):>9} {g(ref_rho,d):>7} | '
              f'{f(sel_post,d):>10} {g(rho_post,d):>7} | '
              f'{f(sel_pre,d):>9} {g(rho_pre,d):>7}')


if __name__ == '__main__':
    main()
