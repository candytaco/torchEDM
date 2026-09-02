"""Full MDE run, parameters matched where the APIs allow.

Reference: the package's own Fly FWD validation config (D=8, lib=[1,300],
pred=[301,600], crossMapRhoMin=0.2, embedDimRhoMin=0.65, ccmSlope=0.01,
ccmSeed=7777) — reproduces MDE_Fly_2_Valid.csv.

torchEDM (post-alignment): MaxD=8, matched windows, MinPredictionThreshold=0.2
(~crossMapRhoMin), MinCandidateCorrelation=0.65 (~embedDimRhoMin), CCM
threshold/seed/samples/percentile grid matched. Run with Convergent='post'
and Convergent='pre'.

Expected: reference path TS33, TS4, TS8, TS9, TS32, TS24, TS26, TS71; torch
post and pre agree with each other and match the reference through dim 5,
splitting at dim 6 (TS24 vs TS23) where the deliberately kept
train/test-separation divergences flip TS24's marginal slope (see 03).
"""
import numpy as np
import torch

from common import load_fly, ts_columns, fly_split, FLY_FIT_KWARGS


def torch_run(df, ts_cols, convergent):
    from torchEDM.Fitters.MDEFitter import MDEFitter
    XTrain, YTrain, XTest, YTest = fly_split(df, ts_cols)
    fitter = MDEFitter(MaxD=8, Convergent=convergent, PredictionHorizon=1,
                       MinPredictionThreshold=0.2, MinCandidateCorrelation=0.65,
                       IterativeDimensionSearch=True,
                       CCMLibraryPercentiles=np.array([10, 15, 85, 90]),
                       CCMNumSamples=20, CCMConvergenceThreshold=0.01,
                       CCMSeed=7777, CCMMaxEmbeddingDimensions=15,
                       dtype=torch.float64, progressBar=False)
    result = fitter.Fit(XTrain, YTrain, XTest, YTest, **FLY_FIT_KWARGS)
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
