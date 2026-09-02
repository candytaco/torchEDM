"""Lorenz5D full run (tau=-5, exclusionRadius=10 codepaths) plus a targeted
check of sample-mode CCM exclusionRadius handling.

Reference test params (test_mde_Lorenz5D) except:
  - Time dropped up front with noTime=True instead of removeTime=True: the
    reference's removeTime drops Time in Validate(), then PrepareNumericFrame
    (noTime=False) drops the new first column, so V1 silently leaves the
    candidate pool (the shipped Lorenz validation therefore stops at 3
    variables). noTime=True keeps all of V1..V4 as candidates.
  - pred trimmed to [501,999] and ccmSeed=7777 both sides (the reference test
    runs unseeded); firstEMax=False (torchEDM has no first-local-peak rule).

Expected: identical selection and rho on all 4 dims —
V3 0.398759, V4 0.806760, V2 0.946372, V1 0.976646 — with identical
per-candidate E (V1:5, V2:5, V3:4, V4:4); slope values differ under the
deliberately kept train/test-separation divergences but no decision flips.

The final block runs torchEDM sample-mode CCM with exclusionRadius 0 vs 10 on
identical draws: the self-match is always excluded, and radius=10 additionally
excludes all pairs within 10 rows, so accuracy drops below the radius=0 curve
(pre-fix, radius>0 disabled all exclusion and inflated accuracy instead).
"""
import numpy as np
import torch


def main():
    import dimx as dx
    from pyEDM import sampleData
    from torchEDM.Fitters.MDEFitter import MDEFitter
    from torchEDM.EDM.ConvergentCrossMap import ConvergentCrossMap

    data = sampleData['Lorenz5D'].drop(columns=['Time'])

    mde_ref = dx.MDE(data, noTime=True, removeColumns=['V5'], D=4,
                     target='V5', tau=-5, exclusionRadius=10,
                     lib=[1, 500], pred=[501, 999],
                     crossMapRhoMin=0.3, embedDimRhoMin=0.4,
                     firstEMax=False, ccmSeed=7777,
                     mpMethod='forkserver', consoleOut=False)
    mde_ref.Run()
    print('reference MDEOut:')
    print(mde_ref.MDEOut.to_string(index=False))
    print('reference E per candidate:', mde_ref._edimCache)
    print('reference CCM slopes:', mde_ref._ccmCache)

    cols = ['V1', 'V2', 'V3', 'V4']
    X = data[cols].values
    y = data['V5'].values
    fitter = MDEFitter(MaxD=4, Convergent='post', PredictionHorizon=1,
                       Step=-5, ExclusionRadius=10,
                       MinPredictionThreshold=0.3, MinCandidateCorrelation=0.4,
                       CCMLibraryPercentiles=np.array([10, 15, 85, 90]),
                       CCMNumSamples=20, CCMConvergenceThreshold=0.01,
                       CCMSeed=7777, CCMMaxEmbeddingDimensions=15,
                       dtype=torch.float64, progressBar=False)
    result = fitter.Fit(X[0:500], y[0:500], X[500:1000], y[500:1000],
                        TrainStart=0, TrainEnd=1, TestStart=0, TestEnd=1)
    sel = [cols[i] for i in result.selected_variables[0] if i >= 0]
    rho = [round(float(r), 6) for r in result.performance[0] if not np.isnan(r)]
    ccm = [round(float(r), 5) for r in result.ccm_values[0] if not np.isnan(r)]
    print(f'\ntorchEDM post: vars={sel} rho={rho} ccm_slopes={ccm}')
    print('torch E per candidate:',
          {cols[k]: v for k, v in fitter.MDE.candidateEmbedDimensions[0].items()})

    print('\nsample-mode CCM exclusionRadius handling (V5 manifold -> V1):')
    for radius in [0, 10]:
        ccm2 = ConvergentCrossMap(
            X=y, Y=data[['V1']].values, trainSizes=[50, 75, 425, 450],
            repeats=20, embedDimensions=5, predictionHorizon=1, step=-5,
            exclusionRadius=radius, trainIndices=[(0, 499)],
            testIndices=[(500, 998)], device='cpu', batchMode='sample',
            dtype=torch.float64, seed=7777, showProgress=False)
        r = np.asarray(ccm2.Run().forward_performance)
        print(f'  exclusionRadius={radius}: rho by libSize = {np.round(r, 4)}')


if __name__ == '__main__':
    main()
