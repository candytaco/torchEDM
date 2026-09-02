"""Shared helpers for the MDE verification scripts.

Setup (once):
    pip install pyEDM torch scikit-learn scipy pandas
    git clone https://github.com/pao-unit/MDE <somewhere>/MDE
    pip install -e <somewhere>/MDE
    pip install -e <this repo>
    export MDE_REPO=<somewhere>/MDE

Window equivalence used throughout (torchEDM windows are 0-based, half-open
[start, stop) pairs; rows whose target index is out of bounds are trimmed):
reference lib=[a,b], pred=[c,d] with horizon 1 is reproduced by
  - class API: train=[(a-1, b-1)], test=[(c-1, d)]
  - sklearn API: XTrain = rows a-1..c-2, XTest = rows c-1..d, and
    Fit(..., TrainStart=0, TrainEnd=1, TestStart=0, TestEnd=1)
For the Fly runs (lib=[1,300], pred=[301,600]) both give train rows 0..298
and test rows 300..599, matching the reference exactly (verified in 01).
"""
import os

import pandas as pd

MDE_REPO = os.environ.get('MDE_REPO', os.path.expanduser('~/MDE'))


def load_fly():
    """Fly80XY_norm_1061.csv from the reference repo: 1061 x 83
    (index, TS1..TS80, Left_Right, FWD)."""
    path = os.path.join(MDE_REPO, 'dimx', 'data', 'Fly80XY_norm_1061.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Clone https://github.com/pao-unit/MDE and '
            'set MDE_REPO to the clone path.')
    return pd.read_csv(path)


def ts_columns(df):
    return [c for c in df.columns if c.startswith('TS')]


def fly_split(df, ts_cols):
    """XTrain/YTrain/XTest/YTest reproducing reference lib=[1,300],
    pred=[301,600] when passed with TrainStart=0, TrainEnd=1, TestStart=0,
    TestEnd=1."""
    X = df[ts_cols].values
    y = df['FWD'].values
    return X[0:300], y[0:300], X[300:601], y[300:601]


FLY_FIT_KWARGS = dict(TrainStart=0, TrainEnd=1, TestStart=0, TestEnd=1)
