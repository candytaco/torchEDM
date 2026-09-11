"""Shared helpers for the MDE verification scripts.

Setup (once):
    pip install pyEDM torch scikit-learn scipy pandas
    git clone https://github.com/pao-unit/MDE <somewhere>/MDE
    pip install -e <somewhere>/MDE
    pip install -e <this repo>
    export MDE_REPO=<somewhere>/MDE

Array equivalence used throughout. torchEDM takes X_train/Y_train/X_test/Y_test
arrays; a training state is any row whose history is complete and whose
horizon-shifted target lies inside the array, and Y_pred has Y_test's shape
with NaN where no complete state predicts the row. A reference training window [a,b]
and test window [c,d] (1-offset inclusive) with horizon 1 are reproduced by
  X_train = rows a-1..b-1 (training states a-1..b-2, targets a..b-1) and
  X_test = rows c-1-h..d where h is the longest history span needed, with the
  Y_test entries before row c set to NaN so those rows are predicted but never
  scored (scored states c-1..d-1, targets c..d).
For the Fly runs (training window [1,300], test window [301,600]) this gives training states
0..298 and scored test states 300..599, matching the reference (verified in 01).
"""
import os

import numpy as np

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


def fly_split(df, ts_cols, historySpan=14):
    """X_train/Y_train/X_test/Y_test reproducing the reference training window [1,300]
    and test window [301,600]. The test arrays start historySpan rows early so the longest
    history (15 samples) is complete for the reference's first test state (row
    300); the targets of those early rows are NaN so they are predicted but
    never scored. Scored states 300..599, targets 301..600."""
    X = df[ts_cols].values
    y = df['FWD'].values.astype(float)
    testStart = 300 - historySpan
    Y_test = y[testStart:601].copy()
    Y_test[:historySpan + 1] = np.nan
    return X[0:300], y[0:300], X[testStart:601], Y_test
