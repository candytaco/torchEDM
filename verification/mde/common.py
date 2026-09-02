"""Shared helpers for the MDE verification scripts.

Setup (once):
    pip install pyEDM torch scikit-learn scipy pandas
    git clone https://github.com/pao-unit/MDE <somewhere>/MDE
    pip install -e <somewhere>/MDE
    export MDE_REPO=<somewhere>/MDE

Window equivalence used throughout (established in 01_crossmap_sweep.py):
reference lib=[1,300], pred=[301,600] on an N-row frame is reproduced through
the sklearn-like API by splitting the SAME series as XTrain=rows 0..300
(301 rows), XTest=rows 301..600 (300 rows) and calling
Fit(..., TrainStart=1, TestStart=0). Both then use train rows 0..298
(Tp=1 trims the last library row) and test rows 300..599.
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
