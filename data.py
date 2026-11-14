"""Synthetic imbalanced benchmark with known ground truth + generic CSV loader for the real datasets."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_NAMES_MC = ["Benign", "Smurf", "Nmap", "ARP", "DoS"]
# ECU-IoHT counts scaled by ~1/6.5 (original: 23453 / 77920 / 6836 / 2359 / 639)
COUNTS_MC = [3600, 12000, 1050, 363, 98]


def make_synthetic(counts, d=40, binary=False, seed=0):
    """Ground-truth structure (indices):
       f0,f1  Smurf-informative         f3,f4  Nmap-informative        f5,f6,f7 ARP-informative (weak)
       f2     weak Benign shift         f10    weak DoS shift          f11 weak ARP shift
       f8,f9  DoS XOR-synergy pair (identical marginals for all classes)
       f12-19 noisy redundant copies    f20-39 correlated noise features
    """
    rng = np.random.RandomState(seed)
    ys = np.concatenate([np.full(n, c) for c, n in enumerate(counts)])
    N = len(ys)
    X = rng.randn(N, d)
    sh = lambda c, f, v: X.__setitem__((ys == c, f), X[ys == c, f] + v)
    sh(1, 0, 2.0); sh(1, 1, 1.6)
    sh(2, 3, 1.8); sh(2, 4, -1.5)
    for f in (5, 6, 7): sh(3, f, 1.1)
    sh(0, 2, 0.8); sh(4, 10, 1.2); sh(3, 11, 0.9)
    s = 1.8
    a = rng.choice([-1, 1], N)
    X[:, 8] = a * s + 0.6 * rng.randn(N)
    X[:, 9] = np.where(ys == 4, a, -a) * s + 0.6 * rng.randn(N)
    src = [0, 1, 3, 4, 5, 6, 10, 2]
    for i, f in enumerate(range(12, 20)):
        X[:, f] = 0.9 * X[:, src[i]] + 0.35 * rng.randn(N)
    lat = rng.randn(N, 5)
    for i, f in enumerate(range(20, 40)):
        X[:, f] = 0.7 * lat[:, i % 5] + 0.7 * rng.randn(N)
    for f in (0, 3, 12, 22, 31):
        X[:, f] = np.exp(0.5 * X[:, f])
    if binary:
        ys = (ys > 0).astype(int)
    info = dict(informative=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10], weak=[2, 11], redundant=list(range(12, 20)),
                noise=list(range(20, 40)), synergy=[8, 9])
    perm = rng.permutation(N)
    return X[perm].astype(np.float32), ys[perm].astype(np.int64), info


def split_standardise(X, y, seed, val=0.15, test=0.15):
    """Stratified 70/15/15 split; skew-aware log transform + z-score fitted on training data only."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test, stratify=y, random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=val / (1 - test), stratify=ytr, random_state=seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    f = lambda A: ((A - mu) / sd).astype(np.float32)
    return f(Xtr), ytr, f(Xva), yva, f(Xte), yte


def frame_to_xy(df, label_col, drop_cols=(), positive_if=None, max_onehot=10):
    """DataFrame -> (X, y, column names). Duplicate rows removed, constant/identifier columns should be passed in drop_cols."""
    df = df.drop_duplicates()
    y = df[label_col]
    y = positive_if(y).astype(int).values if positive_if else pd.factorize(y, sort=True)[0]
    X = df.drop(columns=[label_col, *[c for c in drop_cols if c in df.columns]])
    for c in X.select_dtypes(exclude=[np.number]).columns:
        if X[c].nunique() <= max_onehot:
            X = pd.concat([X.drop(columns=c), pd.get_dummies(X[c].astype(str), prefix=c, dtype=float)], axis=1)
        else:
            X[c] = pd.factorize(X[c])[0]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X.loc[:, X.nunique() > 1]          # constant columns carry no information
    for c in X.columns:                    # log1p for heavy-tailed non-negative attributes
        if X[c].min() >= 0 and X[c].skew() > 5:
            X[c] = np.log1p(X[c])
    return X.values.astype(np.float32), np.asarray(y).astype(np.int64), list(X.columns)


def load_csv(path, label_col, drop_cols=(), positive_if=None, max_onehot=10):
    """Generic loader for CSV files (see frame_to_xy)."""
    return frame_to_xy(pd.read_csv(path, low_memory=False), label_col, drop_cols, positive_if, max_onehot)
