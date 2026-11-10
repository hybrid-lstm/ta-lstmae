import numpy as np
from sklearn.metrics import precision_score

from talstmae.metrics import mcnemar_exact, mean_ci, temporal_metrics


def test_temporal_metrics_toy():
    y = np.array([0, 1, 1, 0, 0, 1, 0, 0])
    p = np.array([1, 0, 1, 0, 0, 0, 0, 1])
    ts = np.arange(8, dtype=float)
    s = temporal_metrics(y, p, ts, tol=1, causal=False)
    c = temporal_metrics(y, p, ts, tol=1, causal=True)
    assert s["detected_TP"] == 2 and s["MTTD"] == 0.5 and c["MTTD"] == 0.5
    assert abs(s["APR"] - precision_score(y, p)) < 1e-12        # APR == Precision by definition


def test_causal_never_uses_earlier_detection():
    y = np.array([0, 0, 1, 1, 0])
    p = np.array([0, 0, 0, 1, 0])
    ts = np.array([0., 1., 2., 3., 4.])
    c = temporal_metrics(y, p, ts, tol=2, causal=True)
    assert c["detected_TP"] == 2 and abs(c["MTTD"] - 0.5) < 1e-12   # attack@2 detected at 3 (1 s), attack@3 at 3 (0 s)


def test_mean_ci_and_mcnemar():
    m, sd, lo, hi = mean_ci([1.0, 2.0, 3.0])
    assert m == 2.0 and lo < m < hi
    y = np.array([1, 1, 0, 0]); a = y.copy(); b = 1 - y
    assert mcnemar_exact(y, a, a) == 1.0 and mcnemar_exact(y, a, b) < 0.2
