"""Metrics engine (Section 3.4) and uncertainty / significance utilities.

* Classification metrics are computed per evaluated target (the final event of
  each ten-event window; for record-level baselines the same records).
* MTTD: for every attack target a, t_detect(a) is the timestamp of the first
  True Positive inside the matching window; delay = max(0, t_detect - t_start).
  - causal (onset-forward, PRIMARY): offsets 0..+tol
  - symmetric (sensitivity only):    offsets -tol..+tol
* pMTTR = MTTD + configurable response latency (a projection, not a measurement).
* APR = 1 - FP/(TP+FP) = TP/(TP+FP)  (identical to Precision).
* Detection Rate: an attack is credited if ANY predicted-positive target falls in
  the matching window (incident-level aggregate, distinct from Recall).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y_true, y_pred), precision=precision_score(y_true, y_pred, zero_division=0),
                recall=recall_score(y_true, y_pred, zero_division=0), f1=f1_score(y_true, y_pred, zero_division=0),
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp), n=int(len(y_true)))


def detection_indices(y_true, y_pred, tol: int = 5, causal: bool = True):
    """For every attack target return the index of the first TP in its matching window (-1 if none)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    att = np.flatnonzero(y_true == 1)
    det = np.full(len(att), -1, dtype=np.int64)
    any_pos = np.zeros(len(att), bool)
    tp_mask = (y_pred == 1) & (y_true == 1)
    for off in (range(0, tol + 1) if causal else range(-tol, tol + 1)):         # ascending = earliest first
        j = att + off
        ok = (j >= 0) & (j < n)
        jj = np.clip(j, 0, n - 1)
        hit = ok & tp_mask[jj] & (det < 0)
        det[hit] = jj[hit]
        any_pos |= ok & (y_pred[jj] == 1)
    return att, det, any_pos


def temporal_metrics(y_true, y_pred, ts, tol: int = 5, causal: bool = True, t_response: float = 30.0) -> dict:
    y_true, y_pred, ts = np.asarray(y_true), np.asarray(y_pred), np.asarray(ts, dtype=float)
    att, det, any_pos = detection_indices(y_true, y_pred, tol, causal)
    ok = det >= 0
    delays = np.maximum(0.0, ts[det[ok]] - ts[att[ok]])
    mttd = float(delays.mean()) if delays.size else float("nan")
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return dict(MTTD=mttd, pMTTR=mttd + t_response, APR=(1 - fp / (tp + fp)) if tp + fp else float("nan"),
                DetectionRate=float(any_pos.mean()) if len(att) else float("nan"),
                attacks=int(len(att)), detected_TP=int(ok.sum()))


def evaluate_predictions(y_true, prob, ts, threshold: float = 0.5, tol: int = 5, t_response: float = 30.0) -> dict:
    y_pred = (np.asarray(prob) >= threshold).astype(np.int8)
    cls = classification_metrics(y_true, y_pred)
    cau = temporal_metrics(y_true, y_pred, ts, tol, causal=True, t_response=t_response)
    sym = temporal_metrics(y_true, y_pred, ts, tol, causal=False, t_response=t_response)
    return dict(**cls, MTTD_causal=cau["MTTD"], pMTTR_causal=cau["pMTTR"], MTTD_symmetric=sym["MTTD"],
                APR=cau["APR"], DetectionRate_causal=cau["DetectionRate"],
                DetectionRate_symmetric=sym["DetectionRate"])


# ----------------------------------------------------------------------------- uncertainty
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "MTTD_causal", "MTTD_symmetric", "APR",
               "DetectionRate_causal"]


def mean_ci(x, conf: float = 0.95):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan
    m, sd = x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0
    h = stats.t.ppf((1 + conf) / 2, len(x) - 1) * sd / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return m, sd, m - h, m + h


def summarize_runs(df: pd.DataFrame, by=("dataset", "model"), cols=METRIC_COLS) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(list(by), sort=False):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)), runs=len(g))
        for c in cols:
            if c in g:
                m, sd, lo, hi = mean_ci(g[c])
                row[f"{c}_mean"], row[f"{c}_sd"], row[f"{c}_ci_low"], row[f"{c}_ci_high"] = m, sd, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def format_mean_sd(summary: pd.DataFrame, cols=("accuracy", "precision", "recall", "f1", "MTTD_causal"),
                   digits: int = 4) -> pd.DataFrame:
    out = summary[[c for c in summary.columns if c in ("dataset", "model", "runs")]].copy()
    for c in cols:
        if f"{c}_mean" in summary:
            out[c] = [f"{m:.{digits}f} ± {s:.{digits}f}" for m, s in zip(summary[f"{c}_mean"], summary[f"{c}_sd"])]
    return out


def mcnemar_exact(y, pred_a, pred_b) -> float:
    """Exact (binomial) McNemar test on paired predictions over the same test targets."""
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    n01 = int(((a == y) & (b != y)).sum())
    n10 = int(((a != y) & (b == y)).sum())
    return 1.0 if n01 + n10 == 0 else float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)


def paired_seed_test(df: pd.DataFrame, metric: str, model_a: str, model_b: str, dataset: str) -> dict:
    """Paired comparison across seeds (same seed list for both models): paired t-test and Wilcoxon."""
    a = df[(df.dataset == dataset) & (df.model == model_a)].set_index("seed")[metric]
    b = df[(df.dataset == dataset) & (df.model == model_b)].set_index("seed")[metric]
    common = a.index.intersection(b.index)
    a, b = a.loc[common].values, b.loc[common].values
    out = dict(dataset=dataset, metric=metric, model_a=model_a, model_b=model_b, n_seeds=len(common),
               mean_diff=float(np.mean(a - b)) if len(common) else np.nan)
    if len(common) >= 2 and np.any(a != b):
        out["paired_t_p"] = float(stats.ttest_rel(a, b).pvalue)
        out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue) if len(common) >= 5 else np.nan
    else:
        out["paired_t_p"] = out["wilcoxon_p"] = np.nan
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj
