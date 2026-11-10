"""Strict data loading, preprocessing, contiguous splits, timestamps and windowing.

Design rule adopted after the revision (Section 4.8): loaders NEVER substitute
placeholder data. A missing file raises an error. Synthetic data exist only in
the explicit smoke-test path, whose outputs are written to a separate folder.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

from .config import Config

PROVENANCE: dict = {}

NSL_COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
            "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
            "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
            "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
            "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]

EXPECTED_DIM = {"nsl_kdd": 122, "unsw_nb15": 194, "cicids2018": 78}
EXPECTED_WINDOWS = {"nsl_kdd": (113_367, 12_588, 22_535), "unsw_nb15": (157_798, 17_525, 82_323),
                    "cicids2018": (110_093, 12_224, 52_419)}
NAMES = {"nsl_kdd": "NSL-KDD", "unsw_nb15": "UNSW-NB15", "cicids2018": "CICIDS-2018"}


# ----------------------------------------------------------------------------- provenance
def sha256(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def require(path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset file not found: {path}\n"
            "This pipeline deliberately refuses to substitute placeholder data (manuscript Section 4.8). "
            "Download the official release (see README) or use Config().smoke() to test the code only.")
    PROVENANCE[str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return path


def check_count(name: str, got: int, expected: int) -> None:
    if got != expected:
        warnings.warn(f"{name}: {got:,} records, expected {expected:,} (official release).")
    print(f"  {name:<28s} {got:>9,d}  (expected {expected:,d})  {'OK' if got == expected else 'MISMATCH'}")


# ----------------------------------------------------------------------------- loaders
def load_nsl_kdd(cfg: Config) -> dict:
    d = Path(cfg.data_dir)
    tr = pd.read_csv(require(d / "nsl-kdd/KDDTrain+.txt"), header=None, names=NSL_COLS)
    te = pd.read_csv(require(d / "nsl-kdd/KDDTest+.txt"), header=None, names=NSL_COLS)
    check_count("NSL-KDD train", len(tr), 125_973)
    check_count("NSL-KDD test", len(te), 22_544)
    y_tr = (tr.pop("label") != "normal").astype(np.int8).values
    y_te = (te.pop("label") != "normal").astype(np.int8).values
    tr.pop("difficulty"); te.pop("difficulty")
    return dict(train=tr, test=te, y_train=y_tr, y_test=y_te,
                cat_cols=["protocol_type", "service", "flag"], ts_train=None, ts_test=None)


def load_unsw_nb15(cfg: Config) -> dict:
    d = Path(cfg.data_dir)
    tr = pd.read_csv(require(d / "unsw-nb15/UNSW_NB15_training-set.csv"), encoding="utf-8-sig")
    te = pd.read_csv(require(d / "unsw-nb15/UNSW_NB15_testing-set.csv"), encoding="utf-8-sig")
    if len(tr) < len(te):
        # The official partitioned release names the 82,332-record file "training-set" and the 175,341-record file
        # "testing-set". Following the convention of the compared studies (175,341 train / 82,332 test), the larger
        # partition is used for training.
        print("  note: UNSW-NB15 official file names are reversed relative to their sizes; using the 175,341-record "
              "partition for training and the 82,332-record partition for testing")
        tr, te = te, tr
    check_count("UNSW-NB15 train", len(tr), 175_341)
    check_count("UNSW-NB15 test", len(te), 82_332)
    # The partitioned UNSW-NB15 files are ordered by class (e.g. the 82,332-record file contains only five label
    # blocks), not by time. Windows built in file order would therefore contain neighbours with the same label and
    # inflate window-level accuracy. Each partition is permuted once with a fixed, documented seed before synthetic
    # timestamps and windows are created, so that record order carries no label information.
    if cfg.unsw_shuffle_seed is not None:
        before = (label_runs(tr["label"].values), label_runs(te["label"].values))
        tr = tr.sample(frac=1.0, random_state=cfg.unsw_shuffle_seed).reset_index(drop=True)
        te = te.sample(frac=1.0, random_state=cfg.unsw_shuffle_seed + 1).reset_index(drop=True)
        print(f"  note: UNSW-NB15 records permuted (seed {cfg.unsw_shuffle_seed}); label runs train {before[0]:,} -> "
              f"{label_runs(tr['label'].values):,}, test {before[1]:,} -> {label_runs(te['label'].values):,}")
    y_tr = tr.pop("label").astype(np.int8).values
    y_te = te.pop("label").astype(np.int8).values
    for df in (tr, te):
        df.drop(columns=[c for c in ("id", "attack_cat") if c in df.columns], inplace=True)
    return dict(train=tr, test=te, y_train=y_tr, y_test=y_te,
                cat_cols=["proto", "service", "state"], ts_train=None, ts_test=None)


def label_runs(y) -> int:
    y = np.asarray(y)
    return int(np.sum(y[1:] != y[:-1]) + 1) if len(y) else 0


def _cicids_file(cfg: Config) -> Path:
    d = Path(cfg.data_dir) / "cicids2018"
    p = d / "02-14-2018.csv"
    if not p.exists():                       # accept common download names, e.g. CSE-CIC-IDS2018_02-14-2018.csv
        cands = sorted(d.glob("*02-14-2018*.csv")) + sorted(d.glob("*14-02-2018*.csv"))
        if cands:
            return cands[0]
    return p


def load_cicids2018(cfg: Config) -> dict:
    df = pd.read_csv(require(_cicids_file(cfg)), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Label"] != "Label"]                                          # repeated header rows, if any
    ts = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    feat = df.drop(columns=["Timestamp", "Label"]).apply(pd.to_numeric, errors="coerce")
    # non-finite flow statistics (division by a zero duration) are set to 0 instead of dropping records,
    # so that the chronological subsample keeps every sixth flow of the capture
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ok = ts.notna()
    feat, ts, lab = feat[ok], ts[ok], df.loc[ok, "Label"]
    order = np.argsort(ts.values, kind="stable")                             # chronological order
    feat, ts, lab = feat.iloc[order], ts.iloc[order], lab.iloc[order]
    step = cfg.cic_subsample
    feat, ts, lab = feat.iloc[::step], ts.iloc[::step], lab.iloc[::step]
    print("  CICIDS-2018 attack categories:", sorted(lab.unique()))
    check_count("CICIDS-2018 subset (1/6)", len(feat), 174_763)
    y = (lab != "Benign").astype(np.int8).values
    t = (ts - ts.iloc[0]).dt.total_seconds().values                          # genuine native timestamps (s)
    n = len(feat)
    n_test = int(cfg.cic_test_fraction * n)
    return dict(train=feat.iloc[: n - n_test].reset_index(drop=True), test=feat.iloc[n - n_test:].reset_index(drop=True),
                y_train=y[: n - n_test], y_test=y[n - n_test:], cat_cols=[],
                ts_train=t[: n - n_test], ts_test=t[n - n_test:])


def make_smoke_dataset(name: str, n_train: int = 12000, n_test: int = 3000) -> dict:
    """Explicit, clearly-labelled placeholder data for the smoke test only."""
    rng = np.random.default_rng(0)

    def frame(n):
        y = (rng.random(n) < 0.45).astype(np.int8)
        X = pd.DataFrame(rng.normal(size=(n, 12)) + y[:, None] * 1.5, columns=[f"f{i}" for i in range(12)])
        X["cat_a"] = rng.choice(["tcp", "udp", "icmp"], n)
        X["cat_b"] = rng.choice(list("abcde"), n)
        return X, y

    tr, ytr = frame(n_train)
    te, yte = frame(n_test)
    ts_tr = ts_te = None
    if name == "cicids2018":
        ts_tr = np.sort(rng.uniform(0, 3600, n_train))
        ts_te = np.sort(rng.uniform(3600, 5000, n_test))
    return dict(train=tr, test=te, y_train=ytr, y_test=yte, cat_cols=["cat_a", "cat_b"], ts_train=ts_tr, ts_test=ts_te)


LOADERS = {"nsl_kdd": load_nsl_kdd, "unsw_nb15": load_unsw_nb15, "cicids2018": load_cicids2018}


# ----------------------------------------------------------------------------- preprocessing
def encode(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols):
    tr = pd.get_dummies(train_df, columns=cat_cols, dtype=np.float32)
    te = pd.get_dummies(test_df, columns=cat_cols, dtype=np.float32).reindex(columns=tr.columns, fill_value=0.0)
    return tr.astype(np.float32), te.astype(np.float32)


def contiguous_split(n: int, val_fraction: float):
    """Train/validation split on contiguous blocks, BEFORE windowing (no shuffling)."""
    n_val = int(val_fraction * n)
    return np.arange(0, n - n_val), np.arange(n - n_val, n)


def synthetic_timestamps(n: int, seed: int, mean: float) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).exponential(mean, size=n))


def apply_resolution(ts: np.ndarray, res):
    return ts if res is None else np.floor(ts / res) * res


def prepare_dataset(name: str, cfg: Config, verbose: bool = True) -> dict:
    """Load -> encode -> scale (train split only) -> class weights -> timestamps. Returns a dict of arrays."""
    if verbose:
        print(f"[{name}]")
    raw = make_smoke_dataset(name) if cfg.smoke_test else LOADERS[name](cfg)
    tr_df, te_df = encode(raw["train"], raw["test"], raw["cat_cols"])
    idx_tr, idx_val = contiguous_split(len(tr_df), cfg.val_fraction)
    scaler = MinMaxScaler().fit(tr_df.values[idx_tr])                        # fitted on the training split only
    X_full = np.clip(scaler.transform(tr_df.values), 0, 1).astype(np.float32)
    X_te = np.clip(scaler.transform(te_df.values), 0, 1).astype(np.float32)
    y_full = raw["y_train"]
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_full[idx_tr])

    if raw["ts_train"] is None:                                              # synthetic (NSL-KDD, UNSW-NB15)
        seed = cfg.ts_seeds.get(name, 2000)
        ts_full = synthetic_timestamps(len(y_full), seed, cfg.mean_interarrival)
        ts_te = synthetic_timestamps(len(raw["y_test"]), seed + 500, cfg.mean_interarrival)
        ts_source = f"synthetic exponential (mean={cfg.mean_interarrival}s, seeds={seed}/{seed + 500})"
        native = False
    else:                                                                    # genuine native timestamps
        ts_full, ts_te = raw["ts_train"], raw["ts_test"]
        ts_source, native = "native flow timestamps (chronological split)", True
    res = cfg.ts_resolution_s.get(name)
    D = dict(name=name, X_tr=X_full[idx_tr], y_tr=y_full[idx_tr], X_val=X_full[idx_val], y_val=y_full[idx_val],
             X_te=X_te, y_te=raw["y_test"], idx_tr=idx_tr, idx_val=idx_val,
             ts_tr=apply_resolution(ts_full[idx_tr], res), ts_val=apply_resolution(ts_full[idx_val], res),
             ts_te=apply_resolution(ts_te, res), native_timestamps=native, ts_source=ts_source,
             class_weight={0: float(cw[0]), 1: float(cw[1])}, feature_names=list(tr_df.columns))

    out = cfg.out / name
    out.mkdir(exist_ok=True)
    (out / "class_weights.json").write_text(json.dumps(D["class_weight"], indent=2))
    np.savez_compressed(out / "partition_indices.npz", train=idx_tr, val=idx_val, test=np.arange(len(X_te)))
    np.savez_compressed(out / "timestamps.npz", train=D["ts_tr"], val=D["ts_val"], test=D["ts_te"])
    if verbose:
        dim = X_full.shape[1]
        chk = "" if cfg.smoke_test else ("OK" if dim == EXPECTED_DIM[name] else f"MISMATCH (expected {EXPECTED_DIM[name]})")
        print(f"  input_dim={dim} {chk} | train={len(idx_tr):,} val={len(idx_val):,} test={len(X_te):,} | "
              f"class_weight={ {k: round(v, 4) for k, v in D['class_weight'].items()} } | timestamps: {ts_source}, "
              f"resolution={res}s")
    return D


# ----------------------------------------------------------------------------- windowing
def window_targets(D: dict, split: str, seq_len: int):
    """Labels and timestamps of the final event of every window in a split."""
    y, ts = D[f"y_{split}"], D[f"ts_{split}"]
    return y[seq_len - 1:], ts[seq_len - 1:]


def make_window_ds(X, y, seq_len: int, batch: int, shuffle: bool = False, seed: int = 42):
    """Windows X[i:i+seq_len] -> label y[i+seq_len-1]; always built inside ONE split."""
    import tensorflow as tf
    return tf.keras.utils.timeseries_dataset_from_array(
        X, np.asarray(y)[seq_len - 1:], sequence_length=seq_len, sequence_stride=1,
        batch_size=batch, shuffle=shuffle, seed=seed)


def leakage_audit(D: dict, cfg: Config) -> tuple:
    a, b = D["idx_tr"], D["idx_val"]
    assert np.all(np.diff(a) == 1) and np.all(np.diff(b) == 1), "splits must be contiguous"
    assert a[-1] < b[0] and len(np.intersect1d(a, b)) == 0, "train/validation overlap"
    for t in (D["ts_tr"], D["ts_val"], D["ts_te"]):
        assert np.all(np.diff(t) >= 0), "timestamps must be non-decreasing within each split"
    counts = tuple(len(D[k]) - (cfg.seq_len - 1) for k in ("y_tr", "y_val", "y_te"))
    exp = EXPECTED_WINDOWS[D["name"]]
    status = "n/a (smoke)" if cfg.smoke_test else ("OK" if counts == exp else f"MISMATCH, expected {exp}")

    def hashes(X, k=20000):
        W = np.lib.stride_tricks.sliding_window_view(X[: k + cfg.seq_len - 1], (cfg.seq_len, X.shape[1]))[:, 0]
        return {hashlib.md5(np.round(w, 6).tobytes()).hexdigest() for w in W}

    dup = len(hashes(D["X_tr"]) & hashes(D["X_te"]))
    y = D["y_te"]; p = float(np.mean(y)); expected = 1 + 2 * (len(y) - 1) * p * (1 - p)
    ratio = label_runs(y) / expected if expected > 0 else float("nan")
    order_note = "random-like" if ratio > 0.5 else ("chronological (genuine timestamps)" if D["native_timestamps"]
                                                    else "WARNING: records appear ordered by class")
    print(f"{D['name']:<11s} windows train/val/test = {counts}  {status} | contiguous, disjoint, per-split "
          f"windowing: OK | exact duplicate windows train∩test (first 20k): {dup} | test label-run ratio "
          f"{ratio:.2f} ({order_note})")
    return counts
