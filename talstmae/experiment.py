"""Training / evaluation runners shared by the notebooks (single runs, ablation, head-to-head, multi-seed)."""
from __future__ import annotations

import gc
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from . import baselines as B
from . import models as M
from .config import Config, set_all_seeds
from .data import make_window_ds
from .metrics import evaluate_predictions

TA_FULL = "TA-LSTMAE (TAE + HBLT, full)"
TA_CAUSAL = "TA-LSTMAE causal (TAE + unidirectional LSTM)"
TA_CNN = "TA-LSTMAE simplified (TAE + CNN only)"
HAE_HRL = "HAE-HRL (re-implemented)"
ADFCNN = "ADFCNN-BiLSTM (re-implemented)"
HEAD_TO_HEAD_MODELS = [TA_FULL, TA_CAUSAL, TA_CNN, HAE_HRL, ADFCNN]
_TA_BRANCHES = {TA_FULL: M.CONFIGS[M.FULL]["branches"], TA_CAUSAL: M.CONFIGS[M.CAUSAL]["branches"],
                TA_CNN: M.CONFIGS["TAE + CNN only"]["branches"]}


def free_memory():
    """Release Keras/TensorFlow graph memory between independent trainings (long multi-seed runs)."""
    tf.keras.backend.clear_session()
    gc.collect()


# ----------------------------------------------------------------------------- TAE
def train_tae(D: dict, cfg: Config, seed: int, save_dir=None):
    set_all_seeds(seed, cfg.deterministic_ops)
    ae, enc = M.build_tae(D["X_tr"].shape[1], cfg.latent_dim[D["name"]], cfg.lr)
    cbs = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)]
    h = ae.fit(D["X_tr"], D["X_tr"], validation_data=(D["X_val"], D["X_val"]), epochs=cfg.tae_epochs,
               batch_size=cfg.batch_size, callbacks=cbs, verbose=0)
    lat = {k: enc.predict(D[f"X_{k}"], batch_size=4096, verbose=0) for k in ("tr", "val", "te")}
    if save_dir is not None:
        ae.save(save_dir / "tae.keras")
        enc.save(save_dir / "tae_encoder.keras")
        pd.DataFrame(h.history).to_csv(save_dir / "tae_history.csv", index=False)
    return ae, enc, lat


# ----------------------------------------------------------------------------- window classifiers (HBLT family)
def predict_windows(model, X, cfg: Config, batch: int = 4096) -> np.ndarray:
    return model.predict(make_window_ds(X, np.zeros(len(X), np.int8), cfg.seq_len, batch), verbose=0).ravel()


def train_window_classifier(D: dict, Xs: dict, cfg: Config, branches, epochs: int, seed: int, name="HBLT"):
    set_all_seeds(seed, cfg.deterministic_ops)
    model = M.build_hblt(cfg.seq_len, Xs["tr"].shape[1], branches, cfg.dropout, cfg.lr, name=name)
    h = model.fit(make_window_ds(Xs["tr"], D["y_tr"], cfg.seq_len, cfg.batch_size, shuffle=True, seed=seed),
                  validation_data=make_window_ds(Xs["val"], D["y_val"], cfg.seq_len, cfg.batch_size),
                  epochs=epochs, class_weight=D["class_weight"],
                  callbacks=[tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)], shuffle=False, verbose=0)
    return model, h, predict_windows(model, Xs["val"], cfg), predict_windows(model, Xs["te"], cfg)


# ----------------------------------------------------------------------------- record-level baselines
def _fit_record_classifier(model, D, Xs, cfg, seed):
    set_all_seeds(seed, cfg.deterministic_ops)
    cbs = [tf.keras.callbacks.EarlyStopping(patience=cfg.baseline_patience, restore_best_weights=True,
                                            start_from_epoch=3)]          # let BatchNorm statistics settle first
    h = model.fit(Xs["tr"], D["y_tr"], validation_data=(Xs["val"], D["y_val"]), epochs=cfg.baseline_max_epochs,
                  batch_size=cfg.batch_size, class_weight=D["class_weight"], callbacks=cbs, verbose=0)
    s = cfg.seq_len - 1                                    # evaluate exactly the TA-LSTMAE window targets
    p_val = model.predict(Xs["val"], batch_size=4096, verbose=0).ravel()[s:]
    p_te = model.predict(Xs["te"], batch_size=4096, verbose=0).ravel()[s:]
    return h, p_val, p_te


def train_hae_hrl(D: dict, cfg: Config, seed: int):
    set_all_seeds(seed, cfg.deterministic_ops)
    ae, enc = B.build_hae(D["X_tr"].shape[1], cfg.latent_dim[D["name"]], cfg.lr)
    ae.fit(D["X_tr"], D["X_tr"], validation_data=(D["X_val"], D["X_val"]), epochs=cfg.tae_epochs,
           batch_size=cfg.batch_size, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,
                                                                                  restore_best_weights=True)],
           verbose=0)
    Z = {k: enc.predict(D[f"X_{k}"], batch_size=4096, verbose=0) for k in ("tr", "val", "te")}
    clf = B.build_hrl(Z["tr"].shape[1], cfg.dropout, cfg.lr)
    h, p_val, p_te = _fit_record_classifier(clf, D, Z, cfg, seed)
    return dict(models=(enc, clf), params=ae.count_params() + clf.count_params(), history=h, p_val=p_val, p_te=p_te)


def train_adfcnn(D: dict, cfg: Config, seed: int):
    set_all_seeds(seed, cfg.deterministic_ops)
    m = B.build_adfcnn_bilstm(D["X_tr"].shape[1], lr=B.ADFCNN_LR.get(D["name"], 5e-4))
    Xs = {k: D[f"X_{k}"] for k in ("tr", "val", "te")}
    h, p_val, p_te = _fit_record_classifier(m, D, Xs, cfg, seed)
    return dict(models=(m,), params=m.count_params(), history=h, p_val=p_val, p_te=p_te)


# ----------------------------------------------------------------------------- test targets
def test_targets(D: dict, cfg: Config):
    s = cfg.seq_len - 1
    return D["y_te"][s:], D["ts_te"][s:]


def val_targets(D: dict, cfg: Config):
    s = cfg.seq_len - 1
    return D["y_val"][s:], D["ts_val"][s:]


# ----------------------------------------------------------------------------- progress log + resume support
def make_logger(cfg: Config, name: str = "progress.log"):
    """Print and append to outputs/progress.log (readable while a long notebook is running)."""
    path = cfg.out / name

    def log(msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(path, "a") as f:
            f.write(line + "\n")
    return log


def _load_partial(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["dataset", "model", "seed"])


def _append_row(path, row: dict):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def _done(partial: pd.DataFrame, name, model, seed) -> bool:
    return bool(((partial.dataset == name) & (partial.model == model) & (partial.seed == seed)).any())


def _tag(model_name: str) -> str:
    return model_name.split(" (")[0].replace(" ", "_").replace("-", "")


# ----------------------------------------------------------------------------- multi-seed head-to-head
def run_head_to_head(DATA: dict, cfg: Config, models=HEAD_TO_HEAD_MODELS, seeds=None, save_predictions=True,
                     log=None, resume: bool = True) -> pd.DataFrame:
    """Train every model with every seed on identical partitions and evaluate on identical test targets.

    Every finished run is appended to outputs/head_to_head_runs_partial.csv; with resume=True an interrupted
    session continues where it stopped, and sessions run on different datasets accumulate in one file."""
    seeds = tuple(seeds or cfg.seeds)
    log = log or make_logger(cfg)
    part_path = cfg.out / "head_to_head_runs_partial.csv"
    if not resume and part_path.exists():
        part_path.unlink()
    partial = _load_partial(part_path)
    total = len(DATA) * len(seeds) * len(models)
    k = 0
    for name, D in DATA.items():
        y, ts = test_targets(D, cfg)
        run_dir = cfg.out / name / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            lat = None
            for model_name in models:
                k += 1
                pred_file = run_dir / f"{_tag(model_name)}_seed{seed}.npz"
                if resume and _done(partial, name, model_name, seed) and (pred_file.exists() or not save_predictions):
                    log(f"[{k}/{total}] {name} seed={seed} {model_name}: already done, skipped (resume)")
                    continue
                t0 = time.time()
                if model_name in _TA_BRANCHES:
                    if lat is None:                                   # one TAE per (dataset, seed), shared by variants
                        _, _, lat = train_tae(D, cfg, seed)
                    m, h, p_val, p_te = train_window_classifier(D, lat, cfg, _TA_BRANCHES[model_name],
                                                                cfg.hblt_epochs[name], seed)
                    params = M.build_tae(D["X_tr"].shape[1], cfg.latent_dim[name])[0].count_params() + m.count_params()
                elif model_name == HAE_HRL:
                    r = train_hae_hrl(D, cfg, seed)
                    p_val, p_te, params = r["p_val"], r["p_te"], r["params"]
                elif model_name == ADFCNN:
                    r = train_adfcnn(D, cfg, seed)
                    p_val, p_te, params = r["p_val"], r["p_te"], r["params"]
                else:
                    raise ValueError(model_name)
                train_s = time.time() - t0
                res = evaluate_predictions(y, p_te, ts, cfg.threshold, cfg.match_tolerance, cfg.t_response_s)
                if save_predictions:
                    np.savez_compressed(pred_file, y_true=y, timestamp=ts,
                                        prob_test=p_te.astype(np.float32), prob_val=p_val.astype(np.float32))
                _append_row(part_path, dict(dataset=name, model=model_name, seed=seed, params=int(params),
                                            train_seconds=round(train_s, 1), **res))
                log(f"[{k}/{total}] {name} seed={seed} {model_name}: Acc={res['accuracy']:.4f} F1={res['f1']:.4f} "
                    f"MTTD_causal={res['MTTD_causal']:.4f}s ({train_s:.0f}s)")
                free_memory()
    allrows = _load_partial(part_path).drop_duplicates(["dataset", "model", "seed"], keep="last")
    allrows.to_csv(cfg.out / "head_to_head_runs.csv", index=False)
    df = allrows[allrows.dataset.isin(list(DATA)) & allrows.model.isin(models) & allrows.seed.isin(seeds)]
    df = df.assign(_d=df.dataset.map({d: i for i, d in enumerate(DATA)}),
                   _m=df.model.map({m: i for i, m in enumerate(models)}))
    return df.sort_values(["_d", "seed", "_m"]).drop(columns=["_d", "_m"]).reset_index(drop=True)


def load_run_predictions(cfg: Config, dataset: str, model_name: str, seed: int) -> dict:
    return dict(np.load(cfg.out / dataset / "runs" / f"{_tag(model_name)}_seed{seed}.npz"))


def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=float))


# ----------------------------------------------------------------------------- ablation over seeds (Table 7)
def run_ablation(DATA: dict, cfg: Config, seeds=None, configs=None, log=None, resume: bool = True) -> pd.DataFrame:
    """Every Table 7 configuration (+ causal-only) for every seed; one TAE per (dataset, seed) is shared.
    Finished runs are appended to outputs/ablation_runs_partial.csv (resume after interruption)."""
    seeds = tuple(seeds or cfg.seeds)
    configs = configs or list(M.CONFIGS)
    log = log or make_logger(cfg)
    part_path = cfg.out / "ablation_runs_partial.csv"
    if not resume and part_path.exists():
        part_path.unlink()
    partial = _load_partial(part_path)
    total = len(DATA) * len(seeds) * len(configs)
    k = 0
    for name, D in DATA.items():
        y, ts = test_targets(D, cfg)
        for seed in seeds:
            todo = [c for c in configs if not (resume and _done(partial, name, c, seed))]
            k += len(configs) - len(todo)
            if not todo:
                log(f"{name} seed={seed}: all ablation configurations already done, skipped (resume)")
                continue
            _, _, lat = train_tae(D, cfg, seed)
            raw = {kk: D[f"X_{kk}"] for kk in ("tr", "val", "te")}
            for c in todo:
                k += 1
                spec = M.CONFIGS[c]
                t0 = time.time()
                m, _, _, p_te = train_window_classifier(D, raw if spec.get("raw_input") else lat, cfg,
                                                        spec["branches"], cfg.ablation_epochs, seed)
                res = evaluate_predictions(y, p_te, ts, cfg.threshold, cfg.match_tolerance, cfg.t_response_s)
                _append_row(part_path, dict(dataset=name, model=c, seed=seed, params=m.count_params(),
                                            train_seconds=round(time.time() - t0, 1), **res))
                log(f"[{k}/{total}] {name} seed={seed} {c}: F1={res['f1']:.4f} MTTD_causal={res['MTTD_causal']:.4f}s")
                del m
                free_memory()
    allrows = _load_partial(part_path).drop_duplicates(["dataset", "model", "seed"], keep="last")
    allrows.to_csv(cfg.out / "ablation_runs.csv", index=False)
    df = allrows[allrows.dataset.isin(list(DATA)) & allrows.model.isin(configs) & allrows.seed.isin(seeds)]
    return df.reset_index(drop=True)
