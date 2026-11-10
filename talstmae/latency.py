"""End-to-end latency measurement (reviewer requests R1-1, R2-3, R2-6).

The test stream is replayed event by event in timestamp order. For every event
the wall-clock time of each pipeline stage is measured on the actual hardware:

    encode (TAE) -> window assembly -> classify (HBLT) -> ECS JSON -> SIEM ingest
    (Elasticsearch ``refresh=wait_for`` = searchable/dashboard-visible, or Logstash
    -> Elasticsearch with polling) -> automated response action (optional)

Measured service times are then combined with the event arrival times through a
single-server FIFO queue (Lindley recursion), so that queueing under the real
arrival rate is included:  start_j = max(arrival_j, finish_{j-1}),
finish_j = start_j + service_j.  For every attack event i detected (causal rule)
at event j, the end-to-end detection latency is  finish_j - arrival_i, i.e. the
event-time delay (ts_j - ts_i) plus window accumulation already contained in it,
plus queueing, inference, JSON generation and SIEM ingestion. Adding the measured
response-action time gives a measured automated-response latency (testbed).
"""
from __future__ import annotations

import http.client
import json
import os
import platform
import time
import urllib.parse
import uuid
from collections import deque

import numpy as np
import pandas as pd
import tensorflow as tf

from .alerts import INDEX, ecs_document
from .metrics import detection_indices

perf = time.perf_counter


# ----------------------------------------------------------------------------- hardware
def hardware_info() -> dict:
    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return dict(cpu=cpu, logical_cores=os.cpu_count(), platform=platform.platform(),
                gpus=[d.name for d in tf.config.list_physical_devices("GPU")], tensorflow=tf.__version__)


# ----------------------------------------------------------------------------- SIEM sinks
class _HTTP:
    def __init__(self, url: str, timeout: float = 30.0):
        u = urllib.parse.urlparse(url)
        self.base = u.path.rstrip("/")
        self.auth = None
        if u.username:
            import base64
            self.auth = "Basic " + base64.b64encode(f"{u.username}:{u.password}".encode()).decode()
        cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        self.conn = cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout)
        self.conn.connect()
        import socket
        self.conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # avoid Nagle/delayed-ACK stalls

    def request(self, method: str, path: str, body: str | None = None, ctype="application/json"):
        headers = {"Content-Type": ctype}
        if self.auth:
            headers["Authorization"] = self.auth
        self.conn.request(method, (self.base + path) or "/", body=body, headers=headers)
        r = self.conn.getresponse()
        data = r.read()
        return r.status, data


class NullSink:
    name = "none"

    def send(self, doc_json: str, alert_id: str) -> None:
        pass


class FileSink:
    name = "ndjson-file"

    def __init__(self, path):
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a")

    def send(self, doc_json: str, alert_id: str) -> None:
        self.f.write(json.dumps({"index": {"_index": INDEX}}) + "\n" + doc_json + "\n")
        self.f.flush()


class ElasticsearchSink:
    """Index each alert with refresh=wait_for: returns only once the alert is searchable (dashboard-visible)."""
    name = "elasticsearch"

    def __init__(self, url: str, index: str = INDEX):
        self.http, self.index = _HTTP(url), index
        status, _ = self.http.request("PUT", f"/{index}", json.dumps({"settings": {"number_of_replicas": 0}}))
        if status not in (200, 400):                                   # 400 = already exists
            raise RuntimeError(f"Elasticsearch index creation failed: HTTP {status}")

    def send(self, doc_json: str, alert_id: str) -> None:
        status, body = self.http.request("PUT", f"/{self.index}/_doc/{alert_id}?refresh=wait_for", doc_json)
        if status not in (200, 201):
            raise RuntimeError(f"Elasticsearch indexing failed: HTTP {status} {body[:200]!r}")


class LogstashSink:
    """Send the alert to a Logstash HTTP input and poll Elasticsearch until the document is searchable."""
    name = "logstash->elasticsearch"

    def __init__(self, logstash_url: str, es_url: str, index: str = INDEX, poll_s: float = 0.005,
                 timeout_s: float = 30.0):
        self.ls, self.es, self.index = _HTTP(logstash_url), _HTTP(es_url), index
        self.poll_s, self.timeout_s = poll_s, timeout_s

    def send(self, doc_json: str, alert_id: str) -> None:
        status, _ = self.ls.request("POST", "/", doc_json)
        if status not in (200, 201):
            raise RuntimeError(f"Logstash HTTP input failed: HTTP {status}")
        q = json.dumps({"query": {"match": {"event.id": alert_id}}, "size": 0})
        t_end = perf() + self.timeout_s
        while perf() < t_end:
            status, body = self.es.request("POST", f"/{self.index}/_search", q)
            if status == 200 and json.loads(body)["hits"]["total"]["value"] > 0:
                return
            time.sleep(self.poll_s)
        raise TimeoutError(f"alert {alert_id} not searchable after {self.timeout_s}s")


# ----------------------------------------------------------------------------- automated response actions
class NullResponder:
    name = "none"

    def act(self, doc: dict) -> None:
        pass


class FileBlocklistResponder:
    """Testbed containment action: append a block rule to a blocklist file consumed by a firewall, then fsync."""
    name = "blocklist-file"

    def __init__(self, path):
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a")

    def act(self, doc: dict) -> None:
        self.f.write(json.dumps({"action": "block", "source.ip": doc["source"]["ip"], "alert": doc["event"]["id"],
                                 "issued": time.time()}) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())


class WebhookResponder:
    """POST the alert to a SOAR webhook and wait for its acknowledgement."""
    name = "soar-webhook"

    def __init__(self, url: str):
        self.http = _HTTP(url)

    def act(self, doc: dict) -> None:
        status, _ = self.http.request("POST", "", json.dumps(doc))
        if status >= 300:
            raise RuntimeError(f"SOAR webhook failed: HTTP {status}")


# ----------------------------------------------------------------------------- streaming pipelines
class WindowPipeline:
    """TA-LSTMAE online pipeline: TAE encodes each arriving record; the classifier scores the latest window."""

    def __init__(self, encoder, classifier, seq_len: int, name: str = "TA-LSTMAE"):
        self.seq_len, self.name = seq_len, name
        d_in, d_lat = encoder.input_shape[-1], encoder.output_shape[-1]
        self._enc = tf.function(lambda x: encoder(x, training=False),
                                input_signature=[tf.TensorSpec([1, d_in], tf.float32)])
        self._clf = tf.function(lambda w: classifier(w, training=False),
                                input_signature=[tf.TensorSpec([1, seq_len, d_lat], tf.float32)])
        self.buf = deque(maxlen=seq_len)

    def reset(self):
        self.buf.clear()

    def step(self, x: np.ndarray):
        t0 = perf()
        z = self._enc(x[None].astype(np.float32)).numpy()[0]
        t1 = perf()
        self.buf.append(z)
        if len(self.buf) < self.seq_len:
            return None, dict(t_encode=t1 - t0, t_window=perf() - t1, t_classify=0.0)
        w = np.stack(self.buf)[None]
        t2 = perf()
        p = float(self._clf(w).numpy().ravel()[0])
        t3 = perf()
        return p, dict(t_encode=t1 - t0, t_window=t2 - t1, t_classify=t3 - t2)


class RecordPipeline:
    """Record-level baseline pipeline: optional encoder followed by a classifier, one record at a time."""

    def __init__(self, classifier, encoder=None, name: str = "baseline"):
        self.name = name
        d_in = (encoder or classifier).input_shape[-1]
        self._enc = None
        if encoder is not None:
            self._enc = tf.function(lambda x: encoder(x, training=False),
                                    input_signature=[tf.TensorSpec([1, d_in], tf.float32)])
            d_cls = encoder.output_shape[-1]
        else:
            d_cls = d_in
        self._clf = tf.function(lambda x: classifier(x, training=False),
                                input_signature=[tf.TensorSpec([1, d_cls], tf.float32)])

    def reset(self):
        pass

    def step(self, x: np.ndarray):
        t0 = perf()
        z = self._enc(x[None].astype(np.float32)) if self._enc is not None else x[None].astype(np.float32)
        t1 = perf()
        p = float(self._clf(z).numpy().ravel()[0])
        return p, dict(t_encode=t1 - t0, t_window=0.0, t_classify=perf() - t1)


def replay(pipeline, X, y, ts, dataset: str, threshold: float = 0.5, sink=None, responder=None,
           start: int = 0, n_events: int | None = None, warmup: int = 50) -> pd.DataFrame:
    """Replay events [start, start+n_events) through the pipeline and time every stage (seconds)."""
    sink, responder = sink or NullSink(), responder or NullResponder()
    end = len(X) if n_events is None else min(len(X), start + n_events)
    pipeline.reset()                                                    # graph tracing outside the timed region
    for i in range(min(len(X), getattr(pipeline, "seq_len", 1) + 5)):
        pipeline.step(X[i])
    pipeline.reset()
    for i in range(max(0, start - warmup), start):                      # warm-up: fill the window buffer
        pipeline.step(X[i])
    rows = []
    for i in range(start, end):
        p, t = pipeline.step(X[i])
        t_json = t_ingest = t_resp = 0.0
        alert = p is not None and p >= threshold
        if alert:
            a0 = perf()
            alert_id = uuid.uuid4().hex
            doc = ecs_document(dataset, i, ts[i], p, model=pipeline.name, alert_id=alert_id)
            s = json.dumps(doc)
            a1 = perf()
            sink.send(s, alert_id)
            a2 = perf()
            responder.act(doc)
            a3 = perf()
            t_json, t_ingest, t_resp = a1 - a0, a2 - a1, a3 - a2
        rows.append(dict(index=i, ts=float(ts[i]), y_true=int(y[i]), prob=np.nan if p is None else p,
                         y_pred=int(alert), **t, t_json=t_json, t_ingest=t_ingest, t_response=t_resp))
    df = pd.DataFrame(rows)
    df["t_detect_path"] = df[["t_encode", "t_window", "t_classify", "t_json", "t_ingest"]].sum(axis=1)
    df["t_service"] = df["t_detect_path"] + df["t_response"]
    return df


def queue_latency(arrival: np.ndarray, service: np.ndarray, speedup: float = 1.0) -> np.ndarray:
    """Single-server FIFO queue (Lindley recursion). Returns finish times relative to the time origin."""
    a = (np.asarray(arrival, float) - arrival[0]) / speedup
    finish = np.empty_like(a)
    prev = -np.inf
    for j, (aj, sj) in enumerate(zip(a, service)):
        prev = max(aj, prev) + sj
        finish[j] = prev
    return finish


def end_to_end_alerts(rep: pd.DataFrame, tol: int = 5, speedup: float = 1.0) -> pd.DataFrame:
    """Per-attack end-to-end latency under the causal (onset-forward) matching rule."""
    y, pred, ts = rep.y_true.values, rep.y_pred.values, rep.ts.values
    att, det, _ = detection_indices(y, pred, tol, causal=True)
    a = (ts - ts[0]) / speedup
    fin_detect = queue_latency(ts, rep.t_service.values, speedup) - rep.t_response.values   # alert searchable
    fin_resp = queue_latency(ts, rep.t_service.values, speedup)                             # response executed
    ok = det >= 0
    i, j = att[ok], det[ok]
    return pd.DataFrame(dict(attack_index=rep["index"].values[i], detect_index=rep["index"].values[j],
                             event_time_delay_s=(ts[j] - ts[i]) / speedup,
                             processing_and_queue_s=fin_detect[j] - a[j],
                             end_to_end_detection_s=fin_detect[j] - a[i],
                             automated_response_s=fin_resp[j] - a[i]))


def summarize_latency(rep: pd.DataFrame, e2e: pd.DataFrame, ms: bool = True) -> pd.DataFrame:
    k = 1000.0 if ms else 1.0
    alerts = rep[rep.y_pred == 1]
    cols = {"TAE encode": rep.t_encode, "window assembly": rep.t_window, "classify": rep.t_classify,
            "ECS JSON (alerts)": alerts.t_json, "SIEM ingest (alerts)": alerts.t_ingest,
            "response action (alerts)": alerts.t_response, "per-event service": rep.t_service,
            "end-to-end detection": e2e.end_to_end_detection_s, "automated response": e2e.automated_response_s,
            "event-time delay (causal)": e2e.event_time_delay_s}
    rows = []
    for name, s in cols.items():
        s = pd.Series(s).dropna()
        if len(s):
            rows.append(dict(stage=name, n=len(s), mean=s.mean() * k, median=s.median() * k,
                             p95=s.quantile(0.95) * k, p99=s.quantile(0.99) * k, max=s.max() * k))
    return pd.DataFrame(rows).rename(columns=lambda c: f"{c}_ms" if c in ("mean", "median", "p95", "p99", "max") and ms
                                     else c)


def throughput(pipeline_models, X, seq_len: int, batch_sizes=(1, 32, 512), repeats: int = 3) -> pd.DataFrame:
    """Offline batched throughput (events/s) of encoder+classifier for several batch sizes."""
    enc, clf = pipeline_models
    rows = []
    for bs in batch_sizes:
        n = min(len(X) - seq_len, max(bs * 20, 2000))
        xb = X[: n + seq_len - 1]
        best = np.inf
        for _ in range(repeats):
            t0 = perf()
            z = enc.predict(xb, batch_size=max(bs, 32), verbose=0) if enc is not None else xb
            if clf.input_shape[1:] == (seq_len, z.shape[1]):
                w = np.lib.stride_tricks.sliding_window_view(z, (seq_len, z.shape[1]))[:, 0]
                clf.predict(w, batch_size=bs, verbose=0)
            else:
                clf.predict(z, batch_size=bs, verbose=0)
            best = min(best, perf() - t0)
        rows.append(dict(batch_size=bs, events=n, seconds=best, events_per_second=n / best))
    return pd.DataFrame(rows)
