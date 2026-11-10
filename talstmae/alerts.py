"""Elastic Common Schema (ECS) alert documents (Section 3.5)."""
from __future__ import annotations

import datetime as dt
import json
import uuid

BASE_TIME = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)   # anchor for relative benchmark timestamps
INDEX = "ta-lstmae-alerts"


def severity(conf: float) -> str:
    return "critical" if conf >= 0.95 else "high" if conf >= 0.8 else "medium" if conf >= 0.6 else "low"


def ecs_document(dataset: str, index: int, ts_seconds: float, prob: float, mttd: float | None = None,
                 t_response: float = 30.0, model: str = "TA-LSTMAE", source_ip: str | None = None,
                 alert_id: str | None = None) -> dict:
    doc = {
        "@timestamp": (BASE_TIME + dt.timedelta(seconds=float(ts_seconds))).isoformat(),
        "event": {"kind": "alert", "category": ["intrusion_detection"], "dataset": dataset,
                  "id": alert_id or uuid.uuid4().hex},
        "source": {"ip": source_ip},
        "rule": {"name": f"{model} binary intrusion detector"},
        "ta_lstmae": {"prediction": "attack", "confidence": round(float(prob), 6), "event_index": int(index),
                      "model": model},
        "siem": {"severity": severity(float(prob))},
    }
    if mttd is not None:
        doc["ta_lstmae"].update(mttd_seconds=round(float(mttd), 6),
                                mttr_seconds=round(float(mttd) + t_response, 6),
                                mttr_type="projected (MTTD + configurable response latency)")
    return doc


def write_bulk_ndjson(docs, path, index: str = INDEX) -> int:
    n = 0
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps({"index": {"_index": index}}) + "\n" + json.dumps(d) + "\n")
            n += 1
    return n
