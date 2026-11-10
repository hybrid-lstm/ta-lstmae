# TA-LSTMAE — Reproducibility Repository

Code accompanying the manuscript **"Enhancing Temporal Precision of MTTD and MTTR Metrics: A Hybrid LSTM-Autoencoder Framework for Log Sequence Analysis in SIEM Platforms"** (Y. Ariyanto, Y. W. Syaifudin, P. Y. Saputra, C. Setiadi, T. Fatmawati; submitted to the *International Journal of Intelligent Engineering and Systems*).

The repository regenerates every quantity reported in the paper from the official benchmark files. It also contains the experiments added in response to the second review round:

| Notebook | Content | Reviewer comments |
|---|---|---|
| `notebooks/01_reproduce_main_results.ipynb` | Main results (Tables 2–7), raw confusion matrices, causal MTTD as the primary metric, multi-seed ablation with 95% CIs and paired tests, MTTD sensitivity to tolerance and inter-arrival scale, threshold sweep, ECS alerts, environment file, SHA-256 manifest | R1-1, R1-2, R1-5, R1-6, R1-7, R1-8, R1-9, R2-5 |
| `notebooks/02_head_to_head_baselines.ipynb` | Controlled head-to-head comparison with **re-implemented HAE-HRL** and **ADFCNN-BiLSTM (2025)** on identical partitions, preprocessing, test targets, seeds and metrics (including MTTD for every model) | R1-4, R1-10, R2-1, R2-4 |
| `notebooks/03_end_to_end_latency.ipynb` | Event-by-event replay of the natively timestamped CICIDS-2018 stream with wall-clock timing of encode → classify → ECS JSON → Elasticsearch/Logstash ingest (searchable) → automated response; queueing under 1×/10×/100× load | R1-1, R1-3, R2-3, R2-6 |

## Repository layout
```
talstmae/            Python package used by all notebooks
  config.py          single experiment configuration (seeds, splits, epochs, tolerances)
  data.py            strict loaders (no placeholder fallback), preprocessing, contiguous splits, timestamps, windowing, leakage audit
  models.py          TAE and the HBLT family (all ablation variants, causal-only model)
  baselines.py       HAE-HRL and ADFCNN-BiLSTM re-implementations (incl. 1D deformable convolution, ECA, spatial attention)
  metrics.py         classification metrics, causal/symmetric MTTD, pMTTR, APR, Detection Rate, CIs, paired tests, McNemar, Holm
  experiment.py      training runners, multi-seed head-to-head and ablation
  alerts.py          Elastic Common Schema (ECS) alert documents
  latency.py         streaming replay, Elasticsearch/Logstash sinks, response actions, FIFO queue model, throughput
notebooks/           01–03 (see above)
docker/              docker-compose for Elasticsearch + Logstash + Kibana (latency testbed)
tests/               unit tests (parameter counts vs. manuscript, metrics engine, deformable convolution)
data/                place the official dataset files here (see data/README.md)
```

## Easiest way: `notebooks/00_RUN_ME.ipynb`
Open `notebooks/00_RUN_ME.ipynb` in VS Code, Jupyter or Google Colab, edit the **SETTINGS** cell, and choose **Run All**. The launcher then:
* finds the project folder (and mounts Google Drive on Colab);
* installs any missing packages and runs the unit tests;
* checks the hardware and the dataset files, downloading CICIDS-2018 automatically if it is missing;
* starts Elasticsearch/Logstash with Docker when notebook 03 is selected;
* executes notebooks 01 → 02 → 03 with live progress;
* shows the key tables and writes `results_to_send.zip`.

Start with `MODE = "smoke"` (synthetic data; it only checks that the code runs), then use `"quick"` (3 seeds) or `"full"` (5 seeds). Interrupted runs resume automatically, and finished models and seeds are skipped. Progress is also written to `outputs/progress.log`.

In VS Code, select the project's `.venv` as the notebook kernel (top-right *Select Kernel*). The child notebooks run in the same Python environment.

## Manual start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q tests                      # parameter counts, metrics engine and baselines
# put the datasets in data/ (see data/README.md), then:
jupyter lab notebooks/
```
1. Run **01**. It writes checkpoints, predictions and tables to `outputs/`.
2. Run **02**. This is the most compute-intensive step (5 models × 5 seeds × 3 datasets).
3. Start the ELK testbed (`cd docker && docker compose up -d`), set `ES_URL=http://localhost:9200` and `LOGSTASH_URL=http://localhost:8080`, then run **03**.

To check the code without the datasets, set `SMOKE_TEST = True` in the first cell of any notebook. It uses small synthetic data and writes to `outputs_SMOKE_TEST_INVALID/`. Its numbers are meaningless and must never be reported.

## Protocol guarantees
* **No silent data substitution.** A missing dataset file raises an error. Each input file's SHA-256 is recorded in `outputs/MANIFEST.json`.
* **Leakage-free windowing.** The train/validation split uses contiguous blocks and is made *before* windowing. Windows are built inside each split, and the audit asserts contiguity, disjointness and monotone timestamps.
* **No resampling.** Class imbalance is handled with `class_weight` computed from the training split, for every model including the baselines.
* **Common evaluation.** Every model is scored on the same test targets (the final events of the TA-LSTMAE windows) with the same threshold and the same metrics engine.
* **Causal MTTD is primary.** The symmetric ±5-event rule is reported only as a sensitivity analysis.
* **Projected vs. measured response.** `pMTTR = MTTD + 30 s` is a configurable projection. Notebook 03 measures a testbed **automated-containment latency**, which is not a human incident-response time.

## Reconstruction notes and assumptions (please read)
* `models.py` reproduces every parameter count in the manuscript exactly: Table 2 and Section 4.6 (28,833 / 81,089 / 104,001 / 128,033 / 179,617 / 261,025 / 227,233). Three layout details were reconstructed to match those counts and should be confirmed against the original implementation:
  - the decoder input projection Dense(64)+Reshape(4×16), which is not listed in Table 2;
  - the gated-fusion head;
  - the BatchNorm in the CNN branch.
* The baselines are re-implemented from their published descriptions. Hyperparameters that the original papers do not report are listed in `talstmae/baselines.py::BASELINE_NOTES` and must be stated as assumptions when reporting results.
* The synthetic timestamp seeds (1001/1002) and the one-second resolution used for UNSW-NB15 are set in `config.py`.

## Citation
If you use this code, please cite the paper (full reference to be added on publication).

## License
MIT (see `LICENSE`).
