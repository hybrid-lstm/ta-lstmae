# TA-LSTMAE — Reproducibility Repository

Code accompanying the manuscript **"A Hybrid LSTM-Autoencoder Framework with Causal MTTD and Projected MTTR Evaluation for SIEM-Oriented Intrusion Detection"** (Y. Ariyanto, Y. W. Syaifudin, P. Y. Saputra, C. Setiadi, T. Fatmawati; submitted to the *International Journal of Intelligent Engineering and Systems*).

The repository regenerates every quantity reported in the paper from the official benchmark files. It also contains the experiments added in response to the second review round:

| Notebook | Content | Reviewer comments |
|---|---|---|
| `notebooks/01_reproduce_main_results.ipynb` | Main results (Tables 2–6 and the MTTD sensitivity analysis of Section 4.8), raw confusion matrices, causal MTTD as the primary metric, multi-seed ablation with 95% CIs and paired tests, MTTD sensitivity to tolerance and inter-arrival scale, threshold sweep, ECS alerts, environment file, SHA-256 manifest | R1-1, R1-2, R1-5, R1-6, R1-7, R1-8, R1-9, R2-5 |
| `notebooks/02_head_to_head_baselines.ipynb` | Table 7: controlled head-to-head comparison with **re-implemented HAE-HRL** and **ADFCNN-BiLSTM (2025)** on identical partitions, preprocessing, test targets, seeds and metrics (including MTTD for every model) | R1-4, R1-10, R2-1, R2-4 |
| `notebooks/03_end_to_end_latency.ipynb` | Table 8: event-by-event replay of the natively timestamped CICIDS-2018 stream with wall-clock timing of encode → window assembly → classify → ECS JSON → alert hand-off → automated response; queueing under 1×/10×/100× load. Elasticsearch/Logstash sinks are included; the reported run used the NDJSON file hand-off because no SIEM cluster was available, so its end-to-end values are lower bounds | R1-1, R1-3, R2-3, R2-6 |

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

Start with `MODE = "smoke"` (synthetic data; it only checks that the code runs), then use `"quick"` (3 seeds) or `"full"` (5 seeds). The results reported in the manuscript use three seeds (42, 43, 44), as stored in `outputs/run_config.json`. Interrupted runs resume automatically, and finished models and seeds are skipped. Progress is also written to `outputs/progress.log`.

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
2. Run **02**. This is the most compute-intensive step (5 models × 3 seeds × 3 datasets in the reported run).
3. Start the ELK testbed (`cd docker && docker compose up -d`), set `ES_URL=http://localhost:9200` and `LOGSTASH_URL=http://localhost:8080`, then run **03**.

To check the code without the datasets, set `SMOKE_TEST = True` in the first cell of any notebook. It uses small synthetic data and writes to `outputs_SMOKE_TEST_INVALID/`. Its numbers are meaningless and must never be reported.

## Protocol guarantees
* **No silent data substitution.** A missing dataset file raises an error. Each input file's SHA-256 is recorded in `outputs/MANIFEST.json`.
* **Leakage-free windowing.** The train/validation split uses contiguous blocks and is made *before* windowing. Windows are built inside each split, and the audit asserts contiguity, disjointness and monotone timestamps.
* **No resampling.** Class imbalance is handled with `class_weight` computed from the training split, for every model including the baselines.
* **Common evaluation.** Every model is scored on the same test targets (the final events of the TA-LSTMAE windows) with the same threshold and the same metrics engine.
* **Causal MTTD is primary.** The symmetric ±5-event rule is reported only as a sensitivity analysis.
* **Projected vs. measured response.** `pMTTR = MTTD + 30 s` is a configurable projection. Notebook 03 measures a testbed **automated-containment latency**, which is not a human incident-response time.

## Mapping of output files to the manuscript
| Manuscript | File(s) in `outputs/` |
|---|---|
| Table 2 | `tables/table2_tae_parameters.csv` |
| Table 3 | `tables/table3_dataset_summary.csv` |
| Table 4 (TA-LSTMAE row, seed 42) | `tables/table4_ta_lstmae_seed42.csv` |
| Table 5 (seed 42) | `tables/table5_temporal_metrics_causal_primary.csv` |
| Table 6 (mean ± SD over seeds 42–44, 95% CI) | `tables/table6_ablation_mean_sd_ci.csv`, `ablation_paired_tests.csv`, `ablation_runs.csv` |
| Table 7 (mean ± SD over seeds 42–44) | `head_to_head_summary.csv`, `head_to_head_paired_tests.csv`, `head_to_head_mcnemar_seed42.csv`, `head_to_head_runs.csv` |
| Table 8 | `latency/latency_summary.csv`, `latency/latency_load_test.csv`, `latency/throughput.csv` |
| Section 4.8 (MTTD sensitivity) | `mttd_sensitivity.csv` |
| Threshold sweep (Section 4.3) | `threshold_sweep.csv` |

Tables 4 and 5 report the seed-42 run, which also provides the confusion matrices and the latency replay; Table 7 reports means over seeds 42–44. The ablation (Table 6) uses a common budget of eight epochs, whereas the main model uses the per-dataset schedule in `config.py`.

## Implementation notes
* `models.py` reproduces every parameter count reported in the manuscript (Table 2 and Section 4.6: 28,833 / 81,089 / 104,001 / 128,033 / 179,617 / 261,025 / 227,233); `tests/test_models.py` checks them. Three layout details are not spelled out in the manuscript and are documented here:
  - the decoder input projection Dense(64)+Reshape(4×16) that feeds the decoder BiGRU;
  - the gated-fusion head of the HBLT;
  - the BatchNorm layer in the CNN branch.
* The baselines (HAE-HRL and ADFCNN-BiLSTM) are re-implemented from their published descriptions. Hyperparameters that the original papers do not report were fixed before testing and are listed in `talstmae/baselines.py::BASELINE_NOTES`.
* The synthetic timestamp seeds (1001/1002) and the one-second resolution used for UNSW-NB15 are set in `config.py`.

## Citation
If you use this code, please cite the paper (full reference to be added on publication).

## License
MIT (see `LICENSE`).
