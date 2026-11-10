Place the official dataset files here (they are not redistributed in this repository):

    data/nsl-kdd/KDDTrain+.txt
    data/nsl-kdd/KDDTest+.txt
    data/unsw-nb15/UNSW_NB15_training-set.csv
    data/unsw-nb15/UNSW_NB15_testing-set.csv
    data/cicids2018/02-14-2018.csv

Sources: NSL-KDD (Canadian Institute for Cybersecurity, University of New Brunswick); UNSW-NB15 partitioned
training/testing sets (UNSW Canberra Cyber); CSE-CIC-IDS2018 "Processed Traffic Data for ML Algorithms"
(AWS Open Data Registry, s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv,
renamed to 02-14-2018.csv). The SHA-256 of every file used is recorded in outputs/MANIFEST.json.

Notes on the files:
* UNSW-NB15: in some distributions the 175,341-record file is named "testing-set" and the 82,332-record file
  "training-set". The loader detects this and always trains on the 175,341-record partition. Both partitions are
  ordered by class rather than by time, so the loader permutes each partition once with a fixed seed
  (Config.unsw_shuffle_seed) before synthetic timestamps and windows are created.
* CICIDS-2018: any file in data/cicids2018/ whose name contains "02-14-2018" (e.g. CSE-CIC-IDS2018_02-14-2018.csv)
  is accepted.
