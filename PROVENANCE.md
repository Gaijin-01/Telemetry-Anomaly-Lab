# Provenance

## Acknowledgements / Inspiration

This project was inspired by and informed by the public repository [`IndraniB2020/All_machine_learning`](https://github.com/IndraniB2020/All_machine_learning), particularly its anomaly-detection examples. Telemetry Anomaly Lab is an independent clean-room implementation. The architecture, validation rules, result model, CLI, tests, documentation, and telemetry workflow were designed and implemented separately. No substantial source code, notebook text, comments, datasets, or project structure from that repository are redistributed here. When reviewed, the upstream repository had no explicit software license; it was treated strictly as reference and inspiration, not reusable source code. Credit for the initial inspiration goes to IndraniB2020.

The project uses general algorithm concepts (Isolation Forest, density-based clustering/noise, and median/MAD robust statistics) and independently designed data validation, score semantics, result model, CLI, tests, and documentation. No substantial upstream source, comments, notebook prose, or data files have been copied. The upstream clone is stored outside this repository.

Actual algorithm implementations use documented APIs from `scikit-learn` (`IsolationForest`, `StandardScaler`, and `DBSCAN`); array/data handling uses NumPy and pandas. The robust MAD detector is an independent implementation of the published mathematical definition using NumPy.
