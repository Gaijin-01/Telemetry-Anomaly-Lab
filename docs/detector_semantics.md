# Detector semantics

## Isolation Forest
Uses `sklearn.ensemble.IsolationForest` after `StandardScaler`. Default random seed is 42; configuration records seed, estimator count, and contamination. The boolean flag follows the estimator's `predict` result (`-1` means anomaly). The numeric score is `-score_samples`, so larger means more anomalous. This is a relative model score, not a probability or calibrated severity. Scaling is explicit and recorded even though Isolation Forest is less distance-sensitive than DBSCAN.

## DBSCAN
Uses `StandardScaler` followed by sklearn DBSCAN. Cluster labels are integer labels; `-1` represents density noise and is the only label treated as anomalous. A continuous anomaly score is intentionally absent (`null` in JSON); no fake precision is implied. Labels and noise membership depend on eps, min_samples, selected features, and the population. Cluster label numbers are arbitrary. Silhouette is not used to turn labels into an anomaly score.

## Robust Z-score
For each active feature, center is the median and scale is `1.4826 * MAD`, where MAD is median absolute deviation from the median. For positive MAD, feature deviation is absolute difference divided by this scale. Row score is the maximum across features, and anomaly means strictly greater than threshold (default 3.5). When MAD is exactly zero, equal-to-median values contribute zero; any unequal value contributes a capped ordinal value `threshold + 1`. It flags deviation while remaining finite/JSON-safe, but does not express its magnitude. The row score is not a probability.

## Shared interpretation
A reported anomaly means only “unusual under this input, feature selection, detector, and parameters.” It does not establish hardware failure, model regression, configuration error, or a causal explanation. Always inspect raw values and contextual run metadata.
