# Credit Card Customer Segmentation

Unsupervised segmentation of 10,128 credit card customers with K-means, including a diagnosed and corrected modelling failure.

Dataquest guided project, Data Scientist in Python path.

## The result

Three behavioural segments, `k=3` selected by silhouette maximum (0.1479).

| | Cluster 0: affluent but idle | Cluster 1: heavy spenders | Cluster 2: thin-limit revolvers |
|---|---|---|---|
| Customers | 3,669 (36.2%) | 1,496 (14.8%) | 4,962 (49.0%) |
| Estimated income | $78,343 (index 126) | $70,104 (113) | $47,632 (77) |
| Credit limit | $14,172 (164) | $14,693 (170) | $2,708 (31) |
| Annual spend | $2,816 (64) | $11,223 (255) | $3,523 (80) |
| Transactions | 54.7 (84) | 96.1 (148) | 62.9 (97) |
| Basket size | $50.79 (81) | $115.64 (185) | $55.37 (88) |
| Utilisation | 0.10 (36) | 0.16 (57) | 0.44 (160) |
| Products held | 4.24 (111) | 2.22 (58) | 3.97 (104) |

Index: 100 = portfolio average. Portfolio mean annual spend is $4,404.

**Cluster 2 is half the portfolio**, running 44% utilisation against an average limit of $2,708. Their transaction count sits at the portfolio norm, so the limit, not the appetite, is what caps their spend. Risk-screened line increases are the obvious test.

**Cluster 1 is 15% of customers and 2.6x the average spend**, with the worst cross-sell penetration in the book: 2.22 products held against a portfolio average of 3.8.

**Cluster 0 holds $14,172 of extended credit and uses 10% of it.** An activation problem, not a credit problem.

## The modelling failure, and the fix

The first model scored a respectable silhouette of 0.1435 at `k=6` and produced six clusters of plausible size. It was also, in four of those six, a one-hot encoding of marital status: clusters 1, 3, 4 and 5 were each 100% a single status, and mean marital purity across all six was 84.2%.

The cause is mechanical. After standardisation a one-hot column puts a fixed, large distance between two customers who differ on it, while a log-scaled continuous feature typically separates them by a fraction of a standard deviation. Euclidean distance has no way to know that one of those differences is less interesting, so K-means spends its partitions on the splits it can make cleanly.

Dropping the low-cardinality dummies from the distance metric, while keeping them in the dataset for description, improved every measure:

| | Model A (all features) | Model B (behaviour only) |
|---|---|---|
| Best silhouette | 0.1435 at k=6 | **0.1479 at k=3** |
| Variance in 2 PCA components | 30.1% | **38.8%** |
| Mean marital purity per cluster | 84.2% | **46.3%** (portfolio baseline) |
| Features in the distance metric | 23 | 13 |

Both models are kept in the notebook. The diagnosis is the part worth reading.

## Method

1. **Quality check.** 10,128 rows, no missing values, no duplicate `customer_id`.
2. **Feature engineering.** Education mapped to an ordinal scale with an explicit `Unknown` flag; gender binarised; marital status one-hot encoded. Derived features: average transaction value, utilisation in dollars, income-to-limit ratio. `log1p` applied to income, credit limit, spend, basket size and income-to-limit, all heavily right skewed.
3. **Standardisation**, since K-means minimises squared Euclidean distance and raw dollar columns would otherwise dominate.
4. **Selecting k** by sweeping 2 to 10 and reading the elbow curve against silhouette score, computed on a 5,000-row sample. `k` is chosen by the silhouette maximum in code, so it does not drift silently when the feature set changes.
5. **Profiling** each cluster as raw means and as an index against the portfolio average, plus crosstabs of the demographic fields deliberately excluded from the model.
6. **Naming** derived from the feature on which each cluster deviates furthest from the portfolio mean, in standard deviations, with a fallthrough so two clusters cannot take the same name. Reproducible from the data rather than hand-assigned.

## Limitations

- Silhouette near 0.15 is weak. These are dense regions on a continuum, not discrete customer types. Useful for targeting and sizing; not identities.
- K-means assumes roughly spherical, similarly sized clusters. A Gaussian mixture would allow elliptical shapes and soft assignment; HDBSCAN would allow a genuinely unclustered remainder. Both are the natural next test.
- `estimated_income` is estimated, not reported, so income-driven separation partly reflects whatever model produced it.
- The data is a snapshot. `months_inactive_12_mon` is a count, not a trend, so a customer winding down cannot be distinguished from one who was always quiet.
- No causal claim is made. The line-increase hypothesis for cluster 2 needs a controlled test.

## Stack

Python, pandas, NumPy, scikit-learn (KMeans, StandardScaler, PCA, silhouette_score), matplotlib, seaborn.

## Files

- `notebook.ipynb`: the full analysis with executed outputs and figures.
- The notebook writes `segment_profiles.csv` and `customer_segments.csv` on run.