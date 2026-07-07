# Customer Segmentation

Segment customers by **behavior** (recency, frequency, monetary spend, channel
preference) and **demographics** (age, income, region, tenure) using K-Means
clustering, then translate the clusters into named, actionable segments.

## What's inside

| File | Purpose |
|---|---|
| `generate_data.py` | Creates a realistic synthetic dataset of 2,000 customers (`data/customers.csv`) from hidden personas. |
| `segmentation.py` | End-to-end pipeline: clean → scale → choose `k` → cluster → profile → save charts & tables. |
| `analysis.ipynb` | Notebook walkthrough of the same analysis with inline plots and an action plan. |
| `build_notebook.py` | Regenerates `analysis.ipynb` from source (keeps it diff-friendly). |
| `outputs/` | Generated charts (`.png`), the segmented customer table, and a segment summary. |

## Quick start

```bash
pip install -r requirements.txt
python generate_data.py     # -> data/customers.csv
python segmentation.py      # -> outputs/ (charts + summary)
# or, for the interactive version:
jupyter notebook analysis.ipynb
```

## Method

1. **Clean** — median-impute missing income, drop duplicate IDs.
2. **Scale** — `StandardScaler`, since K-Means is distance-based.
3. **Choose k** — sweep k = 2..10, pick the highest **silhouette** score
   (cross-checked against the **elbow** of inertia). On the default data, k = 5.
4. **Cluster** — `KMeans(n_clusters=k, random_state=42)`.
5. **Profile & name** — rank each cluster on spend / recency / frequency / order
   value / age / channel to assign a descriptive label.
6. **Visualize** — PCA scatter of the segments + a z-scored profile heatmap.

## Segments found (default seed)

| Segment | Share | Avg spend | Profile |
|---|---|---|---|
| VIP / Champions | ~15% | highest | High income, frequent, recent |
| Big-Ticket Occasional | ~13% | high | Large but infrequent orders |
| Loyal Online Regulars | ~46% | mid | Younger, frequent, online-first |
| Senior Value Seekers | ~11% | mid | Older, offline, value-driven |
| At-Risk / Lapsed | ~15% | lowest | No purchase in a long time |

## Validation

The synthetic data is generated from six hidden personas the model never sees.
Both scripts print a crosstab of *discovered segment* vs. *true persona* — on the
default seed the recovered segments line up almost one-to-one, confirming the
pipeline finds real structure rather than noise.

## Adapting to your own data

Point `DATA_PATH` in `segmentation.py` at your CSV and update the `FEATURES`
list to your columns. Keep features numeric (encode categoricals first) and
re-run — the rest of the pipeline is column-agnostic.
