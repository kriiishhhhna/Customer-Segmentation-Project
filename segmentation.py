"""
Customer segmentation pipeline.

Pipeline:
    1. Load & clean the customer data.
    2. Engineer + scale features (demographics + RFM-style behavior).
    3. Choose k with the elbow method and silhouette score.
    4. Fit K-Means and label each customer with a segment.
    5. Profile each segment and save charts + a summary table.

Run (after generate_data.py):
    python segmentation.py
Outputs land in ./outputs/.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "customers.csv")
OUT_DIR = os.path.join(HERE, "outputs")

# Features fed into the clustering model. Mix of demographics and behavior.
FEATURES = [
    "age",
    "annual_income_k",
    "tenure_months",
    "recency_days",
    "frequency_yr",
    "avg_order_value",
    "monetary_total",
    "online_ratio",
]

sns.set_theme(style="whitegrid")


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Median-impute the small number of missing incomes.
    df["annual_income_k"] = df["annual_income_k"].fillna(df["annual_income_k"].median())
    df = df.drop_duplicates(subset="customer_id")
    return df


def choose_k(X: np.ndarray, k_range=range(2, 11)) -> tuple[int, pd.DataFrame]:
    """Compute inertia (elbow) and silhouette for a range of k; pick best silhouette."""
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        records.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
        })
    scores = pd.DataFrame(records)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    return best_k, scores


def plot_diagnostics(scores: pd.DataFrame, best_k: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(scores["k"], scores["inertia"], "o-")
    axes[0].axvline(best_k, color="crimson", ls="--", alpha=0.7)
    axes[0].set(title="Elbow method", xlabel="k (clusters)", ylabel="Inertia")

    axes[1].plot(scores["k"], scores["silhouette"], "o-", color="seagreen")
    axes[1].axvline(best_k, color="crimson", ls="--", alpha=0.7)
    axes[1].set(title="Silhouette score", xlabel="k (clusters)", ylabel="Score")
    fig.suptitle(f"Choosing k  (selected k = {best_k})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_choose_k.png"), dpi=130)
    plt.close(fig)


def plot_pca_scatter(X: np.ndarray, labels: np.ndarray) -> None:
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels,
                    palette="tab10", s=28, alpha=0.8, ax=ax, legend="full")
    ax.set(title="Customer segments (PCA projection)",
           xlabel="PC 1", ylabel="PC 2")
    ax.legend(title="Segment")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_segments_pca.png"), dpi=130)
    plt.close(fig)


def plot_profiles(df: pd.DataFrame) -> None:
    """Heatmap of mean feature values per segment (z-scored for comparability)."""
    means = df.groupby("segment")[FEATURES].mean()
    z = (means - means.mean()) / means.std()
    fig, ax = plt.subplots(figsize=(10, 0.8 * len(means) + 2))
    sns.heatmap(z, annot=means.round(1), fmt=".1f", cmap="RdBu_r", center=0,
                cbar_kws={"label": "z-score vs. overall mean"}, ax=ax)
    ax.set(title="Segment profiles (cell text = actual mean)", ylabel="Segment")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_segment_profiles.png"), dpi=130)
    plt.close(fig)


def name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Derive a distinct, descriptive label for each segment from its profile.

    Rules are applied in priority order against thresholds drawn from the
    spread of segment means, so well-separated segments get distinct names.
    """
    spend_hi = profile["monetary_total"].quantile(0.66)
    recency_hi = profile["recency_days"].quantile(0.66)
    freq_hi = profile["frequency_yr"].median()
    aov_hi = profile["avg_order_value"].quantile(0.66)
    age_hi = profile["age"].quantile(0.75)
    online_hi = profile["online_ratio"].median()

    names = {}
    for seg, row in profile.iterrows():
        if row["monetary_total"] >= spend_hi and row["frequency_yr"] >= freq_hi:
            label = "VIP / Champions"
        elif row["avg_order_value"] >= aov_hi and row["frequency_yr"] <= freq_hi:
            # Buys high-value items but rarely — looks "lapsed" by recency,
            # yet is valuable. Check this before the lapsed rule.
            label = "Big-Ticket Occasional"
        elif row["recency_days"] >= recency_hi and row["frequency_yr"] <= freq_hi:
            label = "At-Risk / Lapsed"
        elif row["age"] >= age_hi and row["online_ratio"] < online_hi:
            label = "Senior Value Seekers"
        elif row["frequency_yr"] >= freq_hi and row["online_ratio"] >= online_hi:
            label = "Loyal Online Regulars"
        else:
            label = "Mid-Market Regulars"
        names[seg] = f"{seg}: {label}"
    return names


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_and_clean()
    print(f"Loaded {len(df):,} customers, {len(FEATURES)} features.")

    X = StandardScaler().fit_transform(df[FEATURES])

    best_k, scores = choose_k(X)
    print("\nk-selection diagnostics:")
    print(scores.to_string(index=False))
    print(f"\nSelected k = {best_k} (highest silhouette).")
    plot_diagnostics(scores, best_k)

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    df["segment"] = km.fit_predict(X)

    profile = df.groupby("segment")[FEATURES].mean()
    df["segment"] = df["segment"].map(name_segments(profile))

    plot_pca_scatter(X, df["segment"].values)
    plot_profiles(df)

    # Save a tidy summary table.
    summary = (
        df.groupby("segment")
        .agg(customers=("customer_id", "size"),
             pct=("customer_id", lambda s: round(100 * len(s) / len(df), 1)),
             avg_age=("age", "mean"),
             avg_income_k=("annual_income_k", "mean"),
             avg_recency_days=("recency_days", "mean"),
             avg_frequency_yr=("frequency_yr", "mean"),
             avg_spend=("monetary_total", "mean"),
             online_ratio=("online_ratio", "mean"))
        .round(1)
        .sort_values("avg_spend", ascending=False)
    )
    summary.to_csv(os.path.join(OUT_DIR, "segment_summary.csv"))
    df.to_csv(os.path.join(OUT_DIR, "customers_segmented.csv"), index=False)

    print("\nSegment summary:")
    print(summary.to_string())

    # Quick sanity check: how well do discovered segments line up with the
    # hidden personas the data was generated from?
    if "_true_persona" in df.columns:
        ct = pd.crosstab(df["segment"], df["_true_persona"])
        print("\nDiscovered segment vs. true persona (rows=segment):")
        print(ct.to_string())

    print(f"\nDone. Charts + tables written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
