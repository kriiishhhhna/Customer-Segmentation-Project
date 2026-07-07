"""Generate analysis.ipynb from code cells (keeps the notebook in version-friendly form)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
"""# Customer Segmentation

Segment customers by **behavior** (RFM: recency, frequency, monetary) and
**demographics** (age, income, region, tenure, channel preference) using
K-Means clustering.

**Workflow:** load & clean → engineer/scale features → pick `k` (elbow +
silhouette) → cluster → profile & name segments → translate into action.

> Run `python generate_data.py` once first to create `data/customers.csv`.
"""))

c.append(nbf.v4.new_code_cell(
"""import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
sns.set_theme(style="whitegrid")

df = pd.read_csv("data/customers.csv")
print(df.shape)
df.head()"""))

c.append(nbf.v4.new_markdown_cell("## 1. Clean & explore"))
c.append(nbf.v4.new_code_cell(
"""# Median-impute missing income; drop dup IDs.
df["annual_income_k"] = df["annual_income_k"].fillna(df["annual_income_k"].median())
df = df.drop_duplicates(subset="customer_id")
df.describe().round(1)"""))

c.append(nbf.v4.new_code_cell(
"""# Distributions of the behavioral features.
behav = ["recency_days", "frequency_yr", "monetary_total", "avg_order_value"]
df[behav].hist(bins=30, figsize=(10, 6)); plt.tight_layout(); plt.show()"""))

c.append(nbf.v4.new_markdown_cell("## 2. Feature scaling\nK-Means uses Euclidean distance, so features must be on a comparable scale."))
c.append(nbf.v4.new_code_cell(
"""FEATURES = ["age", "annual_income_k", "tenure_months", "recency_days",
            "frequency_yr", "avg_order_value", "monetary_total", "online_ratio"]
X = StandardScaler().fit_transform(df[FEATURES])
X.shape"""))

c.append(nbf.v4.new_markdown_cell("## 3. Choose k — elbow + silhouette"))
c.append(nbf.v4.new_code_cell(
"""rows = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    lab = km.fit_predict(X)
    rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, lab)})
scores = pd.DataFrame(rows)
best_k = int(scores.loc[scores.silhouette.idxmax(), "k"])

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(scores.k, scores.inertia, "o-"); ax[0].set(title="Elbow", xlabel="k", ylabel="inertia")
ax[1].plot(scores.k, scores.silhouette, "o-", color="seagreen"); ax[1].set(title="Silhouette", xlabel="k")
for a in ax: a.axvline(best_k, color="crimson", ls="--")
plt.show()
print("Selected k =", best_k)"""))

c.append(nbf.v4.new_markdown_cell("## 4. Fit K-Means"))
c.append(nbf.v4.new_code_cell(
"""km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
df["cluster"] = km.fit_predict(X)
df["cluster"].value_counts().sort_index()"""))

c.append(nbf.v4.new_markdown_cell("## 5. Visualize segments (PCA to 2D)"))
c.append(nbf.v4.new_code_cell(
"""coords = PCA(n_components=2, random_state=42).fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=df["cluster"],
                palette="tab10", s=28, alpha=0.8)
plt.title("Customer segments (PCA projection)"); plt.xlabel("PC 1"); plt.ylabel("PC 2")
plt.legend(title="cluster"); plt.show()"""))

c.append(nbf.v4.new_markdown_cell("## 6. Profile & name the segments"))
c.append(nbf.v4.new_code_cell(
"""profile = df.groupby("cluster")[FEATURES].mean()

def label(row):
    spend_hi, rec_hi = profile.monetary_total.quantile(.66), profile.recency_days.quantile(.66)
    freq_hi, aov_hi = profile.frequency_yr.median(), profile.avg_order_value.quantile(.66)
    age_hi, on_hi = profile.age.quantile(.75), profile.online_ratio.median()
    if row.monetary_total >= spend_hi and row.frequency_yr >= freq_hi: return "VIP / Champions"
    if row.avg_order_value >= aov_hi and row.frequency_yr <= freq_hi:   return "Big-Ticket Occasional"
    if row.recency_days >= rec_hi and row.frequency_yr <= freq_hi:      return "At-Risk / Lapsed"
    if row.age >= age_hi and row.online_ratio < on_hi:                  return "Senior Value Seekers"
    if row.frequency_yr >= freq_hi and row.online_ratio >= on_hi:       return "Loyal Online Regulars"
    return "Mid-Market Regulars"

names = {seg: f"{seg}: {label(r)}" for seg, r in profile.iterrows()}
df["segment"] = df["cluster"].map(names)

z = (profile - profile.mean()) / profile.std()
z.index = [names[i] for i in z.index]
plt.figure(figsize=(10, 5))
sns.heatmap(z, annot=profile.round(1).set_axis([names[i] for i in profile.index]),
            fmt=".1f", cmap="RdBu_r", center=0)
plt.title("Segment profiles (text = actual mean)"); plt.tight_layout(); plt.show()"""))

c.append(nbf.v4.new_code_cell(
"""summary = (df.groupby("segment")
    .agg(customers=("customer_id", "size"),
         pct=("customer_id", lambda s: round(100*len(s)/len(df), 1)),
         avg_age=("age", "mean"), avg_income_k=("annual_income_k", "mean"),
         avg_recency=("recency_days", "mean"), avg_freq=("frequency_yr", "mean"),
         avg_spend=("monetary_total", "mean"), online_ratio=("online_ratio", "mean"))
    .round(1).sort_values("avg_spend", ascending=False))
summary"""))

c.append(nbf.v4.new_markdown_cell(
"""## 7. From segments to action

| Segment | Who they are | Recommended play |
|---|---|---|
| **VIP / Champions** | High spend, frequent, high income | VIP perks, early access, referral asks |
| **Big-Ticket Occasional** | Large but rare orders | Targeted high-value launches, financing offers |
| **Loyal Online Regulars** | Frequent mid-value, online-first | Subscriptions, app engagement, cross-sell |
| **Senior Value Seekers** | Older, offline, value-driven | Loyalty discounts, in-store / mail outreach |
| **At-Risk / Lapsed** | Haven't bought in a long time | Win-back campaign, reactivation discount |

**Validation:** because the synthetic data was built from hidden personas, we can
check the clustering against ground truth:"""))

c.append(nbf.v4.new_code_cell(
"""if "_true_persona" in df.columns:
    display(pd.crosstab(df["segment"], df["_true_persona"]))"""))

nb["cells"] = c
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
with open("analysis.ipynb", "w") as f:
    nbf.write(nb, f)
print("Wrote analysis.ipynb")
