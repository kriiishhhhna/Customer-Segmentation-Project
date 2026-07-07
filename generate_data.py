"""
Generate a realistic synthetic customer dataset for segmentation.

The generator builds the data from a set of hidden "true" personas (young
budget shoppers, affluent loyalists, lapsed customers, etc.). The clustering
pipeline never sees these personas — it has to rediscover them from behavior
and demographics, which is exactly the point of the exercise.

Run:
    python generate_data.py
Output:
    data/customers.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_CUSTOMERS = 2000

# Each persona: (weight, age range, income k$, recency days, frequency/yr,
#                avg order value $, online ratio 0-1)
PERSONAS = {
    "Young Budget Shopper":   (0.22, (18, 30), (20, 45),  (5, 60),   (4, 12),  (15, 45),  (0.75, 0.98)),
    "Affluent Loyalist":      (0.16, (35, 60), (90, 180), (1, 25),   (12, 30), (120, 350),(0.30, 0.60)),
    "Mid-Market Regular":     (0.24, (28, 50), (45, 85),  (10, 50),  (6, 14),  (40, 90),  (0.45, 0.75)),
    "Occasional Big Spender": (0.14, (30, 55), (70, 140), (40, 180), (1, 4),   (150, 400),(0.25, 0.55)),
    "Lapsed Customer":        (0.14, (25, 65), (30, 90),  (200, 540),(1, 5),   (25, 80),  (0.40, 0.80)),
    "Senior Value Seeker":    (0.10, (60, 80), (35, 70),  (15, 90),  (5, 11),  (30, 70),  (0.10, 0.40)),
}

GENDERS = np.array(["Female", "Male", "Other"])
GENDER_P = np.array([0.49, 0.49, 0.02])
REGIONS = np.array(["North", "South", "East", "West"])


def _uniform(rng, lo_hi, size):
    lo, hi = lo_hi
    return rng.uniform(lo, hi, size)


def generate(n: int = N_CUSTOMERS) -> pd.DataFrame:
    names = list(PERSONAS.keys())
    weights = np.array([PERSONAS[k][0] for k in names])
    weights = weights / weights.sum()
    assigned = RNG.choice(len(names), size=n, p=weights)

    rows = []
    for i in range(n):
        persona = names[assigned[i]]
        _, age_r, inc_r, rec_r, freq_r, aov_r, online_r = PERSONAS[persona]

        age = int(np.clip(_uniform(RNG, age_r, 1)[0] + RNG.normal(0, 2), 18, 85))
        income = float(max(12, _uniform(RNG, inc_r, 1)[0] + RNG.normal(0, 6)))
        recency = int(max(1, _uniform(RNG, rec_r, 1)[0] + RNG.normal(0, 8)))
        frequency = int(max(1, round(_uniform(RNG, freq_r, 1)[0] + RNG.normal(0, 1))))
        aov = float(max(5, _uniform(RNG, aov_r, 1)[0] + RNG.normal(0, 10)))
        online = float(np.clip(_uniform(RNG, online_r, 1)[0] + RNG.normal(0, 0.05), 0, 1))

        monetary = round(frequency * aov, 2)          # total annual spend
        tenure = int(np.clip(RNG.normal(36, 24), 1, 180))  # months as a customer

        rows.append({
            "customer_id": 100000 + i,
            "age": age,
            "gender": RNG.choice(GENDERS, p=GENDER_P),
            "region": RNG.choice(REGIONS),
            "annual_income_k": round(income, 1),
            "tenure_months": tenure,
            "recency_days": recency,                  # days since last purchase
            "frequency_yr": frequency,                # purchases per year
            "avg_order_value": round(aov, 2),
            "monetary_total": monetary,               # total annual spend
            "online_ratio": round(online, 3),         # share of orders placed online
            "_true_persona": persona,                 # ground truth (not used for clustering)
        })

    df = pd.DataFrame(rows)
    # Inject a few realistic missing values to practice cleaning.
    miss_idx = RNG.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[miss_idx, "annual_income_k"] = np.nan
    return df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "data")
    os.makedirs(out_dir, exist_ok=True)
    df = generate()
    path = os.path.join(out_dir, "customers.csv")
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} customers -> {path}")
    print(df.drop(columns="_true_persona").head())


if __name__ == "__main__":
    main()
