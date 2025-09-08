import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import log_loss

# -----------------------------
# 1) Prepare data
# -----------------------------
# Example column choices — adjust to your schema
A_col   = "arm"           # 0,1,2,3  (0 = control)
Y_col   = "Y"             # your outcome (Δ spending or post)
X_cols  = ["month", "income_band", "B", "limit", "extralimit", "eligibility"]  # add more if needed

df = df.dropna(subset=[A_col, Y_col] + X_cols).copy()
df[A_col] = df[A_col].astype(int)

# Identify categorical vs numeric
cat_cols = [c for c in X_cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
num_cols = [c for c in X_cols if c not in cat_cols]

# Encoder for X
encoder = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop"
)

# -----------------------------
# 2) Multinomial propensity e_a(X) for all arms
# -----------------------------
prop_model = Pipeline(steps=[
    ("x", encoder),
    ("clf", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=2000))
])
prop_model.fit(df[X_cols], df[A_col])

# Predicted probabilities e_hat: shape (n, K)
e_hat = prop_model.predict_proba(df[X_cols])  # columns ordered by sorted classes
classes_ = prop_model.named_steps["clf"].classes_
class_index = {a:i for i,a in enumerate(classes_)}
A_idx = df[A_col].map(class_index).to_numpy()
n, K = e_hat.shape  # K = #arms (incl. control)

# -----------------------------
# 3) Common support trimming
#    keep units where all arms are plausible (min e_a >= eps)
# -----------------------------
eps = 0.02  # tune as needed (0.01–0.05 typical)
keep = (e_hat.min(axis=1) >= eps)
df_trim = df.loc[keep].copy()
e_hat = e_hat[keep]
A_idx = A_idx[keep]
print(f"Kept {keep.mean():.1%} of units after common-support trimming.")

# -----------------------------
# 4) Weights
# -----------------------------
# (a) Stabilized multinomial IPTW: w_i = p_{A_i} / e_{A_i}(X_i)
p_marg = np.bincount(df_trim[A_col], minlength=K) / len(df_trim)
w_iptw = p_marg[A_idx] / e_hat[np.arange(len(df_trim)), A_idx]

# (b) Generalized Overlap Weights (multi-arm): w_i ∝ 1 - e_{A_i}(X_i)
w_gow = 1.0 - e_hat[np.arange(len(df_trim)), A_idx]
# normalize to mean 1 (optional)
w_gow = w_gow * (len(w_gow) / w_gow.sum())

# Choose which weight to use for balance diagnostics (try both)
w_balance = w_gow

# -----------------------------
# 5) Weighted SMDs across arms (pairwise vs control, per feature)
#    We compute SMDs on the *encoded* X to see balance
# -----------------------------
# Build encoded X matrix (trimmed)
X_enc = encoder.fit_transform(df[X_cols])  # fit on full before trimming for reproducibility
X_enc = encoder.transform(df_trim[X_cols]) # then transform the trimmed
X_enc = np.asarray(X_enc)

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def weighted_var(x, w):
    mu = weighted_mean(x, w)
    return np.sum(w * (x - mu)**2) / np.sum(w)

def pairwise_smd(x, g1_mask, g0_mask, w):
    # pooled SD with weights
    v1 = weighted_var(x[g1_mask], w[g1_mask])
    v0 = weighted_var(x[g0_mask], w[g0_mask])
    sp = np.sqrt(0.5 * (v1 + v0) + 1e-12)
    m1 = weighted_mean(x[g1_mask], w[g1_mask])
    m0 = weighted_mean(x[g0_mask], w[g0_mask])
    return (m1 - m0) / sp

arm_vals = classes_
ctrl = 0  # value for control arm
ctrl_mask = (df_trim[A_col].to_numpy() == ctrl)

smd_report = {}
for a in arm_vals:
    if a == ctrl:
        continue
    a_mask = (df_trim[A_col].to_numpy() == a)
    smds = np.array([pairwise_smd(X_enc[:,j], a_mask, ctrl_mask, w_balance) for j in range(X_enc.shape[1])])
    smd_report[a] = pd.Series(smds).abs().quantile([0.5, 0.9, 0.99]).rename(index={0.5:"median|SMD|",0.9:"p90|SMD|",0.99:"p99|SMD|"})
balance_table = pd.DataFrame(smd_report)
print("\nWeighted |SMD| quantiles vs control (using overlap weights):\n", balance_table.round(3))

# -----------------------------
# 6) Doubly-Robust AIPW estimates per arm vs control
# -----------------------------
# Outcome models m_a(X) for each arm (on trimmed sample)
models = {}
for a in arm_vals:
    mask = (df_trim[A_col] == a)
    # simple, robust regressor; replace by LightGBM/XGB if you prefer
    m = Pipeline(steps=[("x", encoder), ("gb", GradientBoostingRegressor(random_state=42))])
    m.fit(df_trim.loc[mask, X_cols], df_trim.loc[mask, Y_col])
    models[a] = m

# Predict m_a(X) for all units
m_hat = {a: models[a].predict(df_trim[X_cols]) for a in arm_vals}

# Pull needed pieces
Y = df_trim[Y_col].to_numpy()
e = e_hat  # alias
idx_ctrl = class_index[ctrl]

effects = {}
for a in arm_vals:
    if a == ctrl:
        continue
    idx_a = class_index[a]
    # AIPW influence term
    term = (m_hat[a] - m_hat[ctrl]
            + ( (df_trim[A_col].to_numpy()==a)   * (Y - m_hat[a])   / (e[:, idx_a] + 1e-12) )
            - ( (df_trim[A_col].to_numpy()==ctrl)* (Y - m_hat[ctrl]) / (e[:, idx_ctrl] + 1e-12) ))
    tau_hat = term.mean()
    # simple SE via nonparametric bootstrap (fast-ish; adjust B if needed)
    # (Optional) comment out if you don't want SEs
    B = 200
    rng = np.random.default_rng(123)
    boots = []
    n = len(term)
    for _ in range(B):
        ix = rng.integers(0, n, n)
        boots.append(term[ix].mean())
    se = np.std(boots, ddof=1)
    effects[a] = {"tau_hat": tau_hat, "se": se}

effects_df = pd.DataFrame(effects).T
effects_df.index.name = "arm_vs_control"
print("\nAIPW estimates (arm vs control) on trimmed overlap sample:")
print(effects_df.round(4))

