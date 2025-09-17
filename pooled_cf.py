# ===========
# REQUIREMENTS
# pip install econml lightgbm scikit-learn pandas numpy
# ===========

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from lightgbm import LGBMRegressor
from econml.dml import CausalForestDML
from sklearn.model_selection import StratifiedKFold

# --------------------------------------------
# 0) INPUTS (replace with your columns/arrays)
# --------------------------------------------
# DataFrame with columns: Y (outcome), T (0/1), S (stratum label), X... (features), Y0 (strong baseline e.g. pre-period spend)
# Y can be heavy-tailed; if so consider log1p(Y) or Huber loss inside models_y
df = ...  # your dataframe
y  = df["Y"].to_numpy()
t  = df["T"].to_numpy().astype(int)
S  = df["S"].to_numpy()          # categorical labels {0..7} or strings
X  = df.drop(columns=["Y","T","S"]).copy()   # keep baseline features, including Y0

# Known randomization rates by stratum (length = number of strata).
# Fill with your actual assignment probs, e.g., p_vec = { (income,risk): 0.5, ... }
unique_S = pd.Series(S).astype("category").cat.categories
p_map = {s: 0.5 for s in unique_S}                         # <-- EDIT with your design
p_s = np.array([p_map[s] for s in pd.Series(S).astype("category")])

# ---------------------------------------------------
# 1) CUPED: residualize Y on strong baseline(s) Y0...
# ---------------------------------------------------
def cuped_residual(y, X_base):
    """Return CUPED-residualized outcome and theta."""
    # Use a simple linear reg on strong baseline(s). You can swap to Ridge if needed.
    reg = LinearRegression()
    reg.fit(X_base, y)
    y_hat = reg.predict(X_base)
    theta = 1.0  # implicit in regression residualization form
    y_star = y - y_hat   # CUPED residual
    return y_star, reg

# Choose your CUPED baselines (strong predictors): e.g., ['Y0', 'visits_30d', 'spend_90d']
cuped_cols = [c for c in X.columns if c.lower() in {"y0"} or "baseline" in c.lower()]
if len(cuped_cols) == 0:
    # fallback: no CUPED; set y_star = y
    y_star = y.copy()
    cuped_model = None
else:
    y_star, cuped_model = cuped_residual(y, X[cuped_cols].to_numpy())

# -----------------------------------------------------------
# 2) Build a design matrix with ONE-HOT strata inside features
# -----------------------------------------------------------
# Identify categorical columns (including S) and numeric columns
X_aug = X.copy()
X_aug["__STRATUM__"] = pd.Series(S).astype("category")

cat_cols = ["__STRATUM__"] + [c for c in X_aug.columns if X_aug[c].dtype == "object" or str(X_aug[c].dtype).startswith("category")]
num_cols = [c for c in X_aug.columns if c not in cat_cols]

preproc = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

X_proc = preproc.fit_transform(X_aug)

# -----------------------------------------------------------
# 3) CausalForestDML (pooled, orthogonalized, conservative)
# -----------------------------------------------------------
# Outcome model (robust regressor). Consider robust params for heavy tails.
model_y = LGBMRegressor(
    n_estimators=600,
    learning_rate=0.05,
    min_child_samples=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
# Propensity: since you know p_s, you can pass a "fixed" model_t by pre-supplying ps in fit(). EconML allows
# either model_t or propensities. We'll pass propensities directly below.

# NOTE: If your platform/econml version prefers model_t, you can set:
# model_t = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)

cf = CausalForestDML(
    model_y=model_y,
    model_t=None,                 # passing propensities directly
    n_crossfit_folds=3,
    n_estimators=2000,
    min_samples_leaf=200,         # ↑ for stability in low SNR
    max_depth=None,               # let leaf size control complexity
    subsample_fr=0.7,             # honesty + subsampling
    random_state=42
)

# Fit on CUPED residuals (or raw y if you skipped CUPED).
cf.fit(Y=y_star, T=t, X=X_proc, W=None, cache_values=True, sample_weight=None, inference=None, propensity=p_s)

# --------------------------
# 4) Unit-level CATEs (τ̂_i)
# --------------------------
tau_hat = cf.effect(X_proc)  # out-of-bag/cross-fit style

# Per-stratum CATE by averaging τ̂_i in each stratum (optional weights = 1 here)
df_tau = pd.DataFrame({"S": pd.Series(S).astype("category"), "tau": tau_hat})
cate_by_stratum_pred = df_tau.groupby("S")["tau"].mean()

# --------------------------------------------
# 5) AIPW / DR GATEs by stratum + closed-form SE
# --------------------------------------------
# Retrieve outcome models from the DML (or refit your own model_y for m0/m1).
# EconML stores cross-fitted models; to keep it simple we use its internal predictions:
m1x = cf.models_y[0].predict(X_proc) if hasattr(cf, "models_y") else model_y.fit(X_proc[t==1], y_star[t==1]).predict(X_proc)
m0x = cf.models_y[1].predict(X_proc) if hasattr(cf, "models_y") else model_y.fit(X_proc[t==0], y_star[t==0]).predict(X_proc)

# AIPW orthogonal score on CUPED outcome
psi = (m1x - m0x) + t*(y_star - m1x)/p_s - (1 - t)*(y_star - m0x)/(1 - p_s)

gate_mean = pd.Series(psi).groupby(pd.Series(S).astype("category")).mean()
gate_se   = pd.Series(psi).groupby(pd.Series(S).astype("category")).std(ddof=1) / \
            np.sqrt(pd.Series(S).astype("category").value_counts().sort_index())

gate_ci_lo = gate_mean - 1.96 * gate_se
gate_ci_hi = gate_mean + 1.96 * gate_se

gate_table = pd.DataFrame({
    "GATE_DR": gate_mean,
    "SE": gate_se,
    "CI95_lo": gate_ci_lo,
    "CI95_hi": gate_ci_hi
})

# ----------------------------------------------------
# 6) Overall ATE (DR) and its SE (sample-weighted mean)
# ----------------------------------------------------
ATE_DR = psi.mean()
ATE_SE = psi.std(ddof=1) / np.sqrt(len(psi))
ATE_CI = (ATE_DR - 1.96*ATE_SE, ATE_DR + 1.96*ATE_SE)

# ----------------------------------------------------
# 7) Optional: stratified bootstrap (predictions only)
# ----------------------------------------------------
def stratified_bootstrap_gate(df, S, psi, B=500, random_state=123):
    rng = np.random.default_rng(random_state)
    categories = pd.Series(S).astype("category").cat.categories
    idx_by_s = {s: np.where(pd.Series(S).astype("category")==s)[0] for s in categories}
    boot = []
    for _ in range(B):
        psi_b = []
        for s in categories:
            idx = idx_by_s[s]
            resampled = rng.choice(idx, size=len(idx), replace=True)
            psi_b.append(pd.Series(psi[resampled]).mean())
        boot.append(psi_b)
    boot = np.array(boot)
    lo = np.quantile(boot, 0.025, axis=0)
    hi = np.quantile(boot, 0.975, axis=0)
    return pd.DataFrame({"S": categories, "boot_CI95_lo": lo, "boot_CI95_hi": hi})

# Example:
# boot_gate = stratified_bootstrap_gate(df, S, psi, B=300)

# ----------------------------------------------------
# 8) Results
# ----------------------------------------------------
print("\n=== Overall ATE (DR on CUPED outcome) ===")
print(f"ATE_DR = {ATE_DR:.4f}  (SE {ATE_SE:.4f})  95% CI {ATE_CI}")

print("\n=== Stratum GATEs (AIPW/DR, CUPED) ===")
print(gate_table)

print("\n=== Stratum mean CATEs from pooled CF (for exploration/policy) ===")
print(cate_by_stratum_pred)

# df has columns: 'S' (stratum), 'T' in {0,1}
counts = df.groupby('S')['T'].agg(n_s='count', n1_s='sum')
counts['p_s_empirical'] = counts['n1_s'] / counts['n_s']

# If you know the design targets per stratum, build a map:
p_design = {'(inc1,lowrisk)': 0.5, '(inc1,highrisk)': 0.4, ...}  # fill yours
# Vector for each row in df:
df['p_s'] = df['S'].map(p_design)  # preferred
# Or fallback to empirical:
df['p_s'] = df['S'].map(counts['p_s_empirical'])
