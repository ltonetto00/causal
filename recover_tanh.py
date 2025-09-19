import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# ------------ inputs you already have ------------
# df: your dataframe with columns used below
# Z_post: np.array shape (n,) with tanh(Y_post/5000)
# tau_z:  np.array shape (n,) unit-level CATEs on tanh scale (from cf.effect(X_proc))
# X_proc: np.array shape (n, d) features used in CF (incl. stratum one-hots is fine)
# T:      np.array shape (n,) treatment indicator {0,1}
# S:      pandas Series with stratum labels (Categorical or string)

SCALE = 5000.0  # the divisor used in tanh(y/5000)

# 1) Fit a control-only model for Z_post to estimate mu0_z(x)=E[Z_post|T=0,X]
model_mu0Z = LGBMRegressor(
    n_estimators=600, learning_rate=0.05, min_child_samples=200,
    subsample=0.8, colsample_bytree=0.8, random_state=7
)
model_mu0Z.fit(X_proc[T == 0], Z_post[T == 0])
mu0_z = model_mu0Z.predict(X_proc)  # shape (n,)

# 2) Delta-method back-transform per unit:
#    tau_y_i ≈ 5000 * tau_z_i / (1 - mu0_z_i^2)
den = 1.0 - np.clip(mu0_z, -0.999999, 0.999999)**2
tau_y_unit = SCALE * tau_z / den

# 3) Aggregate to per-stratum CATE (GATE) and overall ATE
#    (Average the currency-scale unit effects within each stratum)
S_codes = pd.Series(S).astype("category")
gate_y = pd.Series(tau_y_unit).groupby(S_codes).mean()
gate_y.index = S_codes.cat.categories  # readable labels
ATE_Y = tau_y_unit.mean()

# 4) Optional: quick normal-approx 95% CIs for GATEs via delta method on the score
#    (For rigorous intervals, prefer a stratified bootstrap.)
se_gate = pd.Series(tau_y_unit).groupby(S_codes).std(ddof=1) / \
          np.sqrt(pd.Series(S_codes).value_counts().sort_index())
ci_lo = gate_y - 1.96 * se_gate
ci_hi = gate_y - 1.96 * se_gate * 0 + 1.96 * se_gate  # explicit for clarity

per_stratum_table = pd.DataFrame({
    "GATE_currency": gate_y,
    "SE_approx":     se_gate,
    "CI95_lo":       ci_lo,
    "CI95_hi":       ci_hi
})

print("\n=== Per-stratum CATE (currency) — delta method ===")
print(per_stratum_table)
print("\n=== Overall ATE (currency) — delta method ===")
print(f"ATE = {ATE_Y:.3f}")
