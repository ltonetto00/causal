import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

SCALE = 5000.0

# --- Utility: safe inverse tanh ---
def atanh_clip(z, eps=1e-7):
    z = np.clip(z, -1+eps, 1-eps)
    return 0.5*np.log((1+z)/(1-z))

# --- Step 1: Fit arm- and stratum-aware models for E[Z_post | T=t, X] ---
# Control model (t=0) for mu0_z; treated model can be built either directly or via mu0_z + tau_z
def fit_arm_models_Z(X_proc, Z_post, T):
    m0 = LGBMRegressor(
        n_estimators=700, learning_rate=0.05,
        min_child_samples=200, subsample=0.8, colsample_bytree=0.8,
        random_state=10
    )
    m1 = LGBMRegressor(
        n_estimators=700, learning_rate=0.05,
        min_child_samples=200, subsample=0.8, colsample_bytree=0.8,
        random_state=11
    )
    m0.fit(X_proc[T==0], Z_post[T==0])
    m1.fit(X_proc[T==1], Z_post[T==1])
    return m0, m1

# Predict arm means on Z scale
def predict_muZ(m0, m1, X_proc, tau_z=None, use_tau_bridge=True):
    mu0_z = m0.predict(X_proc)
    if use_tau_bridge and tau_z is not None:
        mu1_z = mu0_z + tau_z
    else:
        mu1_z = m1.predict(X_proc)
    return mu0_z, mu1_z

# --- Step 2: Build residual pools on Z scale, per (stratum, arm) ---
def residual_pools(Z_post, mu0_z, mu1_z, T, S):
    muT = np.where(T==1, mu1_z, mu0_z)
    resid = Z_post - muT
    pools = {}  # key: (stratum_label, arm) -> residual array
    S_cat = pd.Series(S).astype("category")
    for s_val in S_cat.cat.categories:
        idx_s = np.where(S_cat.values == s_val)[0]
        for arm in (0,1):
            idx_sa = idx_s[T[idx_s]==arm]
            if len(idx_sa) == 0:
                continue
            pools[(s_val, arm)] = resid[idx_sa]
    return pools

# --- Step 3: Smearing expectation back to currency per unit ---
def smear_expect_y(mu_z, resid_samples, B=1000, seed=0, scale=SCALE):
    rng = np.random.default_rng(seed)
    # Sample with replacement from residual pool; broadcast over n rows
    idx = rng.integers(0, len(resid_samples), size=B)
    e = resid_samples[idx][None, :]          # (1,B)
    z_draws = mu_z[:, None] + e              # (n,B)
    y_draws = scale * atanh_clip(z_draws)    # inverse tanh
    return y_draws.mean(axis=1)

# --- Step 4: Point estimates (per-unit CATE in currency, then aggregate to strata/ATE) ---
def smear_point_estimates(Z_post, tau_z, X_proc, T, S, B=1500, seed=42):
    m0, m1 = fit_arm_models_Z(X_proc, Z_post, T)
    mu0_z, mu1_z = predict_muZ(m0, m1, X_proc, tau_z=tau_z, use_tau_bridge=True)

    pools = residual_pools(Z_post, mu0_z, mu1_z, T, S)

    # For each unit, pick residual pool matching its stratum and arm
    S_cat = pd.Series(S).astype("category")
    cats = S_cat.cat.categories
    n = len(Z_post)

    # Expectations by arm USING STRATUM-SPECIFIC residual pools
    Ey0 = np.empty(n)
    Ey1 = np.empty(n)
    rng_base = np.random.SeedSequence(seed)

    for s_idx, s_val in enumerate(cats):
        idx_s = np.where(S_cat.values == s_val)[0]
        if len(idx_s) == 0:
            continue
        # Control arm residual pool for this stratum; if missing, fall back to pooled control
        r0 = pools.get((s_val, 0), np.concatenate([v for (k,v) in pools.items() if k[1]==0])) 
        r1 = pools.get((s_val, 1), np.concatenate([v for (k,v) in pools.items() if k[1]==1]))

        # Smear E[Y | T=0, X] and E[Y | T=1, X] for units in this stratum
        seed0 = rng_base.spawn(1)[0].entropy
        seed1 = rng_base.spawn(1)[0].entropy
        Ey0[idx_s] = smear_expect_y(mu0_z[idx_s], r0, B=B, seed=seed0)
        Ey1[idx_s] = smear_expect_y(mu1_z[idx_s], r1, B=B, seed=seed1)

    tau_y_unit = Ey1 - Ey0  # currency CATE per unit

    # Aggregate
    gate = pd.Series(tau_y_unit).groupby(S_cat.cat.codes).mean()
    gate.index = cats
    ate = tau_y_unit.mean()

    return {
        "tau_y_unit": tau_y_unit,
        "GATE_currency": gate,
        "ATE_currency": ate,
        "mu0_z": mu0_z,
        "mu1_z": mu1_z,
        "pools": pools
    }

# --- Step 5: Stratified bootstrap CIs for per-stratum GATEs and ATE ---
def smear_bootstrap_CIs(Z_post, tau_z, X_proc, T, S, B_smear=800, B_boot=300, seed=123):
    rng = np.random.default_rng(seed)
    S_cat = pd.Series(S).astype("category")
    cats = S_cat.cat.categories
    idx_by_s = {s: np.where(S_cat.values==s)[0] for s in cats}

    boot_gates = []
    boot_ates  = []

    for b in range(B_boot):
        # Stratified resample indices
        idx = np.concatenate([rng.choice(idx_by_s[s], size=len(idx_by_s[s]), replace=True) for s in cats])

        # Refit arm models on the bootstrap sample
        m0_b, m1_b = fit_arm_models_Z(X_proc[idx], Z_post[idx], T[idx])
        # Predict mu0/mu1 on the *original* X (common choice) or on resample; we’ll use original X so strata align
        mu0_z_b = m0_b.predict(X_proc)
        mu1_z_b = mu0_z_b + tau_z  # keep CF tau_z fixed (predict-only bootstrap). For fully rigorous, re-estimate CF.

        # Build residual pools on the bootstrap sample (by stratum, arm)
        pools_b = residual_pools(Z_post[idx], mu0_z_b[idx], mu1_z_b[idx], T[idx], S_cat.values[idx])

        # For each stratum, smear expectations using its pool; fall back to pooled arm pool if a stratum-arm is empty
        Ey0_b = np.empty(len(Z_post))
        Ey1_b = np.empty(len(Z_post))
        for s in cats:
            id_s = np.where(S_cat.values==s)[0]
            if len(id_s)==0: 
                continue
            r0b = pools_b.get((s,0), np.concatenate([v for (k,v) in pools_b.items() if k[1]==0]))
            r1b = pools_b.get((s,1), np.concatenate([v for (k,v) in pools_b.items() if k[1]==1]))
            # New seeds per bootstrap/stratum for independence
            Ey0_b[id_s] = smear_expect_y(mu0_z_b[id_s], r0b, B=B_smear, seed=rng.integers(1e9))
            Ey1_b[id_s] = smear_expect_y(mu1_z_b[id_s], r1b, B=B_smear, seed=rng.integers(1e9))

        tau_y_unit_b = Ey1_b - Ey0_b
        gate_b = pd.Series(tau_y_unit_b).groupby(S_cat.cat.codes).mean().reindex(range(len(cats))).to_numpy()
        ate_b  = tau_y_unit_b.mean()
        boot_gates.append(gate_b)
        boot_ates.append(ate_b)

    boot_gates = np.vstack(boot_gates)  # shape (B_boot, n_strata)
    boot_ates  = np.array(boot_ates)

    gate_lo = np.quantile(boot_gates, 0.025, axis=0)
    gate_hi = np.quantile(boot_gates, 0.975, axis=0)
    ate_lo  = np.quantile(boot_ates,  0.025)
    ate_hi  = np.quantile(boot_ates,  0.975)

    return gate_lo, gate_hi, ate_lo, ate_hi

# ======== Run point estimates ========
point = smear_point_estimates(Z_post, tau_z, X_proc, T, S, B=1500, seed=42)
gate_currency = point["GATE_currency"]
ate_currency  = point["ATE_currency"]

print("\n=== Per-stratum CATE (currency) — Smearing point estimates ===")
print(gate_currency)
print("\n=== Overall ATE (currency) ===")
print(f"{ate_currency:.3f}")

# ======== Bootstrap CIs (predict-only; fast & common) ========
gate_lo, gate_hi, ate_lo, ate_hi = smear_bootstrap_CIs(
    Z_post, tau_z, X_proc, T, S,
    B_smear=600,    # draws per smearing expectation (trade accuracy vs speed)
    B_boot=400,     # bootstrap replications
    seed=202
)

S_cat = pd.Series(S).astype("category")
gate_table = pd.DataFrame({
    "Stratum": S_cat.cat.categories,
    "GATE_currency": gate_currency.values,
    "CI95_lo": gate_lo,
    "CI95_hi": gate_hi
})
print("\n=== Per-stratum CATE (currency) — 95% bootstrap CI ===")
print(gate_table)

print("\n=== Overall ATE (currency) — 95% bootstrap CI ===")
print(f"[{ate_lo:.3f}, {ate_hi:.3f}]")

