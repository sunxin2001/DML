# ============================================================
# CODE 1: rho = 0 (D is conditionally exogenous given X)
# Compare Robinson/Series/Diff vs PLR-DML(XGBoost)
# ============================================================
import numpy as np
import pandas as pd
from math import comb
import xgboost as xgb

# ----------------------------
# 0) DGP (same as yours)
# ----------------------------
def generate_plm_highdim_20(n=20000, theta=1.0, rho=0.0, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 20))

    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7], X[:, 8], X[:, 9]
    X11, X12, X13, X14, X15, X16, X17, X18, X19, X20 = X[:,10], X[:,11], X[:,12], X[:,13], X[:,14], X[:,15], X[:,16], X[:,17], X[:,18], X[:,19]

    g = (
        0.5 * np.sin(np.pi * X1 * X2)
        + 0.3 * X3**2
        + 0.5 * np.exp(-X4**2)
        + 0.3 * X5 * X6
        + 0.2 * X7**2 * X8
        + 0.2 * np.sin(np.pi * X9)
        + 0.1 * X10 * X2
    )

    m = (
        0.5 * X1
        + 0.3 * X2**2
        - 0.5 * np.exp(-X3**2)
        + 0.25 * X7 * X8
        + 0.2 * X9**2
        + 0.2 * X10
        + 0.1 * (X11 + X12 + X13 + X14 + X15 + X16 + X17 + X18 + X19 + X20)
    )

    cov = np.array([[1.0, rho],
                    [rho, 1.0]])
    U, V = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n).T

    D = m + V
    Y = theta * D + g + 0.2 * (X11 + X12 + X13) + U

    cols = {"Y": Y, "D": D}
    for j in range(20):
        cols[f"X{j+1}"] = X[:, j]
    return pd.DataFrame(cols)

# ----------------------------
# 1) OLS helpers
# ----------------------------
def add_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])

def ols_fit(y, X, add_const=True):
    if add_const:
        X = add_intercept(X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

def ols_slope_with_intercept(y, d):
    d_c = d - d.mean()
    y_c = y - y.mean()
    return float((d_c @ y_c) / (d_c @ d_c))

# ----------------------------
# 2) Kernel regression (1D NW)
# ----------------------------
def silverman_bw(z):
    z = np.asarray(z)
    n = len(z)
    s = z.std(ddof=1)
    h = 1.06 * s * (n ** (-1/5))
    return max(h, 1e-8)

def kernel_conditional_means_1d(Z, Y, D, W, h=None):
    Z = np.asarray(Z).reshape(-1)
    Y = np.asarray(Y).reshape(-1)
    D = np.asarray(D).reshape(-1)
    W = np.asarray(W)
    n = len(Z)

    if h is None:
        h = silverman_bw(Z)

    mY = np.empty(n)
    mD = np.empty(n)
    mW = np.empty_like(W, dtype=float)

    for i in range(n):
        u = (Z - Z[i]) / h
        w = np.exp(-0.5 * u * u)
        den = w.sum()
        if den < 1e-12:
            mY[i] = Y.mean()
            mD[i] = D.mean()
            mW[i, :] = W.mean(axis=0)
        else:
            mY[i] = (w @ Y) / den
            mD[i] = (w @ D) / den
            mW[i, :] = (w @ W) / den
    return mY, mD, mW

# ----------------------------
# 3) Robinson kernel partialling-out
# ----------------------------
def est_robinson_kernel_20_manual(df, h=None):
    Z = df["X1"].to_numpy()
    Y = df["Y"].to_numpy()
    D = df["D"].to_numpy()
    W = df[["X2", "X3", "X4", "X5"]].to_numpy()

    mY, mD, mW = kernel_conditional_means_1d(Z, Y, D, W, h=h)

    resY = Y - mY
    resD = D - mD
    resW = W - mW

    X = np.column_stack([resD, resW])
    beta = ols_fit(resY, X, add_const=True)
    return float(beta[1])

# ----------------------------
# 4) Series / Speckman
# ----------------------------
def poly_basis_1d(z, degree):
    z = np.asarray(z).reshape(-1, 1)
    return np.concatenate([z**p for p in range(degree + 1)], axis=1)

def est_series_20_manual(df, degree=5):
    Y = df["Y"].to_numpy()
    D = df["D"].to_numpy()
    Z = df["X1"].to_numpy()
    W = df[["X2", "X3", "X4", "X5"]].to_numpy()

    PZ = poly_basis_1d(Z, degree=degree)
    X = np.column_stack([D, W, PZ])
    beta = ols_fit(Y, X, add_const=True)
    return float(beta[1])

# ----------------------------
# 5) k-order differencing
# ----------------------------
def k_order_diff(arr, k):
    n = len(arr)
    out = np.zeros(n - k)
    for j in range(k + 1):
        out += ((-1)**j) * comb(k, j) * arr[j:n - k + j]
    return out

def est_diff_20_manual(df, k=2):
    df_sorted = df.sort_values("X1").reset_index(drop=True)
    Y = df_sorted["Y"].to_numpy()
    D = df_sorted["D"].to_numpy()
    W = df_sorted[["X2", "X3", "X4", "X5"]].to_numpy()

    dY = k_order_diff(Y, k)
    dD = k_order_diff(D, k)
    dW = np.column_stack([k_order_diff(W[:, j], k) for j in range(W.shape[1])])

    X = np.column_stack([dD, dW])
    beta = ols_fit(dY, X, add_const=True)
    return float(beta[1])

# ----------------------------
# 6) Cross-fitting folds
# ----------------------------
def make_kfold_indices(n, K, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, K)

# ----------------------------
# 7) PLR-DML with XGBoost nuisance (OOF predictions)
# ----------------------------
def _xgb_oof_predict(X, y, folds, seed, xgb_params, num_boost_round=500,
                     val_frac=0.2, early_stopping_rounds=30):
    n = X.shape[0]
    y_hat = np.empty(n, dtype=float)

    for k, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != k])

        Xtr, ytr = X[train_idx], y[train_idx]
        Xte = X[test_idx]

        # small validation split inside training fold (for early stopping)
        rng = np.random.default_rng(seed + 999 + k)
        perm = rng.permutation(len(train_idx))
        n_val = max(50, int(val_frac * len(train_idx)))
        val_loc = perm[:n_val]
        tr_loc = perm[n_val:]

        dtrain = xgb.DMatrix(Xtr[tr_loc], label=ytr[tr_loc])
        dval   = xgb.DMatrix(Xtr[val_loc], label=ytr[val_loc])
        dtest  = xgb.DMatrix(Xte)

        booster = xgb.train(
            params={**xgb_params, "seed": seed + 1000 + k},
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
        best_it = booster.best_iteration + 1
        y_hat[test_idx] = booster.predict(dtest, iteration_range=(0, best_it))

    return y_hat

def est_dml_plr_xgb(df, K=5, seed=42, xgb_params=None,
                    num_boost_round=500, early_stopping_rounds=30):
    Y = df["Y"].to_numpy()
    D = df["D"].to_numpy()
    X = df[[f"X{j+1}" for j in range(20)]].to_numpy().astype(np.float32)

    folds = make_kfold_indices(len(Y), K, seed=seed)

    if xgb_params is None:
        xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,              # ✅ 注意这里是字符串 key
    "min_child_weight": 1.0,
    "verbosity": 0,
    "nthread": -1
}


    g_hat = _xgb_oof_predict(X, Y, folds, seed=seed+11, xgb_params=xgb_params,
                             num_boost_round=num_boost_round,
                             early_stopping_rounds=early_stopping_rounds)
    m_hat = _xgb_oof_predict(X, D, folds, seed=seed+22, xgb_params=xgb_params,
                             num_boost_round=num_boost_round,
                             early_stopping_rounds=early_stopping_rounds)

    y_tilde = Y - g_hat
    d_tilde = D - m_hat

    return float(ols_slope_with_intercept(y_tilde, d_tilde))

# ----------------------------
# 8) Monte Carlo
# ----------------------------
def summarize(vals, theta0):
    arr = np.asarray(vals)
    return {
        "mean": float(arr.mean()),
        "bias": float(arr.mean() - theta0),
        "rmse": float(np.sqrt(((arr - theta0)**2).mean())),
        "sd":   float(arr.std(ddof=1))
    }

def mc_once_20(n=2000, theta0=1.0, rho=0.0, seed=0):
    df = generate_plm_highdim_20(n=n, theta=theta0, rho=rho, seed=seed)
    return {
        "Robinson": est_robinson_kernel_20_manual(df),
        "Series":   est_series_20_manual(df, degree=5),
        "Diff":     est_diff_20_manual(df, k=2),
        "DML":      est_dml_plr_xgb(df, K=5, seed=seed+999,
                                   num_boost_round=500, early_stopping_rounds=30)
    }

def run_mc(R=50, n=2000, theta0=1.0, rho=0.0, seed0=123):
    ests = {k: [] for k in ["Robinson", "Series", "Diff", "DML"]}
    for r in range(R):
        out = mc_once_20(n=n, theta0=theta0, rho=rho, seed=seed0 + r)
        for k, v in out.items():
            ests[k].append(v)

    rows = []
    for k in ests:
        s = summarize(ests[k], theta0)
        s["method"] = k
        rows.append(s)
    return pd.DataFrame(rows).set_index("method")

if __name__ == "__main__":
    theta0 = 1.0
    rho = 0.0

    df = generate_plm_highdim_20(n=2000, theta=theta0, rho=rho, seed=124)
    print(df.head())

    print("Robinson:", est_robinson_kernel_20_manual(df))
    print("Series  :", est_series_20_manual(df))
    print("Diff    :", est_diff_20_manual(df))
    print("DML     :", est_dml_plr_xgb(df))

    tab = run_mc(R=50, n=2000, theta0=theta0, rho=rho, seed0=123)
    print(tab)
