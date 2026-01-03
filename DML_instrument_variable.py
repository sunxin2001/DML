
import numpy as np
import pandas as pd
import xgboost as xgb


def generate_pliv_highdim_20(n=20000, theta=1.0, rho=0.5, pi=1.0, seed=None):
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

    # instrument 
    Z = rng.normal(size=n)


    cov = np.array([[1.0, rho],
                    [rho, 1.0]])
    U, V = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n).T


    D = m + pi * Z + V


    Y = theta * D + g + 0.2 * (X11 + X12 + X13) + U

    cols = {"Y": Y, "D": D, "Z": Z}
    for j in range(20):
        cols[f"X{j+1}"] = X[:, j]
    return pd.DataFrame(cols)


def ols_slope_with_intercept(y, d):
    d_c = d - d.mean()
    y_c = y - y.mean()
    return float((d_c @ y_c) / (d_c @ d_c))

def make_kfold_indices(n, K, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, K)


def _xgb_oof_predict(X, y, folds, seed, xgb_params, num_boost_round=500,
                     val_frac=0.2, early_stopping_rounds=30):
    n = X.shape[0]
    y_hat = np.empty(n, dtype=float)

    for k, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != k])

        Xtr, ytr = X[train_idx], y[train_idx]
        Xte = X[test_idx]

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



def est_iv_dml_xgb(df, K=5, seed=42, xgb_params=None,
                   num_boost_round=500, early_stopping_rounds=30):
    Y = df["Y"].to_numpy()
    D = df["D"].to_numpy()
    Z = df["Z"].to_numpy()
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
    "lambda": 1.0,             
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
    r_hat = _xgb_oof_predict(X, Z, folds, seed=seed+33, xgb_params=xgb_params,
                             num_boost_round=num_boost_round,
                             early_stopping_rounds=early_stopping_rounds)

    y_tilde = Y - g_hat
    d_tilde = D - m_hat
    z_tilde = Z - r_hat

    denom = float(z_tilde @ d_tilde)
    if abs(denom) < 1e-10:
        raise RuntimeError("Weak instrument in this sample: z_tilde' d_tilde is near zero.")

    theta_hat = float((z_tilde @ y_tilde) / denom)
    return theta_hat


def est_plr_dml_xgb_naive(df, K=5, seed=42, xgb_params=None,
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
    "lambda": 1.0,              
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


def summarize(vals, theta0):
    arr = np.asarray(vals)
    return {
        "mean": float(arr.mean()),
        "bias": float(arr.mean() - theta0),
        "rmse": float(np.sqrt(((arr - theta0)**2).mean())),
        "sd":   float(arr.std(ddof=1))
    }

def run_mc_iv(R=50, n=2000, theta0=1.0, rho=0.5, pi=1.0, seed0=123):
    ests = {"IV-DML": [], "Naive-PLR-DML": []}
    for r in range(R):
        df = generate_pliv_highdim_20(n=n, theta=theta0, rho=rho, pi=pi, seed=seed0 + r)
        ests["IV-DML"].append(est_iv_dml_xgb(df, K=5, seed=seed0 + r + 999))
        ests["Naive-PLR-DML"].append(est_plr_dml_xgb_naive(df, K=5, seed=seed0 + r + 999))

    rows = []
    for k in ests:
        s = summarize(ests[k], theta0)
        s["method"] = k
        rows.append(s)
    return pd.DataFrame(rows).set_index("method")

if __name__ == "__main__":
    theta0 = 1.0
    rho = 0.5
    pi = 1.0

    df = generate_pliv_highdim_20(n=2000, theta=theta0, rho=rho, pi=pi, seed=124)
    print(df.head())

    print("Naive PLR-DML (biased when rho=0.5):", est_plr_dml_xgb_naive(df))
    print("IV-DML (should be close to theta0): ", est_iv_dml_xgb(df))

    tab = run_mc_iv(R=30, n=2000, theta0=theta0, rho=rho, pi=pi, seed0=123)
    print(tab)
