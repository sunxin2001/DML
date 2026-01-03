import numpy as np
import pandas as pd
from math import comb
import xgboost as xgb

def generate_plm_highdim_20(n=20000, theta=1.0, rho=0.5, seed=None):
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



# OLS

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



# 1D 核回归

def silverman_bw(z):
    z = np.asarray(z)
    n = len(z)
    s = z.std(ddof=1)
    h = 1.06 * s * (n ** (-1/5))
    return max(h, 1e-8)

def kernel_conditional_means_1d(Z, Y, D, W, h=None):
    """
    对每个 i:
      w_ij = exp(-0.5 * ((Z_j - Z_i)/h)^2)
      mY_i = sum_j w_ij Y_j / sum_j w_ij
    复杂度 O(n^2)，n=2000 没问题；n=20000 会很慢（需要近似/分块/树结构）。
    """
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
        w = np.exp(-0.5 * u * u)   # 高斯核
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



# Robinson 

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
    beta = ols_fit(resY, X, add_const=True)  # [const, theta, ...]
    return float(beta[1])



#  Speckman

def poly_basis_1d(z, degree):
    z = np.asarray(z).reshape(-1, 1)
    # [1, z, z^2, ..., z^degree]
    return np.concatenate([z**p for p in range(degree + 1)], axis=1)

def est_series_20_manual(df, degree=5):
    Y = df["Y"].to_numpy()
    D = df["D"].to_numpy()
    Z = df["X1"].to_numpy()
    W = df[["X2", "X3", "X4", "X5"]].to_numpy()

    PZ = poly_basis_1d(Z, degree=degree)
    X = np.column_stack([D, W, PZ])
    beta = ols_fit(Y, X, add_const=True)  # [const, theta, ...]
    return float(beta[1])



# 差分

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



# Lasso

def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd

def standardize_apply(X, mu, sd):
    return (X - mu) / sd

def soft_threshold(x, lam):
    if x > lam:
        return x - lam
    if x < -lam:
        return x + lam
    return 0.0

def lasso_cd_fit(X, y, lam, max_iter=2000, tol=1e-6):
    
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n, p = X.shape

    muX, sdX = standardize_fit(X)
    Xs = standardize_apply(X, muX, sdX)

    muy = y.mean()
    yc = y - muy

  
    zj = (Xs * Xs).mean(axis=0)

    b = np.zeros(p)
    r = yc.copy()  

    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            bj_old = b[j]

            
            r += Xs[:, j] * bj_old

            rho = (Xs[:, j] @ r) / n

            
            b[j] = soft_threshold(rho, lam) / (zj[j] + 1e-12)

            
            r -= Xs[:, j] * b[j]

            max_change = max(max_change, abs(b[j] - bj_old))

        if max_change < tol:
            break

   
    beta = b / sdX
    intercept = muy - muX @ beta
    return intercept, beta

def make_kfold_indices(n, K, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, K)
    return folds

def lasso_cv_choose_lambda(X, y, K=5, n_lams=40, min_ratio=1e-3, seed=0):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n, p = X.shape

 
    muX, sdX = standardize_fit(X)
    Xs = standardize_apply(X, muX, sdX)
    yc = y - y.mean()

    lam_max = np.max(np.abs(Xs.T @ yc)) / n
    lam_min = lam_max * min_ratio
    lams = np.geomspace(lam_max, lam_min, n_lams)

    folds = make_kfold_indices(n, K, seed=seed)

    mse_path = []
    for lam in lams:
        mse_folds = []
        for k in range(K):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(K) if j != k])

            Xtr, ytr = X[train_idx], y[train_idx]
            Xva, yva = X[val_idx], y[val_idx]

            a, b = lasso_cd_fit(Xtr, ytr, lam=lam)
            yhat = a + Xva @ b
            mse_folds.append(np.mean((yva - yhat)**2))

        mse_path.append(np.mean(mse_folds))

    best = int(np.argmin(mse_path))
    return float(lams[best])



# DML 
def est_dml_20_manual(df, K=5, seed=42,
                      n_lams=40, min_ratio=1e-3,
                      tune_each_fold=False,
                      xgb_params=None,
                      num_boost_round=500):
    Y = df["Y"].to_numpy()
    D = df["D"].to_numpy()
    X = df[[f"X{j+1}" for j in range(20)]].to_numpy().astype(np.float32)
    n = len(Y)

    folds = make_kfold_indices(n, K, seed=seed)


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
            "seed": seed,
            "nthread": -1
        }

    y_hat = np.empty(n, dtype=float)
    d_hat = np.empty(n, dtype=float)

    for k in range(K):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != k])

        Xtr, Ytr, Dtr = X[train_idx], Y[train_idx], D[train_idx]
        Xte = X[test_idx]


        dtrain_y = xgb.DMatrix(Xtr, label=Ytr)
        dtest = xgb.DMatrix(Xte)
        booster_y = xgb.train(
            params={**xgb_params, "seed": seed + 1000 + k},
            dtrain=dtrain_y,
            num_boost_round=num_boost_round
        )
        y_hat[test_idx] = booster_y.predict(dtest)


        dtrain_d = xgb.DMatrix(Xtr, label=Dtr)
        booster_d = xgb.train(
            params={**xgb_params, "seed": seed + 2000 + k},
            dtrain=dtrain_d,
            num_boost_round=num_boost_round
        )
        d_hat[test_idx] = booster_d.predict(dtest)

    y_tilde = Y - y_hat
    d_tilde = D - d_hat

    theta_hat = ols_slope_with_intercept(y_tilde, d_tilde)
    return float(theta_hat)




# Monte Carlo

def mc_once_20(n=2000, theta0=1.0, seed=0):
    df = generate_plm_highdim_20(n=n, theta=theta0, seed=seed)
    return {
        "Robinson": est_robinson_kernel_20_manual(df),
        "Series":   est_series_20_manual(df, degree=5),
        "Diff":     est_diff_20_manual(df, k=2),
        "DML":      est_dml_20_manual(df, K=5, seed=seed+999,
                                      n_lams=30, min_ratio=1e-3,
                                      tune_each_fold=False)
    }

def summarize(vals, theta0):
    arr = np.asarray(vals)
    return {
        "mean": float(arr.mean()),
        "bias": float(arr.mean() - theta0),
        "rmse": float(np.sqrt(((arr - theta0)**2).mean())),
        "sd":   float(arr.std(ddof=1))
    }

def run_mc(R=100, n=2000, theta0=1.0, seed0=123):
    ests = {k: [] for k in ["Robinson", "Series", "Diff", "DML"]}
    for r in range(R):
        out = mc_once_20(n=n, theta0=theta0, seed=seed0 + r)
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
    df = generate_plm_highdim_20(n=2000, theta=theta0, seed=124)
    print(df.head())

    print("Robinson:", est_robinson_kernel_20_manual(df))
    print("Series  :", est_series_20_manual(df))
    print("Diff    :", est_diff_20_manual(df))
    print("DML     :", est_dml_20_manual(df))

    tab = run_mc(R=50, n=2000, theta0=theta0) 
    print(tab)
