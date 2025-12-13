import numpy as np
import pandas as pd
from math import comb
import xgboost as xgb
# ============================================================
# 0) DGP: high-dim PLM (你的版本，基本不动)
# ============================================================
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


# ============================================================
# 1) 手写 OLS: (X'X)^{-1}X'y / lstsq
# ============================================================
def add_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])

def ols_fit(y, X, add_const=True):
    if add_const:
        X = add_intercept(X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

def ols_slope_with_intercept(y, d):
    # 回归 y = a + theta d + e 的 theta（更稳）
    d_c = d - d.mean()
    y_c = y - y.mean()
    return float((d_c @ y_c) / (d_c @ d_c))


# ============================================================
# 2) 手写 1D Nadaraya–Watson 核回归（高斯核）
#    同一套权重同时算 E[Y|Z], E[D|Z], E[W|Z]
# ============================================================
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
        w = np.exp(-0.5 * u * u)   # 高斯核(常数项可省略)
        den = w.sum()
        if den < 1e-12:
            # 极端情况兜底：用全局均值
            mY[i] = Y.mean()
            mD[i] = D.mean()
            mW[i, :] = W.mean(axis=0)
        else:
            mY[i] = (w @ Y) / den
            mD[i] = (w @ D) / den
            mW[i, :] = (w @ W) / den

    return mY, mD, mW


# ============================================================
# 3) 手写 Robinson kernel partialling-out
# ============================================================
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


# ============================================================
# 4) 手写 Series / Speckman：多项式基 + OLS
# ============================================================
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


# ============================================================
# 5) 手写 k 阶差分（你给的版本）
# ============================================================
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


# ============================================================
# 6) 手写 Lasso（坐标下降）+ 手写 CV 选 lambda
# ============================================================
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
    """
    min (1/2n)||y - a - Xb||^2 + lam * ||b||_1
    - 我们用：先标准化 X（列均值0方差1），中心化 y
    - 坐标下降更新 b
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n, p = X.shape

    muX, sdX = standardize_fit(X)
    Xs = standardize_apply(X, muX, sdX)

    muy = y.mean()
    yc = y - muy

    # 预计算列范数：(1/n) * ||X_j||^2
    zj = (Xs * Xs).mean(axis=0)

    b = np.zeros(p)
    r = yc.copy()  # 因为 b=0，所以残差=yc

    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            bj_old = b[j]

            # 把旧的贡献加回残差
            r += Xs[:, j] * bj_old

            rho = (Xs[:, j] @ r) / n  # (1/n) X_j' (y - sum_{k!=j} X_k b_k)

            # soft-threshold
            b[j] = soft_threshold(rho, lam) / (zj[j] + 1e-12)

            # 把新的贡献减出去
            r -= Xs[:, j] * b[j]

            max_change = max(max_change, abs(b[j] - bj_old))

        if max_change < tol:
            break

    # 还原到原尺度：y ≈ a + X beta
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
    """
    手写 CV：在一条 lambda path 上选使验证 MSE 最小的 lambda
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n, p = X.shape

    # 先用全样本标准化来构造 lambda_max（只用于网格，不用于最终拟合）
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


# ============================================================
# 7) 手写 DML (PLR) + cross-fitting + Lasso nuisance
# ============================================================
def est_dml_20_manual(df, K=5, seed=42,
                      n_lams=40, min_ratio=1e-3,
                      tune_each_fold=False,
                      xgb_params=None,
                      num_boost_round=500):
    """
    手写 DML(PLR) + cross-fitting：
      1) 用 ML 估计 g(x)=E[Y|X], m(x)=E[D|X]
      2) y_tilde = Y - ghat, d_tilde = D - mhat
      3) OLS: y_tilde ~ d_tilde 得 theta

    这里 nuisance learner 改用 XGBoost，并把原 Lasso 部分注释掉。
    注意：函数签名里 n_lams/min_ratio/tune_each_fold 为了兼容你 MC 的调用保留，但已不再使用。
    """
    Y = df["Y"].to_numpy()
    D = df["D"].to_numpy()
    X = df[[f"X{j+1}" for j in range(20)]].to_numpy().astype(np.float32)
    n = len(Y)

    folds = make_kfold_indices(n, K, seed=seed)

    # ===== 原先 Lasso 选 lambda（全部注释掉）=====
    # if not tune_each_fold:
    #     lam_y = lasso_cv_choose_lambda(X, Y, K=5, n_lams=n_lams, min_ratio=min_ratio, seed=seed+1)
    #     lam_d = lasso_cv_choose_lambda(X, D, K=5, n_lams=n_lams, min_ratio=min_ratio, seed=seed+2)

    # ===== XGBoost 参数（可按需改）=====
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

        # ===== 原先每折 tune lambda（注释掉）=====
        # if tune_each_fold:
        #     lam_y = lasso_cv_choose_lambda(Xtr, Ytr, K=5, n_lams=n_lams, min_ratio=min_ratio, seed=seed+10+k)
        #     lam_d = lasso_cv_choose_lambda(Xtr, Dtr, K=5, n_lams=n_lams, min_ratio=min_ratio, seed=seed+20+k)

        # ===== 原先 Lasso 拟合（注释掉）=====
        # ay, by = lasso_cd_fit(Xtr, Ytr, lam=lam_y)
        # ad, bd = lasso_cd_fit(Xtr, Dtr, lam=lam_d)
        # y_hat[test_idx] = ay + Xte @ by
        # d_hat[test_idx] = ad + Xte @ bd

        # ===== 改用 XGBoost：训练 ghat=E[Y|X] =====
        dtrain_y = xgb.DMatrix(Xtr, label=Ytr)
        dtest = xgb.DMatrix(Xte)
        booster_y = xgb.train(
            params={**xgb_params, "seed": seed + 1000 + k},
            dtrain=dtrain_y,
            num_boost_round=num_boost_round
        )
        y_hat[test_idx] = booster_y.predict(dtest)

        # ===== 改用 XGBoost：训练 mhat=E[D|X] =====
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



# ============================================================
# 8) Monte Carlo：四种方法手写版
# ============================================================
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


# ============================================================
# 9) 示例运行
# ============================================================
if __name__ == "__main__":
    theta0 = 1.0
    df = generate_plm_highdim_20(n=2000, theta=theta0, seed=124)
    print(df.head())

    # 单次估计
    print("Robinson:", est_robinson_kernel_20_manual(df))
    print("Series  :", est_series_20_manual(df))
    print("Diff    :", est_diff_20_manual(df))
    print("DML     :", est_dml_20_manual(df))

    # Monte Carlo
    tab = run_mc(R=50, n=2000, theta0=theta0)  # 先 50 次试跑
    print(tab)
