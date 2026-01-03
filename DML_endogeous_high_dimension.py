import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge


def generate_pliv_data_dml_win(
    n=2000,
    p=80,
    s_dim=30,
    theta=1.0,
    rho=0.5,
    pi=2.0,
    z_x_strength=2.5,
    z_noise_sd=0.8,
    g_strength=2.5,
    m_strength=1.0,
    seed=None
):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))

    Xa = X[:, :s_dim]
    idx = np.arange(1, s_dim + 1)
    w = 1.0 / np.sqrt(idx)
    w = w / w.sum()

    s_add = (np.sin(Xa) @ w) + 0.5 * (np.tanh(Xa) @ w) + 0.3 * ((Xa**2 - 1.0) @ w)
    s_int = (
        0.4 * (Xa[:, 0] * Xa[:, 1])
        + 0.3 * (np.sin(Xa[:, 2] + Xa[:, 3]))
        + 0.2 * (Xa[:, 4] * np.tanh(Xa[:, 5]))
    )
    s = s_add + s_int

    r = z_x_strength * s
    Z = r + rng.normal(scale=z_noise_sd, size=n)

    m = m_strength * (0.8 * s_add + 0.4 * np.tanh(Xa[:, 0]) + 0.3 * (Xa[:, 1]**2 - 1.0))

    g = g_strength * s + 0.4 * np.cos(Xa[:, 6] + Xa[:, 7]) + 0.3 * (np.sin(Xa[:, 8]) * Xa[:, 9])

    cov = np.array([[1.0, rho], [rho, 1.0]])
    U, V = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n).T

    D = m + pi * Z + V
    Y = theta * D + g + U

    cols = {"Y": Y, "D": D, "Z": Z}
    for j in range(p):
        cols[f"X{j+1}"] = X[:, j]
    return pd.DataFrame(cols)


def _silverman_bandwidth(x):
    x = np.asarray(x).reshape(-1)
    n = x.size
    sd = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(sd, iqr / 1.349) if (sd > 0 and iqr > 0) else (sd if sd > 0 else 1.0)
    h = 0.9 * sigma * n ** (-1 / 5)
    return max(h, 1e-6)

def nw_predict_1d(x_train, y_train, x_eval, h=None, eps=1e-12):
    x_train = np.asarray(x_train).reshape(-1)
    y_train = np.asarray(y_train).reshape(-1)
    x_eval  = np.asarray(x_eval).reshape(-1)
    if h is None:
        h = _silverman_bandwidth(x_train)

    u = (x_eval[:, None] - x_train[None, :]) / h
    w = np.exp(-0.5 * u * u)
    denom = w.sum(axis=1) + eps
    return (w @ y_train) / denom



def iv_robinson_kernel(df, x_col="X1", h=None, return_residuals=False):
    y = df["Y"].to_numpy()
    d = df["D"].to_numpy()
    z = df["Z"].to_numpy()
    x = df[x_col].to_numpy()

    y_hat = nw_predict_1d(x, y, x, h=h)
    d_hat = nw_predict_1d(x, d, x, h=h)
    z_hat = nw_predict_1d(x, z, x, h=h)

    y_tilde = y - y_hat
    d_tilde = d - d_hat
    z_tilde = z - z_hat

    denom = np.sum(z_tilde * d_tilde)
    if abs(denom) < 1e-12:
        raise ValueError("Robinson-IV: denom ~ 0.")

    theta_hat = np.sum(z_tilde * y_tilde) / denom
    if return_residuals:
        return float(theta_hat), y_tilde, d_tilde, z_tilde
    return {"theta_hat": float(theta_hat)}


def _spline_smoother_fit_predict(x, y, n_knots=14, degree=3, ridge_alpha=1.0):
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1)
    model = make_pipeline(
        SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False),
        Ridge(alpha=ridge_alpha)
    )
    model.fit(x, y)
    return model.predict(x)

def iv_speckman_spline(df, x_col="X1", n_knots=14, degree=3, ridge_alpha=1.0, return_residuals=False):
    y = df["Y"].to_numpy()
    d = df["D"].to_numpy()
    z = df["Z"].to_numpy()
    x = df[x_col].to_numpy()

    y_hat = _spline_smoother_fit_predict(x, y, n_knots=n_knots, degree=degree, ridge_alpha=ridge_alpha)
    d_hat = _spline_smoother_fit_predict(x, d, n_knots=n_knots, degree=degree, ridge_alpha=ridge_alpha)
    z_hat = _spline_smoother_fit_predict(x, z, n_knots=n_knots, degree=degree, ridge_alpha=ridge_alpha)

    y_tilde = y - y_hat
    d_tilde = d - d_hat
    z_tilde = z - z_hat

    denom = np.sum(z_tilde * d_tilde)
    if abs(denom) < 1e-12:
        raise ValueError("Speckman-IV: denom ~ 0.")

    theta_hat = np.sum(z_tilde * y_tilde) / denom
    if return_residuals:
        return float(theta_hat), y_tilde, d_tilde, z_tilde
    return {"theta_hat": float(theta_hat)}


def iv_yatchew_differencing(df, x_col="X1", return_residuals=False):
    df_sorted = df.sort_values(by=x_col).reset_index(drop=True)
    y = df_sorted["Y"].to_numpy()
    d = df_sorted["D"].to_numpy()
    z = df_sorted["Z"].to_numpy()

    dy = np.diff(y)
    dd = np.diff(d)
    dz = np.diff(z)

    denom = np.sum(dz * dd)
    if abs(denom) < 1e-12:
        raise ValueError("Yatchew-IV: denom ~ 0.")

    theta_hat = np.sum(dz * dy) / denom
    if return_residuals:
        return float(theta_hat), dy, dd, dz
    return {"theta_hat": float(theta_hat)}


def _fit_predict_xgb_with_es(X_tr, y_tr, X_te, seed, params,
                            val_frac=0.2, early_stopping_rounds=80):

    rng = np.random.default_rng(seed)
    n_tr = X_tr.shape[0]
    idx = rng.permutation(n_tr)
    n_val = max(20, int(np.floor(val_frac * n_tr)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=seed,
        eval_metric="rmse",
        tree_method="hist",
        **params
    )


    try:
        model.fit(
            X_tr[tr_idx], y_tr[tr_idx],
            eval_set=[(X_tr[val_idx], y_tr[val_idx])],
            verbose=False,
            early_stopping_rounds=early_stopping_rounds
        )
        return model.predict(X_te)
    except TypeError:
        pass


    try:
        cb = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True)]
        model.fit(
            X_tr[tr_idx], y_tr[tr_idx],
            eval_set=[(X_tr[val_idx], y_tr[val_idx])],
            verbose=False,
            callbacks=cb
        )
        return model.predict(X_te)
    except TypeError:
        pass


    model.fit(X_tr, y_tr, verbose=False)
    return model.predict(X_te)



def iv_dml_pliv(
    df,
    x_cols,
    K=10,
    seed=123,
    xgb_params_y=None,
    xgb_params_d=None,
    xgb_params_z=None,
    return_residuals=False
):
    if xgb_params_y is None:
        xgb_params_y = dict(
            n_estimators=6000, max_depth=7, learning_rate=0.02,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, min_child_weight=5, gamma=0.0
        )
    if xgb_params_d is None:
        xgb_params_d = dict(
            n_estimators=6000, max_depth=7, learning_rate=0.02,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, min_child_weight=5, gamma=0.0
        )

    if xgb_params_z is None:
        xgb_params_z = dict(
            n_estimators=2500, max_depth=3, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=2.0, min_child_weight=30, gamma=0.0
        )

    Y = df["Y"].to_numpy()
    D = df["D"].to_numpy()
    Z = df["Z"].to_numpy()
    X = df[x_cols].to_numpy()

    n = len(df)
    y_hat = np.empty(n)
    d_hat = np.empty(n)
    z_hat = np.empty(n)

    kf = KFold(n_splits=K, shuffle=True, random_state=seed)

    for fold, (tr, te) in enumerate(kf.split(X), start=1):
        rs = seed + 1000 * fold

        y_hat[te] = _fit_predict_xgb_with_es(X[tr], Y[tr], X[te], rs,     xgb_params_y)
        d_hat[te] = _fit_predict_xgb_with_es(X[tr], D[tr], X[te], rs + 1, xgb_params_d)
        z_hat[te] = _fit_predict_xgb_with_es(X[tr], Z[tr], X[te], rs + 2, xgb_params_z)

    y_tilde = Y - y_hat
    d_tilde = D - d_hat
    z_tilde = Z - z_hat

    denom = np.sum(z_tilde * d_tilde)
    if abs(denom) < 1e-12:
        raise ValueError("IV-DML: denom ~ 0 (residualized instrument too weak).")

    theta_hat = np.sum(z_tilde * y_tilde) / denom

    if return_residuals:
        return float(theta_hat), y_tilde, d_tilde, z_tilde
    return {"theta_hat": float(theta_hat)}



METHODS = ["IV-DML", "Robinson-IV", "Speckman-IV", "Yatchew-IV"]

def summarize_mc(estimates_df, theta0):
    rows = []
    for m in estimates_df.columns:
        vals = estimates_df[m].dropna().to_numpy()
        n_ok = len(vals)
        mean = float(np.mean(vals)) if n_ok else np.nan
        bias = float(mean - theta0) if n_ok else np.nan
        rmse = float(np.sqrt(np.mean((vals - theta0) ** 2))) if n_ok else np.nan
        sd = float(np.std(vals, ddof=1)) if n_ok > 1 else (0.0 if n_ok == 1 else np.nan)
        rows.append({"Method": m, "N_ok": n_ok, "Mean": mean, "Bias": bias, "RMSE": rmse, "SD": sd})
    return pd.DataFrame(rows).sort_values("Method").reset_index(drop=True)

def run_monte_carlo(
    R=100,
    n=2000,
    p=80,
    theta0=1.0,
    rho=0.5,
    pi=2.0,
    s_dim=30,
    z_x_strength=2.5,
    z_noise_sd=0.8,
    g_strength=2.5,
    base_seed=20260101,

    dml_K=10,
    xgb_params_y=None,
    xgb_params_d=None,
    xgb_params_z=None,

    spline_knots=14,
    spline_degree=3,
    spline_ridge_alpha=1.0,
    verbose=True
):
    est_rows = []
    diag_rows = []
    printed_dml_error = False

    for r in range(R):
        seed = base_seed + r

        df = generate_pliv_data_dml_win(
            n=n, p=p, s_dim=s_dim, theta=theta0, rho=rho, pi=pi,
            z_x_strength=z_x_strength, z_noise_sd=z_noise_sd,
            g_strength=g_strength, m_strength=1.0,
            seed=seed
        )

        row = {m: np.nan for m in METHODS}
        diag = {}

        try:
            x_cols = [f"X{j+1}" for j in range(p)]
            th, yt, dt, zt = iv_dml_pliv(
                df, x_cols=x_cols, K=dml_K, seed=seed + 999,
                xgb_params_y=xgb_params_y,
                xgb_params_d=xgb_params_d,
                xgb_params_z=xgb_params_z,
                return_residuals=True
            )
            row["IV-DML"] = th
            diag["IV-DML_denom_over_n"] = float(np.sum(zt * dt) / len(df))
            diag["IV-DML_corr_zd"] = float(np.corrcoef(zt, dt)[0, 1])
        except Exception as e:
            if not printed_dml_error:
                print("[DML ERROR]", repr(e))
                printed_dml_error = True
            diag["IV-DML_denom_over_n"] = np.nan
            diag["IV-DML_corr_zd"] = np.nan

        th, yt, dt, zt = iv_robinson_kernel(df, x_col="X1", return_residuals=True)
        row["Robinson-IV"] = th
        diag["Robinson_denom_over_n"] = float(np.sum(zt * dt) / len(df))
        diag["Robinson_corr_zd"] = float(np.corrcoef(zt, dt)[0, 1])


        th, yt, dt, zt = iv_speckman_spline(
            df, x_col="X1",
            n_knots=spline_knots, degree=spline_degree, ridge_alpha=spline_ridge_alpha,
            return_residuals=True
        )
        row["Speckman-IV"] = th
        diag["Speckman_denom_over_n"] = float(np.sum(zt * dt) / len(df))
        diag["Speckman_corr_zd"] = float(np.corrcoef(zt, dt)[0, 1])


        th, dy, dd, dz = iv_yatchew_differencing(df, x_col="X1", return_residuals=True)
        row["Yatchew-IV"] = th
        diag["Yatchew_denom_over_n"] = float(np.sum(dz * dd) / len(dy))
        diag["Yatchew_corr_zd"] = float(np.corrcoef(dz, dd)[0, 1])

        est_rows.append(row)
        diag_rows.append(diag)

        if verbose and (r + 1) % max(1, R // 10) == 0:
            print(f"[MC] {r+1}/{R} done")

    estimates_df = pd.DataFrame(est_rows, columns=METHODS)
    summary_df = summarize_mc(estimates_df, theta0)
    diag_df = pd.DataFrame(diag_rows)
    return estimates_df, summary_df, diag_df



def plot_hist_grid(estimates_df, theta0=1.0, bins=30):
    methods = list(estimates_df.columns)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, m in zip(axes, methods):
        vals = estimates_df[m].dropna().to_numpy()
        ax.hist(vals, bins=bins)
        ax.axvline(theta0, linestyle="--")
        ax.set_title(f"{m} (n_ok={len(vals)})")
        ax.set_xlabel(r"$\hat\theta$")
        ax.set_ylabel("count")
    plt.tight_layout()
    plt.show()

def plot_bias_rmse(summary_df):
    methods = summary_df["Method"].tolist()
    bias = summary_df["Bias"].to_numpy()
    rmse = summary_df["RMSE"].to_numpy()
    x = np.arange(len(methods))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, bias, width, label="Bias")
    ax.bar(x + width/2, rmse, width, label="RMSE")
    ax.axhline(0.0)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_title("Monte Carlo performance summary")
    ax.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    theta0 = 1.0
    rho = 0.5
    R = 100
    n = 2000
    p = 80
    pi = 2.0
    s_dim = 30
    z_x_strength = 2.5
    z_noise_sd = 0.8
    g_strength = 2.5


    dml_K = 10  

    xgb_params_y = None
    xgb_params_d = None
    xgb_params_z = None

    estimates, summary, diag = run_monte_carlo(
        R=R, n=n, p=p, theta0=theta0, rho=rho, pi=pi,
        s_dim=s_dim, z_x_strength=z_x_strength, z_noise_sd=z_noise_sd, g_strength=g_strength,
        base_seed=20260101,
        dml_K=dml_K,
        xgb_params_y=xgb_params_y,
        xgb_params_d=xgb_params_d,
        xgb_params_z=xgb_params_z,
        verbose=True
    )

    print("\n=== Summary (theta0=1.0) ===")
    print(summary)

    print("\n=== Instrument strength diagnostics (mean over replications) ===")
    print(diag.mean(numeric_only=True))

    plot_hist_grid(estimates, theta0=theta0, bins=30)
    plot_bias_rmse(summary)
