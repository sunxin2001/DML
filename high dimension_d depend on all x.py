import numpy as np
import pandas as pd
import xgboost as xgb
import time


def generate_data(n=3000, theta=1.0, seed=9527):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 10))
    

    g = 1.5 * (X[:, 0] * X[:, 1]) + np.sin(np.pi * X[:, 2]) + 0.8 * X[:, 3] * X[:, 4]
    

    m = 1.0 * X[:, 0] + 1.0 * X[:, 1] + 1.0 * X[:, 2] + 1.0 * X[:, 3]
    D = m + rng.normal(size=n)
    
    Y = theta * D + g + rng.normal(size=n)
    
    cols = {"Y": Y, "D": D}
    for j in range(10):
        cols[f"X{j+1}"] = X[:, j]
    return pd.DataFrame(cols)


def manual_ols(y, d):
    return float((d @ y) / (d @ d))

def nw_kernel_smooth(target, feature, bw):
    """Nadaraya-Watson Kernel"""
    diff = feature[:, np.newaxis] - feature[np.newaxis, :]
    weights = np.exp(-0.5 * (diff / bw)**2)
    return (weights @ target) / weights.sum(axis=1)

def est_robinson_high_dim(df):
    Y, D, X1 = df["Y"].to_numpy(), df["D"].to_numpy(), df["X1"].to_numpy()
    h = 1.06 * np.std(X1) * len(Y)**(-1/5)
    y_tilde = Y - nw_kernel_smooth(Y, X1, h)
    d_tilde = D - nw_kernel_smooth(D, X1, h)
    return manual_ols(y_tilde, d_tilde)

def est_speckman_high_dim(df):
    Y, D, X1 = df["Y"].to_numpy(), df["D"].to_numpy(), df["X1"].to_numpy()
    h = 1.06 * np.std(X1) * len(Y)**(-1/5)
    d_tilde = D - nw_kernel_smooth(D, X1, h)
    y_tilde = Y - nw_kernel_smooth(Y, X1, h)
    return manual_ols(y_tilde, d_tilde)

def est_yatchew_high_dim(df):
    df_s = df.sort_values(by="X1")
    dy = (df_s["Y"].to_numpy()[1:] - df_s["Y"].to_numpy()[:-1]) / np.sqrt(2)
    dd = (df_s["D"].to_numpy()[1:] - df_s["D"].to_numpy()[:-1]) / np.sqrt(2)
    return manual_ols(dy, dd)


def est_dml_high_dim(df, K=5, seed=42):
    Y, D = df["Y"].to_numpy(), df["D"].to_numpy()
    X = df[[f"X{j+1}" for j in range(10)]].to_numpy().astype(np.float32)
    
    indices = np.arange(len(Y))
    np.random.default_rng(seed).shuffle(indices)
    folds = np.array_split(indices, K)
    
    y_hat, d_hat = np.zeros(len(Y)), np.zeros(len(D))
    params = {"max_depth": 4, "eta": 0.1, "objective": "reg:squarederror", "nthread": 1}

    for k in range(K):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
        
        for target, hat_arr in zip([Y, D], [y_hat, d_hat]):
            dtr = xgb.DMatrix(X[train_idx], label=target[train_idx])
            model = xgb.train(params, dtr, num_boost_round=100)
            hat_arr[test_idx] = model.predict(xgb.DMatrix(X[test_idx]))

    return manual_ols(Y - y_hat, D - d_hat)


def run_high_dim_comparison(R=30, n=15000, theta0=1.0):
    print(f"High-Dim MC: R={R}, n={n}, g(X) depends on all X1-X10")
    results = {"Robinson (X1 only)": [], "Speckman (X1 only)": [], 
               "Yatchew (X1 only)": [], "DML (All X)": []}
    
    for r in range(R):
        df = generate_high_dim_plr(n=n, theta=theta0, seed=9527)
        results["Robinson (X1 only)"].append(est_robinson_high_dim(df))
        results["Speckman (X1 only)"].append(est_speckman_high_dim(df))
        results["Yatchew (X1 only)"].append(est_yatchew_high_dim(df))
        results["DML (All X)"].append(est_dml_high_dim(df, seed=9527))

    summary = []
    for m, v in results.items():
        summary.append({"Method": m, "Mean": np.mean(v), "Bias": np.mean(v)-theta0, 
                        "RMSE": np.sqrt(np.mean((np.array(v)-theta0)**2))})
    return pd.DataFrame(summary).set_index("Method")

if __name__ == "__main__":
    print(run_high_dim_comparison())
