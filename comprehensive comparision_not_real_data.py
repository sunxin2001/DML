import numpy as np
import pandas as pd
import xgboost as xgb
import time


def generate_data(n=5000, theta=1.0, seed=None):
    rng = np.random.default_rng(seed)

    X = rng.uniform(-1, 1, size=(n, 10))
    

    g = 1.5 * (X[:, 0] * X[:, 1]) + np.sin(np.pi * X[:, 2]) + 1.2 * X[:, 3]**2
    
 
    m = 1.2 * X[:, 0] + 1.0 * X[:, 1] + 0.8 * X[:, 2] + 1.0 * X[:, 3]
    D = m + rng.normal(size=n)
    
 
    Y = theta * D + g + rng.normal(size=n)
    
    cols = {"Y": Y, "D": D}
    for j in range(10):
        cols[f"X{j+1}"] = X[:, j]
    return pd.DataFrame(cols)


def nw_smooth(target, feature, bw):
    """Nadaraya-Watson Kernel Smoothing"""
    diff = feature[:, np.newaxis] - feature[np.newaxis, :]
    weights = np.exp(-0.5 * (diff / bw)**2)
    return (weights @ target) / weights.sum(axis=1)

def manual_ols(y, d):
    return float((d @ y) / (d @ d))

def est_robinson_1d(df):
    """Robinson (1988): 仅对 X1 剥离，忽略高维干扰"""
    Y, D, X1 = df["Y"].to_numpy(), df["D"].to_numpy(), df["X1"].to_numpy()
    h = 1.06 * np.std(X1) * len(Y)**(-1/5)
    y_tilde = Y - nw_smooth(Y, X1, h)
    d_tilde = D - nw_smooth(D, X1, h)
    return manual_ols(y_tilde, d_tilde)

def est_speckman_1d(df):
    """Speckman (1988): 仅对 X1 进行偏投影"""
    Y, D, X1 = df["Y"].to_numpy(), df["D"].to_numpy(), df["X1"].to_numpy()
    h = 1.06 * np.std(X1) * len(Y)**(-1/5)
    d_tilde = D - nw_smooth(D, X1, h)
    y_tilde = Y - nw_smooth(Y, X1, h)
    return manual_ols(y_tilde, d_tilde)

def est_yatchew_1d(df):
    """Yatchew (1997): 仅按 X1 排序差分"""
    df_sorted = df.sort_values(by="X1")
    dy = (df_sorted["Y"].to_numpy()[1:] - df_sorted["Y"].to_numpy()[:-1]) / np.sqrt(2)
    dd = (df_sorted["D"].to_numpy()[1:] - df_sorted["D"].to_numpy()[:-1]) / np.sqrt(2)
    return manual_ols(dy, dd)


def est_dml_full(df, K=5, seed=42):
    Y, D = df["Y"].to_numpy(), df["D"].to_numpy()
    X = df[[f"X{j+1}" for j in range(10)]].to_numpy().astype(np.float32)
    
    indices = np.arange(len(Y))
    np.random.default_rng(seed).shuffle(indices)
    folds = np.array_split(indices, K)
    
    y_hat, d_hat = np.zeros(len(Y)), np.zeros(len(D))
    xgb_params = {"max_depth": 5, "eta": 0.05, "objective": "reg:squarederror", "nthread": 1}

    for k in range(K):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
        
        for target, hat_arr in zip([Y, D], [y_hat, d_hat]):
            dtr = xgb.DMatrix(X[train_idx], label=target[train_idx])
            model = xgb.train(xgb_params, dtr, num_boost_round=300) # 增加迭代次数
            hat_arr[test_idx] = model.predict(xgb.DMatrix(X[test_idx]))

    return manual_ols(Y - y_hat, D - d_hat)


def run_comprehensive_mc(R=20, n=5000, theta0=1.0):
    print(f"Starting Stress MC: R={R}, n={n}, g(X) involves multiple X")
    results = {"Robinson (X1)": [], "Speckman (X1)": [], "Yatchew (X1)": [], "DML (All X)": []}
    
    for r in range(R):
        df = generate_comprehensive_stress_data(n=n, theta=theta0, seed=r+100)
        results["Robinson (X1)"].append(est_robinson_1d(df))
        results["Speckman (X1)"].append(est_speckman_1d(df))
        results["Yatchew (X1)"].append(est_yatchew_1d(df))
        results["DML (All X)"].append(est_dml_full(df, seed=r))
        if (r+1) % 5 == 0: print(f"  -> Run {r+1}/{R} complete.")

    summary = []
    for method, vals in results.items():
        arr = np.array(vals)
        summary.append({
            "Method": method,
            "Mean": np.mean(arr),
            "Bias": np.mean(arr) - theta0,
            "RMSE": np.sqrt(np.mean((arr - theta0)**2)),
            "SD": np.std(arr, ddof=1)
        })
    return pd.DataFrame(summary).set_index("Method")

if __name__ == "__main__":

    final_res = run_comprehensive_mc(R=30, n=10000, theta0=1.0)
    print("\n" + "="*60)
    print("COMPREHENSIVE STRESS TEST RESULTS")
    print("="*60)
    print(final_res)
