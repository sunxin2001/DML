import numpy as np
import pandas as pd
import xgboost as xgb
import time


def generate_data(n=5000, theta=1.0, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 10))
    

    g = 1.2 * (X[:, 0] * X[:, 1]) + np.sin(np.pi * X[:, 2]) + 0.8 * X[:, 3]**2
    

    m = 1.0 * X[:, 0] + 1.2 * X[:, 1] + 1.0 * X[:, 2] + 0.8 * X[:, 3]
    D = m + rng.normal(size=n)
    
    Y = theta * D + g + rng.normal(size=n)
    
    cols = {"Y": Y, "D": D}
    for j in range(10):
        cols[f"X{j+1}"] = X[:, j]
    return pd.DataFrame(cols)

def est_dml_optimized(df, K=5, seed=42):
    Y, D = df["Y"].to_numpy(), df["D"].to_numpy()
    X = df[[f"X{j+1}" for j in range(10)]].to_numpy().astype(np.float32)
    
    indices = np.arange(len(Y))
    np.random.default_rng(seed).shuffle(indices)
    folds = np.array_split(indices, K)
    
    y_hat, d_hat = np.zeros(len(Y)), np.zeros(len(D))
    

    params = {
        "max_depth": 5, 
        "eta": 0.05, 
        "objective": "reg:squarederror", 
        "subsample": 0.8,
        "nthread": 1
    }

    for k in range(K):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
        
        for target, hat_arr in zip([Y, D], [y_hat, d_hat]):
            dtr = xgb.DMatrix(X[train_idx], label=target[train_idx])
    
            model = xgb.train(params, dtr, num_boost_round=300)
            hat_arr[test_idx] = model.predict(xgb.DMatrix(X[test_idx]))


    y_res, d_res = Y - y_hat, D - d_hat
    return float((d_res @ y_res) / (d_res @ d_res))


def nw_kernel_smooth(target, feature, bw):
    diff = feature[:, np.newaxis] - feature[np.newaxis, :]
    weights = np.exp(-0.5 * (diff / bw)**2)
    return (weights @ target) / weights.sum(axis=1)

def est_robinson_1d(df):
    """Robinson 估计量：错误地假设只有 X1 产生干扰"""
    Y, D, X1 = df["Y"].to_numpy(), df["D"].to_numpy(), df["X1"].to_numpy()
    h = 1.06 * np.std(X1) * len(Y)**(-1/5)
    y_tilde = Y - nw_kernel_smooth(Y, X1, h)
    d_tilde = D - nw_kernel_smooth(D, X1, h)
    return float((d_tilde @ y_tilde) / (d_tilde @ d_tilde))



def run_scaling_mc(n_list=[2000, 10000], R=20):
    all_summaries = []
    for n in n_list:
        print(f"\nRunning MC with n={n}...")
        results = {"Robinson (X1 only)": [], "DML (All X)": []}
        for r in range(R):
            df = generate_stress_dml_data(n=n, theta=1.0, seed=r+500)
            results["Robinson (X1 only)"].append(est_robinson_1d(df))
            results["DML (All X)"].append(est_dml_optimized(df, seed=r))
        
        for m, v in results.items():
            arr = np.array(v)
            all_summaries.append({
                "n": n, "Method": m, "Mean": np.mean(arr), 
                "Bias": np.mean(arr) - 1.0, "RMSE": np.sqrt(np.mean((arr-1.0)**2))
            })
    return pd.DataFrame(all_summaries)

if __name__ == "__main__":
    summary_df = run_scaling_mc(n_list=[2000, 10000], R=20)
    print("\n" + "="*60)
    print("SCALING MONTE CARLO RESULTS")
    print("="*60)
    print(summary_df)
