import numpy as np
import pandas as pd
import xgboost as xgb
import time


def generate_data(n=2000, theta=1.0, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10))
    X1 = X[:, 0]
    

    g = 0.5 * np.sin(np.pi * X1) + 0.3 * X1**2
    
  
    m = 0.5 * X1 + 0.2 * X[:, 1]
    D = m + rng.normal(size=n)
    

    U = rng.normal(size=n)
    
    Y = theta * D + g + U
    
    cols = {"Y": Y, "D": D, "X1": X1}
    for j in range(1, 10):
        cols[f"X{j+1}"] = X[:, j]
    return pd.DataFrame(cols)


def manual_ols(y, d):
  
    return float((d @ y) / (d @ d))

def nw_kernel_smooth(target, feature, bw):
    n = len(feature)
 
    diff = feature[:, np.newaxis] - feature[np.newaxis, :]
    weights = np.exp(-0.5 * (diff / bw)**2) 
    row_sums = weights.sum(axis=1)
    return (weights @ target) / row_sums


def est_robinson(df):
    Y = df["Y"].to_numpy()
    D = df["D"].to_numpy()
    X1 = df["X1"].to_numpy()
    n = len(Y)
    

    h = 1.06 * np.std(X1) * n**(-1/4.5) 
    
    g_hat = nw_kernel_smooth(Y, X1, h)
    m_hat = nw_kernel_smooth(D, X1, h)
    
    y_tilde = Y - g_hat
    d_tilde = D - m_hat
    return manual_ols(y_tilde, d_tilde)


def est_speckman(df):
    Y = df["Y"].to_numpy()
    D = df["D"].to_numpy()
    X1 = df["X1"].to_numpy()
    n = len(Y)
    h = 1.06 * np.std(X1) * n**(-1/4.5)
    

    S_y = nw_kernel_smooth(Y, X1, h)
    S_d = nw_kernel_smooth(D, X1, h)
    
    y_tilde = Y - S_y
    d_tilde = D - S_d
    return manual_ols(y_tilde, d_tilde)


def est_yatchew(df):

    df_sorted = df.sort_values(by="X1")
    Y = df_sorted["Y"].to_numpy()
    D = df_sorted["D"].to_numpy()

    dy = (Y[1:] - Y[:-1]) / np.sqrt(2)
    dd = (D[1:] - D[:-1]) / np.sqrt(2)
    return manual_ols(dy, dd)

def est_dml(df, K=5, seed=42):
    Y = df["Y"].to_numpy()
    D = df["D"].to_numpy()
    X_cols = [f"X{j+1}" for j in range(10)]
    X = df[X_cols].to_numpy().astype(np.float32)
    
    indices = np.arange(len(Y))
    np.random.default_rng(seed).shuffle(indices)
    folds = np.array_split(indices, K)
    
    y_hat = np.zeros(len(Y))
    d_hat = np.zeros(len(D))
    
    params = {"max_depth": 3, "eta": 0.1, "objective": "reg:squarederror", "nthread": 1}

    for k in range(K):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
        

        dtrain_y = xgb.DMatrix(X[train_idx], label=Y[train_idx])
        model_y = xgb.train(params, dtrain_y, num_boost_round=100)
        y_hat[test_idx] = model_y.predict(xgb.DMatrix(X[test_idx]))
        
 
        dtrain_d = xgb.DMatrix(X[train_idx], label=D[train_idx])
        model_d = xgb.train(params, dtrain_d, num_boost_round=100)
        d_hat[test_idx] = model_d.predict(xgb.DMatrix(X[test_idx]))


    return manual_ols(Y - y_hat, D - d_hat)


def run_mc_comparison(R=30, n=10000, theta0=1.0):
    print(f"Starting MC Simulation: R={R}, n={n}, theta={theta0} (No Endogeneity)")
    
    results = {"Robinson": [], "Speckman": [], "Yatchew": [], "DML": []}
    
    for r in range(R):
        df = generate_plr_data(n=n, theta=theta0, seed=123+r)
        
        results["Robinson"].append(est_robinson(df))
        results["Speckman"].append(est_speckman(df))
        results["Yatchew"].append(est_yatchew(df))
        results["DML"].append(est_dml(df, seed=456+r))
        
        if (r+1) % 5 == 0: print(f"  Iteration {r+1}/{R} complete.")

    summary = []
    for method, vals in results.items():
        arr = np.array(vals)
        summary.append({
            "Method": method,
            "Mean Est": np.mean(arr),
            "Bias": np.mean(arr) - theta0,
            "RMSE": np.sqrt(np.mean((arr - theta0)**2)),
            "SD": np.std(arr, ddof=1)
        })
    
    return pd.DataFrame(summary).set_index("Method")

if __name__ == "__main__":

    res_df = run_mc_comparison(R=30, n=1000, theta0=1.0)
    print("\n" + "="*60)
    print("MONTE CARLO RESULTS (rho = 0)")
    print("="*60)
    print(res_df)
    print("="*60)
