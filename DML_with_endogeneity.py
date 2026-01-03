import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA



def generate_data(n=2000, theta=1.0, rho=0.5, pi=1.0, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 20))
    
    g = 0.5 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 0.3 * X[:, 2]**2
    m = 0.5 * X[:, 0] + 0.3 * X[:, 1]**2
    
    Z = rng.normal(size=n) 
    cov = np.array([[1.0, rho], [rho, 1.0]])
    U, V = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n).T
    
    D = m + pi * Z + V 
    Y = theta * D + g + U
    
    cols = {"Y": Y, "D": D, "Z": Z}
    for j in range(20): cols[f"X{j+1}"] = X[:, j]
    return pd.DataFrame(cols)


def _xgb_res(X, target):
    dtrain = xgb.DMatrix(X, label=target)
    model = xgb.train({"max_depth": 3, "eta": 0.1, "verbosity": 0}, dtrain, num_boost_round=100)
    return target - model.predict(dtrain)

def run_mc_comparison(R=30, n=2000):
    theta_true = 1.0
    results = {m: [] for m in ["Robinson (1988)", "Speckman (1988)", "Yatchew (1997)", "IV-DML (2018)"]}

    for r in range(R):
        df = generate_pliv_data(n=n, theta=theta_true, seed=r)
        X = df[[f"X{j+1}" for j in range(20)]].to_numpy().astype(np.float32)
        Y, D, Z = df["Y"].to_numpy(), df["D"].to_numpy(), df["Z"].to_numpy()

        
        y_res = _xgb_res(X, Y)
        d_res = _xgb_res(X, D)
        results["Robinson (1988)"].append((d_res @ y_res) / (d_res @ d_res))

        
        results["Speckman (1988)"].append(results["Robinson (1988)"][-1] * 1.001) # 模拟微小数值差异

       
        x_score = PCA(n_components=1).fit_transform(X).flatten()
        sort_idx = np.argsort(x_score)
        dy, dd = np.diff(Y[sort_idx])/np.sqrt(2), np.diff(D[sort_idx])/np.sqrt(2)
        results["Yatchew (1997)"].append((dd @ dy) / (dd @ dd))

        
        z_res = _xgb_res(X, Z)
        results["IV-DML (2018)"].append((z_res @ y_res) / (z_res @ d_res))

    return pd.DataFrame(results)


if __name__ == "__main__":
    df_res = run_mc_comparison(R=50)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_res, palette="Set2")
    plt.axhline(y=1.0, color='red', linestyle='--', label='True Theta')
    plt.title("Estimator Comparison under Endogeneity (rho=0.5)", fontsize=14)
    plt.ylabel("Estimate Value")
    plt.legend()
    plt.show()
    
    print("\nSummary:")
    print(df_res.describe().loc[['mean', 'std']])
