import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
import os



def load_data():
    file_path = "/Users/xinsun/Desktop/python/sipp1991.dta"

    

    df = pd.read_stata(file_path)
    

    df.columns = [c.lower() for c in df.columns]
    

    required = ['net_tfa', 'e401', 'inc']
    for req in required:
        if req not in df.columns:
            raise ValueError(f"数据中缺少核心变量: {req}")
            
    df['e401'] = df['e401'].astype(float)
    return df



def manual_ols(y, d):
    denom = d @ d
    if denom == 0: return 0
    return float((d @ y) / denom)

def nw_kernel_smooth_1d(target, feature, bw):
    diff = feature[:, np.newaxis] - feature[np.newaxis, :]
    weights = np.exp(-0.5 * (diff / bw)**2)
    return (weights @ target) / (weights.sum(axis=1) + 1e-10)

def est_robinson(df):
    Y, D, inc = df['net_tfa'].to_numpy(), df['e401'].to_numpy(), df['inc'].to_numpy()
    h = 1.06 * np.std(inc) * len(Y)**(-1/5)
    y_res = Y - nw_kernel_smooth_1d(Y, inc, h)
    d_res = D - nw_kernel_smooth_1d(D, inc, h)
    return manual_ols(y_res, d_res)

def est_yatchew(df):
    df_sorted = df.sort_values(by='inc')
    dy = (df_sorted['net_tfa'].to_numpy()[1:] - df_sorted['net_tfa'].to_numpy()[:-1]) / np.sqrt(2)
    dd = (df_sorted['e401'].to_numpy()[1:] - df_sorted['e401'].to_numpy()[:-1]) / np.sqrt(2)
    return manual_ols(dy, dd)

def est_dml(df, seed=42):
    Y, D = df['net_tfa'].to_numpy(), df['e401'].to_numpy()
    potential_x = ['inc', 'age', 'fsize', 'educ', 'pira', 'hown', 'marr', 'male', 'twoearn']
    available_x = [col for col in potential_x if col in df.columns]
    
    X = df[available_x].to_numpy().astype(np.float32)
    
    n, K = len(Y), 2
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    folds = np.array_split(indices, K)
    
    y_hat, d_hat = np.zeros(n), np.zeros(n)
    params = {"max_depth": 3, "eta": 0.1, "objective": "reg:squarederror", "nthread": 1, "verbosity": 0}

    for k in range(K):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
        for target, hat_arr in zip([Y, D], [y_hat, d_hat]):
            dtr = xgb.DMatrix(X[train_idx], label=target[train_idx])
            model = xgb.train(params, dtr, num_boost_round=50)
            hat_arr[test_idx] = model.predict(xgb.DMatrix(X[test_idx]))
            
    return manual_ols(Y - y_hat, D - d_hat)


def run_comparison_with_stats(df, n_boot=50):
    print(f"数据列名核对: {list(df.columns)}")
    print(f"开始点估计和 Bootstrap 过程 (n_boot={n_boot})...")
    

    original_ests = {
        "Robinson (1988)": est_robinson(df),
        "Speckman (1988)": est_robinson(df),
        "Yatchew (1997)": est_yatchew(df),
        "DML (2018)": est_dml(df)
    }
    
    boot_results = {m: [] for m in original_ests.keys()}
    
    for b in range(n_boot):
        boot_df = df.sample(frac=1.0, replace=True, random_state=b)
        
        try:
            boot_results["Robinson (1988)"].append(est_robinson(boot_df))
            boot_results["Speckman (1988)"].append(est_robinson(boot_df))
            boot_results["Yatchew (1997)"].append(est_yatchew(boot_df))
            boot_results["DML (2018)"].append(est_dml(boot_df, seed=b+100))
        except:
            continue
        
        if (b + 1) % 10 == 0:
            print(f"  已完成 {b + 1}/{n_boot} 次 Bootstrap 迭代...")

    summary = []
    for method, theta_hat in original_ests.items():
        boot_dist = np.array(boot_results[method])
        se = np.std(boot_dist, ddof=1) 
        t_stat = theta_hat / se
        p_val = 2 * (1 - norm.cdf(np.abs(t_stat))) 
        
        summary.append({
            "Method": method,
            "Estimate": theta_hat,
            "Std. Error": se,
            "t-stat": t_stat,
            "P-value": p_val
        })
        
    return pd.DataFrame(summary).set_index("Method")


if __name__ == "__main__":
    try:
        data = load_data()
        results = run_comparison_with_stats(data, n_boot=50)
        
        print("\n" + "="*80)
        print("实证对比结果：401(k) 资格对净金融资产 (net_tfa) 的处理效应")
        print("="*80)
        pd.options.display.float_format = '{:,.4f}'.format
        print(results)
        print("="*80)
        print("结论提示：")
        print("1. Robinson/Speckman 仅控制了收入 (inc) 的非参数趋势。")
        print(f"2. DML 控制了高维变量，包括: { [c for c in ['inc', 'age', 'fsize', 'educ', 'pira', 'hown', 'marr', 'male', 'twoearn'] if c in data.columns] }")
        print(f"3. Yatchew 估计量的渐近分布理论见 image_561a4b.png。")
        
    except Exception as e:
        print(f"\n程序运行出错: {e}")


import matplotlib.pyplot as plt
import numpy as np


methods = ['Robinson (1988)', 'Speckman (1988)', 'Yatchew (1997)', 'DML (2018)']
estimates = np.array([8842.2922, 8842.2922, 8564.9819, 8870.8752])
std_errors = np.array([1073.7764, 1073.7764, 1303.8355, 1094.8957])

ci_lower = estimates - 1.96 * std_errors
ci_upper = estimates + 1.96 * std_errors
error_bars = np.array([estimates - ci_lower, ci_upper - estimates])


plt.figure(figsize=(10, 6))


plt.errorbar(x=estimates, y=np.arange(len(methods)), xerr=error_bars, fmt='o', 
             color='black', ecolor='blue', capsize=5, label='95% CI')


mean_estimate = np.mean(estimates)
plt.axvline(x=mean_estimate, color='red', linestyle='--', label=f'Mean Estimate ({mean_estimate:.2f})')


plt.yticks(np.arange(len(methods)), methods)


plt.xlabel('Estimated 401(k) Eligibility Effect on Net Financial Assets')
plt.title('Forest Plot of Estimated Treatment Effects with 95% Confidence Intervals')


for i, est in enumerate(estimates):
    plt.text(est, i + 0.2, f"{est:.2f}", ha='center', va='center', fontsize=9)

plt.grid(axis='x', linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()

plt.show()
