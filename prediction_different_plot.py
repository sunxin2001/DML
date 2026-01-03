import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os



def load_data():
    file_path = "/Users/xinsun/Desktop/python/sipp1991.dta"


    df = pd.read_stata(file_path)
    df.columns = [c.lower() for c in df.columns]
    

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    test_sample = test_df.sample(500, random_state=42)


    y_train, d_train = train_df['net_tfa'].to_numpy(), train_df['e401'].to_numpy()
    y_test, d_test = test_sample['net_tfa'].to_numpy(), test_sample['e401'].to_numpy()
    inc_tr, inc_te = train_df['inc'].to_numpy(), test_sample['inc'].to_numpy()


    def nw_predict(tr_t, tr_f, te_f, bw):
        diff = te_f[:, np.newaxis] - tr_f[np.newaxis, :]
        w = np.exp(-0.5 * (diff / bw)**2)
        return (w @ tr_t) / (w.sum(axis=1) + 1e-10)


    h = 1.06 * np.std(inc_tr) * len(train_df)**(-1/5)
    y_res = y_train - nw_predict(y_train, inc_tr, inc_tr, h)
    d_res = d_train - nw_predict(d_train, inc_tr, inc_tr, h)
    theta_rob = (d_res @ y_res) / (d_res @ d_res)
    y_pred_rob = d_test * theta_rob + nw_predict(y_train - theta_rob * d_train, inc_tr, inc_te, h)
    rmse_rob = np.sqrt(np.mean((y_test - y_pred_rob)**2))


    x_cols = [c for c in ['inc', 'age', 'fsize', 'educ', 'pira', 'hown', 'marr', 'twoearn'] if c in df.columns]
    x_tr = train_df[x_cols].to_numpy().astype(np.float32)
    x_te = test_sample[x_cols].to_numpy().astype(np.float32)
    model_y = xgb.train({"max_depth": 3, "eta": 0.1, "objective": "reg:squarederror", "verbosity": 0}, 
                        xgb.DMatrix(x_tr, label=y_train), num_boost_round=100)
    model_d = xgb.train({"max_depth": 3, "eta": 0.1, "objective": "reg:squarederror", "verbosity": 0},
                        xgb.DMatrix(x_tr, label=d_train), num_boost_round=100)
    theta_dml = 8870.88
    g_hat_dml = model_y.predict(xgb.DMatrix(x_te)) - theta_dml * model_d.predict(xgb.DMatrix(x_te))
    y_pred_dml = d_test * theta_dml + g_hat_dml
    rmse_dml = np.sqrt(np.mean((y_test - y_pred_dml)**2))


    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))


    xlims = [-20000, 250000]
    ylims = [-40000, 300000]


    axes[0].scatter(y_test, y_pred_rob, alpha=0.6, color='royalblue', s=30, edgecolors='w', linewidth=0.5)
    axes[0].plot(xlims, ylims, 'r--', lw=2, label='Perfect Fit (45° Line)')
    axes[0].set_title(f'Robinson (1D: Income Only)\nOverall RMSE: {rmse_rob:,.0f}', fontsize=12)
    axes[0].set_xlabel('Actual Net Assets ($)', fontsize=11)
    axes[0].set_ylabel('Predicted Net Assets ($)', fontsize=11)
    axes[0].set_xlim(xlims) 
    axes[0].set_ylim(ylims) 
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper left')


    axes[1].scatter(y_test, y_pred_dml, alpha=0.6, color='forestgreen', s=30, edgecolors='w', linewidth=0.5)
    axes[1].plot(xlims, ylims, 'r--', lw=2, label='Perfect Fit (45° Line)')
    axes[1].set_title(f'DML (Full-X: High-Dimensional)\nOverall RMSE: {rmse_dml:,.0f}', fontsize=12)
    axes[1].set_xlabel('Actual Net Assets ($)', fontsize=11)

    axes[1].set_xlim(xlims) 
    axes[1].set_ylim(ylims) 
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper left')

    plt.suptitle('Prediction Accuracy Comparison: Highly Focused View on Core Data\n(Excludes more outliers visually, RMSE is for all data)', y=0.99, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    load_data()
