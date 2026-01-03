import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings




def load_data():
    file_path = "/Users/xinsun/Desktop/python/sipp1991.dta"
    
    df = pd.read_stata(file_path)
    df.columns = [c.lower() for c in df.columns]
    df['e401'] = df['e401'].astype(float)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df


def nw_predict_1d(train_target, train_feature, test_feature, bw):

    diff = test_feature[:, np.newaxis] - train_feature[np.newaxis, :]
    weights = np.exp(-0.5 * (diff / bw)**2)
    return (weights @ train_target) / (weights.sum(axis=1) + 1e-10)


def manual_ols(y, d):
    return float((d @ y) / (d @ d))

def evaluate_methods(train_df, test_df):
    results = []
    
 
    y_test = test_df['net_tfa'].to_numpy()
    d_test = test_df['e401'].to_numpy()
    inc_train = train_df['inc'].to_numpy()
    inc_test = test_df['inc'].to_numpy()
    h = 1.06 * np.std(inc_train) * len(train_df)**(-1/5)

   
    print("Evaluating Robinson...")
    y_train, d_train = train_df['net_tfa'].to_numpy(), train_df['e401'].to_numpy()
    y_res_tr = y_train - nw_predict_1d(y_train, inc_train, inc_train, h)
    d_res_tr = d_train - nw_predict_1d(d_train, inc_train, inc_train, h)
    theta_rob = manual_ols(y_res_tr, d_res_tr)
    g_hat_test = nw_predict_1d(y_train - theta_rob * d_train, inc_train, inc_test, h)
    y_pred_rob = d_test * theta_rob + g_hat_test
    
 
    print("Evaluating Yatchew...")
    df_s = train_df.sort_values(by='inc')
    dy = (df_s['net_tfa'].to_numpy()[1:] - df_s['net_tfa'].to_numpy()[:-1]) / np.sqrt(2)
    dd = (df_s['e401'].to_numpy()[1:] - df_s['e401'].to_numpy()[:-1]) / np.sqrt(2)
    theta_yat = manual_ols(dy, dd)
    g_hat_yat = nw_predict_1d(y_train - theta_yat * d_train, inc_train, inc_test, h)
    y_pred_yat = d_test * theta_yat + g_hat_yat


    print("Evaluating DML...")
    x_cols = ['inc', 'age', 'fsize', 'educ', 'pira', 'hown', 'marr', 'twoearn']
    x_train = train_df[x_cols].to_numpy().astype(np.float32)
    x_test = test_df[x_cols].to_numpy().astype(np.float32)
    

    params = {"max_depth": 3, "eta": 0.1, "objective": "reg:squarederror", "verbosity": 0}
    model_y = xgb.train(params, xgb.DMatrix(x_train, label=y_train), num_boost_round=100)
    model_d = xgb.train(params, xgb.DMatrix(x_train, label=d_train), num_boost_round=100)
    
    y_res_dml = y_train - model_y.predict(xgb.DMatrix(x_train))
    d_res_dml = d_train - model_d.predict(xgb.DMatrix(x_train))
    theta_dml = manual_ols(y_res_dml, d_res_dml)
    

    g_hat_dml = model_y.predict(xgb.DMatrix(x_test)) - theta_dml * model_d.predict(xgb.DMatrix(x_test))
    y_pred_dml = d_test * theta_dml + g_hat_dml


    preds = {
        "Robinson (1D)": y_pred_rob,
        "Speckman (1D)": y_pred_rob,
        "Yatchew (1D)": y_pred_yat,
        "DML (Full-X)": y_pred_dml
    }
    
    for method, y_pred in preds.items():
        results.append({
            "Method": method,
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R-Squared": r2_score(y_test, y_pred)
        })
    
    return pd.DataFrame(results).set_index("Method")


if __name__ == "__main__":
    train_df, test_df = load_and_prepare_data()
    comparison_table = evaluate_methods(train_df, test_df)
    
    print("\n" + "="*60)
    print("PREDICTION PERFORMANCE COMPARISON (on Test Set)")
    print("="*60)
    print(comparison_table)
    print("="*60)



def run_prediction_and_plot():
    file_path = "/Users/xinsun/Desktop/python/sipp1991.dta"
    df = pd.read_stata(file_path)
    df.columns = [c.lower() for c in df.columns]
    

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_sample = test_df.sample(500, random_state=42) # 取500个点防止散点图过拥挤


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


    x_cols = [c for c in ['inc', 'age', 'fsize', 'educ', 'pira', 'hown', 'marr', 'twoearn'] if c in df.columns]
    x_tr = train_df[x_cols].to_numpy().astype(np.float32)
    x_te = test_sample[x_cols].to_numpy().astype(np.float32)
    
    model_y = xgb.train({"max_depth": 3, "eta": 0.1, "objective": "reg:squarederror"}, 
                        xgb.DMatrix(x_tr, label=y_train), num_boost_round=100)

    theta_dml = 8870.87
    y_pred_dml = model_y.predict(xgb.DMatrix(x_te))


    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)


    axes[0].scatter(y_test, y_pred_rob, alpha=0.5, color='blue', s=20)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_title(f'Robinson (1D: Income Only)\nRMSE: {np.sqrt(np.mean((y_test-y_pred_rob)**2)):,.2f}')
    axes[0].set_xlabel('Actual Net Assets')
    axes[0].set_ylabel('Predicted Net Assets')


    axes[1].scatter(y_test, y_pred_dml, alpha=0.5, color='green', s=20)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_title(f'DML (Full-X: High-Dimensional)\nRMSE: {np.sqrt(np.mean((y_test-y_pred_dml)**2)):,.2f}')
    axes[1].set_xlabel('Actual Net Assets')

    plt.suptitle('Prediction Accuracy: Traditional 1D vs. High-Dimensional DML', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

run_prediction_and_plot()
