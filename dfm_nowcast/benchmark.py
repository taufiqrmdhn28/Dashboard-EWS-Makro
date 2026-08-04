import os
import warnings
import numpy as np
import pandas as pd

from .utils import init_debug_dir, csvwrite_debug

def ar_realtime_gdp(x_new, i_ser, p_ar, i_q, native=False, debug_dir=None):
    """
    Replicates the MATLAB ML_ARrealtimeGDP function.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    target_dir = init_debug_dir('ML_ARrealtimeGDP', debug_dir=debug_dir)

    temp = np.isfinite(x_new[:, i_ser])
    gdp = x_new[temp, i_ser]

    beta = fit_autoregressive(gdp, 1, 4, p_ar, native=native, debug_dir=debug_dir)[0]

    start_idx = int(i_q - 6)
    end_idx = int(i_q - 6 - 3 * (p_ar - 1))
    
    indices = np.arange(start_idx, end_idx - 1, -3)
    xx = x_new[indices, i_ser].reshape(1, -1)

    csvwrite_debug(target_dir, 'temp_mask.csv', temp)
    csvwrite_debug(target_dir, 'GdP.csv', gdp)
    csvwrite_debug(target_dir, 'XX.csv', xx)
    csvwrite_debug(target_dir, 'beta.csv', beta)

    return beta, xx


# Legacy alias
ml_ar_realtime_gdp = ar_realtime_gdp


def fit_autoregressive(y, det, k, p, native=False, debug_dir=None):
    """
    Univariate autoregressive model estimation with automatic BIC lag selection.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    target_dir = init_debug_dir('ML_autoregressive', debug_dir=debug_dir)

    y = np.array(y).reshape(-1, 1)
    
    if p < 0:
        p_max = abs(p)
        bic = np.zeros((p_max, 1))
        
        for i in range(1, p_max + 1):
            t = y.shape[0]
            yy = y[i:t, 0].reshape(-1, 1)
            xx = np.zeros((t - i, i))
            
            for j in range(1, i + 1):
                xx[:, j - 1] = y[i - j : t - j, 0]
                
            beta, resid, _, _, r2, _ = fit_ols(yy, xx, det, native=native, debug_dir=debug_dir)
            
            num_params = xx.shape[1] + (1 if det > 0 else 0) + (1 if det > 1 else 0)
            bic[i - 1, 0] = np.log(np.var(resid)) + num_params * np.log(len(yy)) / len(yy)
            
        p = np.argmin(bic) + 1

    csvwrite_debug(target_dir, 'p_selected.csv', p)

    t = y.shape[0]
    yy = y[p:t, 0].reshape(-1, 1)
    xx = np.zeros((t - p, p))
    
    for j in range(1, p + 1):
        xx[:, j - 1] = y[p - j : t - j, 0]
        
    beta, resid, _, _, r2, _ = fit_ols(yy, xx, det, native=native, debug_dir=debug_dir)
    
    if det == 0:
        g = 1
    elif det == 2:
        g = 3
    else:
        g = 2
        
    beta_coefs = beta[g - 1 : p + g - 1, 0]
    al = np.zeros((p + 1, 1))
    al[0, 0] = 1.0
    al[1:, 0] = -beta_coefs

    csvwrite_debug(target_dir, 'yy.csv', yy)
    csvwrite_debug(target_dir, 'xx.csv', xx)
    csvwrite_debug(target_dir, 'beta.csv', beta)
    csvwrite_debug(target_dir, 'resid.csv', resid)
    csvwrite_debug(target_dir, 'r2.csv', r2)
    csvwrite_debug(target_dir, 'AL_flat.csv', al)

    return beta, resid, r2, al, p


# Legacy alias
ml_autoregressive = fit_autoregressive


def fit_ols(y, x, det, native=False, debug_dir=None):
    """
    Ordinary Least Squares (OLS) regression helper.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    target_dir = init_debug_dir('ML_ols', debug_dir=debug_dir)

    y = np.array(y).reshape(-1, 1)
    if x.ndim == 1:
        x = np.array(x).reshape(-1, 1)
        
    t, k = x.shape

    cons = np.ones((t, 1), dtype=float)
    trend = np.arange(1, t + 1, dtype=float).reshape(-1, 1)

    if det == 1:
        x_aug = np.hstack([cons, x])
    elif det == 2:
        x_aug = np.hstack([trend, x])
    elif det == 3:
        x_aug = np.hstack([cons, trend, x])
    else:
        x_aug = x.copy()

    if native:
        beta = np.linalg.lstsq(x_aug, y, rcond=None)[0]
        xx = np.linalg.inv(x_aug.T @ x_aug)
    else:
        xtx = x_aug.T @ x_aug
        xty = x_aug.T @ y
        xx = np.linalg.inv(xtx)
        beta = xx @ xty
    
    u = y - x_aug @ beta
    uu = (u.T @ u).item()
    
    esu = uu / (t - k)
    
    yc = y - np.mean(y)
    
    r2 = 1 - (uu / (yc.T @ yc).item())
    
    v = esu * xx
    espar = np.sqrt(np.diag(v)).reshape(-1, 1)

    csvwrite_debug(target_dir, 'beta.csv', beta)
    csvwrite_debug(target_dir, 'u.csv', u)
    csvwrite_debug(target_dir, 'v.csv', v)
    csvwrite_debug(target_dir, 'esu.csv', esu)
    csvwrite_debug(target_dir, 'r2.csv', r2)
    csvwrite_debug(target_dir, 'espar.csv', espar)

    return beta, u, v, esu, r2, espar


# Legacy alias
ml_ols = fit_ols


def update_benchmark_predictions(x_new, beta, xx, p_ar, i_q, i_ser, debug_dir=None):
    """
    Replicates the MATLAB ML_UpdateBenchmarkPredictions function.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    target_dir = init_debug_dir('ML_UpdateBenchmarkPredictions', debug_dir=debug_dir)
    
    i_q = int(i_q)
    i_ser = int(i_ser)
    p_ar = int(p_ar)
    
    if xx.ndim == 1:
        xx = xx.reshape(1, -1)

    # 1. Backcast (ARb) & Random Walk (RW)
    if np.isnan(x_new[i_q - 3, i_ser]):
        start_idx = i_q - 6
        end_idx = i_q - 6 - 3 * (p_ar - 1)
        
        indices = np.arange(start_idx, end_idx - 1, -3)
        x_slice = x_new[indices, i_ser].reshape(1, -1)
        
        regressor = np.hstack([[[1.0]], x_slice])
        
        arb = (regressor @ beta).item()
        rw = x_new[i_q - 6, i_ser].item()
    else:
        arb = x_new[i_q - 3, i_ser].item()
        rw = x_new[i_q - 3, i_ser].item()

    xx = np.hstack([[[arb]], xx])

    # 2. Nowcast (ARn)
    xx_slice_n = xx[0, :p_ar].reshape(1, -1)
    regressor_n = np.hstack([[[1.0]], xx_slice_n])
    arn = (regressor_n @ beta).item()

    xx = np.hstack([[[arn]], xx])

    # 3. Forecast (ARf)
    xx_slice_f = xx[0, :p_ar].reshape(1, -1)
    regressor_f = np.hstack([[[1.0]], xx_slice_f])
    arf = (regressor_f @ beta).item()

    csvwrite_debug(target_dir, 'ARb.csv', arb)
    csvwrite_debug(target_dir, 'ARn.csv', arn)
    csvwrite_debug(target_dir, 'ARf.csv', arf)
    csvwrite_debug(target_dir, 'RW.csv', rw)
    csvwrite_debug(target_dir, 'XX_out.csv', xx)

    return arb, arn, arf, rw, xx


# Legacy alias
ml_update_benchmark_predictions = update_benchmark_predictions


def compute_forecast_errors(fe_m, v_miss, release_time, t, y_new, i_ser, x_last, i_q, date_matrix, ar_back, ar_now, ar_fore, i_ser2=None, debug_dir=None):
    """
    Replicates the MATLAB ML_NowcastForecastError function.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    target_dir = init_debug_dir('ML_NowcastForecastError', t=t, debug_dir=debug_dir)

    if i_ser2 is None:
        i_ser2 = i_ser

    rt_val = release_time[t, 0] if isinstance(release_time, np.ndarray) else release_time[t]
    
    dt = pd.to_datetime(rt_val - 693960, unit='D', origin='1899-12-30')
    month = dt.month

    if month in [1, 4, 7, 10]:
        mm_idx = 0
    elif month in [2, 5, 8, 11]:
        mm_idx = 1
    else:
        mm_idx = 2

    i_q = int(i_q)
    i_ser = int(i_ser)
    i_ser2 = int(i_ser2)

    for var_idx in v_miss:
        var_idx = int(var_idx)
        
        date_back = date_matrix[i_q - 3, 0].item() if date_matrix.ndim > 1 else date_matrix[i_q - 3].item()
        date_now  = date_matrix[i_q, 0].item() if date_matrix.ndim > 1 else date_matrix[i_q].item()
        date_fore = date_matrix[i_q + 3, 0].item() if date_matrix.ndim > 1 else date_matrix[i_q + 3].item()
        
        err_back_dfm = (y_new[0, i_ser2] - x_last[i_q - 3, i_ser]).item()
        err_back_ar  = (ar_back - x_last[i_q - 3, i_ser]).item()
        
        err_now_dfm = (y_new[1, i_ser2] - x_last[i_q, i_ser]).item()
        err_now_ar  = (ar_now - x_last[i_q, i_ser]).item()
        
        err_fore_dfm = (y_new[2, i_ser2] - x_last[i_q + 3, i_ser]).item()
        err_fore_ar  = (ar_fore - x_last[i_q + 3, i_ser]).item()
        
        row = [
            rt_val,        err_back_dfm, err_back_ar, date_back,
            err_now_dfm,   err_now_ar,   date_now,
            err_fore_dfm,  err_fore_ar,  date_fore
        ]
        
        fe_m[var_idx][mm_idx].append(row)

        csvwrite_debug(target_dir, f'fe_m_v{var_idx + 1}_m{mm_idx + 1}.csv', fe_m[var_idx][mm_idx])

    return fe_m


# Legacy alias
ml_nowcast_forecast_error = compute_forecast_errors


def run_competition_benchmark(data_file="data/INO_130726.xlsx", start_est=None, start_eval="2016-01-01", last_eval="2026-03-31", output_dir="output/competition"):
    import time
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
    from dfm_python import DFM, DFMConfig, DFMDataset
    
    from dfm_nowcast.data import read_data, build_calendar, build_pseudo_real_time_vintages
    from dfm_nowcast.prediction import get_current_quarter, update_predictions
    from dfm_nowcast.estimation import fit_dfm_em, mixed_frequency_restrictions, idiosyncratic_law_of_motion
    
    if start_est is None:
        start_est = [2010, 1]
        
    print("=== Loading data and building calendar ===")
    X_Last, Series, ReleaseDay, SeriesQ, Date, nQ = read_data(data_file, start_est)
    T, N = X_Last.shape
    nM = N - nQ
    i_ser_idx = Series.index("RGDP_growth")

    name_vintage, PMV, MM, R_cal = build_calendar(data_file, start_eval, last_eval, nQ, hist_eval=True)
    
    # Filter to monthly frequency: select the last vintage of each month to keep runtime reasonable
    v_dates = pd.to_datetime(name_vintage - 693960, unit='D', origin='1899-12-30')
    df_v = pd.DataFrame({'vintage_val': name_vintage, 'date': v_dates, 'idx': range(len(name_vintage))})
    last_vintage_indices = df_v.groupby([df_v['date'].dt.year, df_v['date'].dt.month])['idx'].max().tolist()
    name_vintage = name_vintage[last_vintage_indices]

    DD, DDname, V, DatesV = build_pseudo_real_time_vintages(X_Last, Date, name_vintage, MM, R_cal, PMV)

    n_vintages = len(name_vintage)
    print(f"Total vintages to evaluate: {n_vintages}")

    horizons = ["Backcast", "Nowcast", "Forecast", "Forecast2S", "Forecast3S"]
    
    preds_nowcast = np.zeros((n_vintages, 5))
    preds_sm = np.zeros((n_vintages, 5))
    preds_dfmpy = np.zeros((n_vintages, 5))
    
    actuals = np.zeros((n_vintages, 5))
    vintage_names = []
    
    times_nowcast = np.zeros(n_vintages)
    times_sm = np.zeros(n_vintages)
    times_dfmpy = np.zeros(n_vintages)

    dates_dt = pd.to_datetime(Date - 693960, unit='D', origin='1899-12-30')

    print("\n=== Running Sequential Nowcasting Loop ===")
    for t in range(n_vintages):
        v_name = pd.to_datetime(name_vintage[t] - 693960, unit='D', origin='1899-12-30').strftime('%Y-%m-%d')
        vintage_names.append(v_name)
        
        X_new = DD[t].copy()
        nobs = X_new.shape[0]
        iQ = get_current_quarter(name_vintage, t, DatesV)
        slice_end = min(int(iQ + 10), nobs)
        target_indices = np.array([iQ - 3, iQ, iQ + 3, iQ + 6, iQ + 9], dtype=int)
        
        actuals[t, :] = X_Last[target_indices, i_ser_idx]
        
        print(f"\n[{t+1}/{n_vintages}] Vintage: {v_name} | nobs={nobs} | slice_end={slice_end} | iQ={iQ}")

        # --- 1. dfm-nowcast (Custom Model) ---
        t_start = time.time()
        try:
            t1_restr, t2_restr = mixed_frequency_restrictions("yoy")
            t1_idio = idiosyncratic_law_of_motion("Autoregressive", nM, nQ)
            P = {
                'p': 2,
                'blocks': np.ones((N, 1)),
                'r': np.array([2]),
                'dyn': np.array([2]),
                'nQ': nQ,
                'nM': nM,
                'Series': Series,
                'max_iter': 300,
                'thresh': 1e-3,
                'Rconstr': t1_restr,
                'q': t2_restr,
                'i_idio': t1_idio
            }
            R_new = fit_dfm_em(X_new[:slice_end, :], P, None)
            X_old = X_new.copy()
            y_nowcast, _, _, _ = update_predictions(X_old, X_new, X_Last, R_new, Date, iQ, i_ser_idx, nobs)
            preds_nowcast[t, :] = y_nowcast[:, i_ser_idx]
            times_nowcast[t] = time.time() - t_start
            print(f"  dfm-nowcast: {times_nowcast[t]:.3f}s | Nowcast={preds_nowcast[t, 1]:.3f}")
        except Exception as e:
            times_nowcast[t] = np.nan
            preds_nowcast[t, :] = np.nan
            print(f"  dfm-nowcast FAILED: {str(e)}")

        # --- 2. statsmodels DynamicFactorMQ ---
        t_start = time.time()
        try:
            df_sm = pd.DataFrame(X_new, index=dates_dt[:nobs], columns=Series)
            df_sm.index.freq = 'MS'
            df_sm_sliced = df_sm.iloc[:slice_end]
            
            model_sm = DynamicFactorMQ(df_sm_sliced, k_endog_monthly=nM, factors=2, factor_orders=2)
            res_sm = model_sm.fit_em(maxiter=100, tolerance=1e-3, disp=False)
            pred_sm = res_sm.predict(start=0, end=slice_end + 12)
            
            preds_sm[t, 0] = pred_sm.iloc[iQ - 3, i_ser_idx]
            preds_sm[t, 1] = pred_sm.iloc[iQ, i_ser_idx]
            preds_sm[t, 2] = pred_sm.iloc[iQ + 3, i_ser_idx]
            preds_sm[t, 3] = pred_sm.iloc[iQ + 6, i_ser_idx]
            preds_sm[t, 4] = pred_sm.iloc[iQ + 9, i_ser_idx]
            
            times_sm[t] = time.time() - t_start
            print(f"  statsmodels: {times_sm[t]:.3f}s | Nowcast={preds_sm[t, 1]:.3f}")
        except Exception as e:
            times_sm[t] = np.nan
            preds_sm[t, :] = np.nan
            print(f"  statsmodels FAILED: {str(e)}")

        # --- 3. dfm-python ---
        t_start = time.time()
        try:
            df_dataset = df_sm_sliced.reset_index().rename(columns={'index': 'date'})
            frequency_map = {}
            for col in Series[:nM]:
                frequency_map[col] = 'm'
            for col in Series[nM:]:
                frequency_map[col] = 'q'

            config_dfm = DFMConfig(
                frequency=frequency_map,
                clock='m',
                blocks={
                    'Block_Global': {
                        'num_factors': 2,
                        'series': Series
                    }
                },
                tent_weights={'q': [1.0, 2.0, 3.0, 2.0, 1.0]},
                threshold=1e-3,
                max_iter=100
            )
            dataset_dfm = DFMDataset(
                config=config_dfm,
                data=df_dataset,
                time_index='date'
            )
            model_dfm = DFM(dataset=dataset_dfm, config=config_dfm, scaler=StandardScaler())
            model_dfm.fit()
            
            x_sm_unscaled = model_dfm.scaler.inverse_transform(model_dfm.get_result().x_sm)
            
            preds_dfmpy[t, 0] = x_sm_unscaled[iQ - 3, i_ser_idx]
            preds_dfmpy[t, 1] = x_sm_unscaled[iQ, i_ser_idx]
            preds_dfmpy[t, 2] = x_sm_unscaled[iQ + 3, i_ser_idx]
            preds_dfmpy[t, 3] = x_sm_unscaled[iQ + 6, i_ser_idx]
            preds_dfmpy[t, 4] = x_sm_unscaled[iQ + 9, i_ser_idx]
            
            times_dfmpy[t] = time.time() - t_start
            print(f"  dfm-python: {times_dfmpy[t]:.3f}s | Nowcast={preds_dfmpy[t, 1]:.3f}")
        except Exception as e:
            times_dfmpy[t] = np.nan
            preds_dfmpy[t, :] = np.nan
            print(f"  dfm-python FAILED: {str(e)}")

    print("\n=== Processing Results ===")
    os.makedirs(output_dir, exist_ok=True)
    
    columns_raw = ["Vintage"] + [f"Actual_{h}" for h in horizons] + \
                  [f"dfm_nowcast_{h}" for h in horizons] + \
                  [f"statsmodels_{h}" for h in horizons] + \
                  [f"dfm_python_{h}" for h in horizons]
                  
    df_raw = pd.DataFrame(index=range(n_vintages))
    df_raw["Vintage"] = vintage_names
    for j, h in enumerate(horizons):
        df_raw[f"Actual_{h}"] = actuals[:, j]
        df_raw[f"dfm_nowcast_{h}"] = preds_nowcast[:, j]
        df_raw[f"statsmodels_{h}"] = preds_sm[:, j]
        df_raw[f"dfm_python_{h}"] = preds_dfmpy[:, j]
        
    df_raw["Time_dfm_nowcast"] = times_nowcast
    df_raw["Time_statsmodels"] = times_sm
    df_raw["Time_dfm_python"] = times_dfmpy
    
    raw_csv_path = os.path.join(output_dir, "competition_raw_results.csv")
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"Raw results saved to {raw_csv_path}")

    # Calculate RMSE & MAE for each model & horizon
    metrics = []
    
    for h_idx, h in enumerate(horizons):
        y_true = actuals[:, h_idx]
        
        mask_valid = ~np.isnan(y_true)
        if not np.any(mask_valid):
            continue
            
        y_true = y_true[mask_valid]
        
        for model_name, preds in [("dfm-nowcast", preds_nowcast), ("statsmodels", preds_sm), ("dfm-python", preds_dfmpy)]:
            y_pred = preds[mask_valid, h_idx]
            
            mask_pred = ~np.isnan(y_pred)
            if not np.any(mask_pred):
                rmse = np.nan
                mae = np.nan
            else:
                rmse = np.sqrt(np.mean((y_true[mask_pred] - y_pred[mask_pred]) ** 2))
                mae = np.mean(np.abs(y_true[mask_pred] - y_pred[mask_pred]))
                
            metrics.append({
                "Horizon": h,
                "Model": model_name,
                "RMSE": rmse,
                "MAE": mae
            })
            
    df_metrics = pd.DataFrame(metrics)
    metrics_csv_path = os.path.join(output_dir, "competition_metrics.csv")
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to {metrics_csv_path}")

    # Calculate average runtime
    runtime_summary = {
        "Model": ["dfm-nowcast", "statsmodels", "dfm-python"],
        "Mean_Time_Sec": [np.nanmean(times_nowcast), np.nanmean(times_sm), np.nanmean(times_dfmpy)],
        "Total_Time_Sec": [np.nansum(times_nowcast), np.nansum(times_sm), np.nansum(times_dfmpy)]
    }
    df_runtimes = pd.DataFrame(runtime_summary)
    runtimes_csv_path = os.path.join(output_dir, "competition_runtimes.csv")
    df_runtimes.to_csv(runtimes_csv_path, index=False)
    print(f"Runtimes saved to {runtimes_csv_path}")

    # Generate charts
    print("\n=== Generating Charts ===")
    
    # 1. Prediction comparison for Nowcast
    plt.figure(figsize=(10, 6))
    plt.plot(df_raw["Vintage"], df_raw["Actual_Nowcast"], label="Actual GDP Growth", color="black", linewidth=2.0)
    plt.plot(df_raw["Vintage"], df_raw["dfm_nowcast_Nowcast"], label="dfm-nowcast", linestyle="--", alpha=0.8)
    plt.plot(df_raw["Vintage"], df_raw["statsmodels_Nowcast"], label="statsmodels", linestyle=":", alpha=0.8)
    plt.plot(df_raw["Vintage"], df_raw["dfm_python_Nowcast"], label="dfm-python", linestyle="-.", alpha=0.8)
    plt.title("Sequential Nowcasts comparison against Actual GDP Growth")
    plt.xlabel("Vintage Date")
    plt.ylabel("GDP growth (%)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    chart_path1 = os.path.join(output_dir, "nowcast_comparison.png")
    plt.savefig(chart_path1, dpi=300)
    plt.close()
    print(f"Chart saved to {chart_path1}")

    # 2. Performance Metric Comparison (RMSE)
    plt.figure(figsize=(8, 5))
    pivot_rmse = df_metrics.pivot(index="Horizon", columns="Model", values="RMSE").reindex(horizons)
    pivot_rmse.plot(kind="bar", figsize=(10, 6))
    plt.title("Root Mean Squared Error (RMSE) by Horizon")
    plt.ylabel("RMSE (%)")
    plt.xlabel("Forecasting Horizon")
    plt.xticks(rotation=0)
    plt.legend()
    plt.tight_layout()
    chart_path2 = os.path.join(output_dir, "rmse_comparison.png")
    plt.savefig(chart_path2, dpi=300)
    plt.close()
    print(f"Chart saved to {chart_path2}")

    # 3. Execution Time Comparison
    plt.figure(figsize=(8, 5))
    plt.bar(df_runtimes["Model"], df_runtimes["Mean_Time_Sec"], color=["blue", "orange", "green"], alpha=0.7)
    plt.title("Average Execution Time per Vintage (Seconds)")
    plt.ylabel("Seconds")
    plt.tight_layout()
    chart_path3 = os.path.join(output_dir, "runtime_comparison.png")
    plt.savefig(chart_path3, dpi=300)
    plt.close()
    print(f"Chart saved to {chart_path3}")

    print("\n=== Benchmarking Loop Complete! ===")
    return df_metrics, df_runtimes
