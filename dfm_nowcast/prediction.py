import os
import warnings
import numpy as np
import pandas as pd

from .utils import init_debug_dir, csvwrite_debug
from .kalman import para_constdg

def update_predictions(x_old, x_new, x_last, r_new, date, i_q, i_ser, nobs, debug_dir=None):
    """
    Replicates the MATLAB ML_UpdatePrediction function.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    target_dir = init_debug_dir('ML_UpdatePrediction', debug_dir=debug_dir)

    x_old = np.array(x_old, copy=True, dtype=float)
    x_new = np.array(x_new, copy=True, dtype=float)
    x_last = np.array(x_last, copy=True, dtype=float)
    date = np.array(date, copy=True, dtype=float)
    
    i_q = int(i_q)
    i_ser = int(i_ser)
    nobs = int(nobs)

    target_len = int(i_q + 10) 
    
    if x_old.shape[0] < target_len:
        x_old = np.vstack([x_old, np.full((target_len - x_old.shape[0], x_old.shape[1]), np.nan)])
    if x_new.shape[0] < target_len:
        x_new = np.vstack([x_new, np.full((target_len - x_new.shape[0], x_new.shape[1]), np.nan)])
    if date.shape[0] < target_len:
        date = np.vstack([date, np.full((target_len - date.shape[0], date.shape[1]), np.nan)])
    if x_last.shape[0] < target_len:
        x_last = np.vstack([x_last, np.full((target_len - x_last.shape[0], x_last.shape[1]), np.nan)])

    if nobs < i_q + 4:
        x_old[nobs : i_q + 10, :] = np.nan
        x_new[nobs : i_q + 10, :] = np.nan
        date[i_q + 3, :] = np.nan
        x_last[i_q + 3, i_ser] = np.nan
        date[i_q + 6, :] = np.nan
        x_last[i_q + 6, i_ser] = np.nan
        date[i_q + 9, :] = np.nan
        x_last[i_q + 9, i_ser] = np.nan

    elif nobs < i_q + 7:
        x_old[nobs : i_q + 10, :] = np.nan
        x_new[nobs : i_q + 10, :] = np.nan
        date[i_q + 6, :] = np.nan
        x_last[i_q + 6, i_ser] = np.nan
        date[i_q + 9, :] = np.nan
        x_last[i_q + 9, i_ser] = np.nan

    elif nobs < i_q + 10:
        x_old[nobs : i_q + 10, :] = np.nan
        x_new[nobs : i_q + 10, :] = np.nan
        date[i_q + 9, :] = np.nan
        x_last[i_q + 9, i_ser] = np.nan

    x_old_slice = x_old[0 : int(i_q + 10), :]
    x_new_slice = x_new[0 : int(i_q + 10), :]

    target_indices = np.array([i_q - 3, i_q, i_q + 3, i_q + 6, i_q + 9], dtype=int)
    
    y_new, v_miss = compute_forecast_news(x_old_slice, x_new_slice, r_new, target_indices, i_ser, debug_dir=debug_dir)

    csvwrite_debug(target_dir, 'x_old.csv', x_old_slice)
    csvwrite_debug(target_dir, 'x_new.csv', x_new_slice)
    csvwrite_debug(target_dir, 'y_new.csv', y_new)
    csvwrite_debug(target_dir, 'v_miss.csv', v_miss)
    csvwrite_debug(target_dir, 'Date.csv', date)
    csvwrite_debug(target_dir, 'X_Last.csv', x_last)

    return y_new, v_miss, x_last, date


# Legacy alias
ml_update_prediction = update_predictions


def compute_forecast_news(x_old, x_new, q, t_fcst, v_news, debug_dir=None):
    """
    Replicates the MATLAB News_DFM_MLdgN3 function.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    target_dir = init_debug_dir('News_DFM_MLdgN3', debug_dir=debug_dir)
    
    miss_old = np.isnan(x_old)
    miss_new = np.isnan(x_new)
    temp = miss_old.astype(int) - miss_new.astype(int)

    v_miss, t_miss = np.nonzero(temp.T)

    res_new = para_constdg(x_new, q, 0, debug_dir=debug_dir)

    y_new = res_new['X_sm'][t_fcst, :]

    csvwrite_debug(target_dir, 't_miss.csv', t_miss + 1)
    csvwrite_debug(target_dir, 'v_miss.csv', v_miss + 1)
    csvwrite_debug(target_dir, 'y_new.csv', y_new)

    return y_new, v_miss


# Legacy alias
news_dfm_mldgn3 = compute_forecast_news


def compute_annual_nowcast(y_new, x_new, x_last, curr_dates, i_q, i_ser, debug_dir=None):
    """
    Replicates the MATLAB ML_AnnualNowcast function.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    target_dir = init_debug_dir('ML_AnnualNowcast', debug_dir=debug_dir)

    curr_dates_flat = np.array(curr_dates).flatten()
    month = curr_dates_flat[1]
    
    i_q = int(i_q)
    i_ser = int(i_ser)
    
    # ---------------------------------------------------------------------
    # FIRST QUARTER
    # ---------------------------------------------------------------------
    if month < 4:
        prediction = np.mean(y_new[1:5, i_ser]).item()
        target = np.mean(x_last[[i_q, i_q+3, i_q+6, i_q+9], i_ser]).item()
        
    # ---------------------------------------------------------------------
    # SECOND QUARTER
    # ---------------------------------------------------------------------
    elif 3 < month < 7:
        if np.isnan(x_new[i_q-3, i_ser]):
            prediction = np.mean(y_new[0:4, i_ser]).item()
        else:
            prediction = (x_last[i_q-3, i_ser] + np.sum(y_new[1:4, i_ser])) / 4.0
            
        target = np.mean(x_last[[i_q-3, i_q, i_q+3, i_q+6], i_ser]).item()
        
    # ---------------------------------------------------------------------
    # THIRD QUARTER
    # ---------------------------------------------------------------------
    elif 6 < month < 10:
        if np.isnan(x_new[i_q-3, i_ser]):
            prediction = (x_last[i_q-6, i_ser] + np.sum(y_new[0:3, i_ser])) / 4.0
        else:
            prediction = (np.sum(x_last[[i_q-6, i_q-3], i_ser]) + np.sum(y_new[1:3, i_ser])) / 4.0
            
        target = np.mean(x_last[[i_q-6, i_q-3, i_q, i_q+3], i_ser]).item()
        
    # ---------------------------------------------------------------------
    # FOURTH QUARTER
    # ---------------------------------------------------------------------
    elif month > 9:
        if np.isnan(x_new[i_q-3, i_ser]):
            prediction = (np.sum(x_last[[i_q-9, i_q-6], i_ser]) + np.sum(y_new[0:2, i_ser])) / 4.0
        else:
            prediction = (np.sum(x_last[[i_q-9, i_q-6, i_q-3], i_ser]) + y_new[1, i_ser]) / 4.0
            
        target = np.mean(x_last[[i_q-9, i_q-6, i_q-3, i_q], i_ser]).item()
        
    else:
        prediction = np.nan
        target = np.nan

    csvwrite_debug(target_dir, 'Prediction.csv', prediction)
    csvwrite_debug(target_dir, 'Target.csv', target)

    return prediction, target


# Legacy alias
ml_annual_nowcast = compute_annual_nowcast


def nowcast_current_output(current_output, release_time, y_new, date_matrix, v_miss, x_last, t, i_ser, i_q, debug_dir=None):
    """
    Replicates the MATLAB ML_NowcastCurrentOutput function.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    target_dir = init_debug_dir('ML_NowcastCurrentOutput', t=t, debug_dir=debug_dir)

    t = int(t)
    i_q = int(i_q)
    i_ser = int(i_ser)

    rt_val = release_time[t, 0] if isinstance(release_time, np.ndarray) else release_time[t]

    for var_idx in v_miss:
        var_idx = int(var_idx)
        
        d_back = date_matrix[i_q - 3, 0].item() if date_matrix.ndim > 1 else date_matrix[i_q - 3].item()
        d_now  = date_matrix[i_q, 0].item()     if date_matrix.ndim > 1 else date_matrix[i_q].item()
        d_f1   = date_matrix[i_q + 3, 0].item() if date_matrix.ndim > 1 else date_matrix[i_q + 3].item()
        d_f2   = date_matrix[i_q + 6, 0].item() if date_matrix.ndim > 1 else date_matrix[i_q + 6].item()
        d_f3   = date_matrix[i_q + 9, 0].item() if date_matrix.ndim > 1 else date_matrix[i_q + 9].item()

        row_back = [rt_val, y_new[0, i_ser].item(), d_back, var_idx + 1, x_last[i_q - 3, i_ser].item()]
        row_now  = [rt_val, y_new[1, i_ser].item(), d_now,  var_idx + 1, x_last[i_q, i_ser].item()]
        row_f1   = [rt_val, y_new[2, i_ser].item(), d_f1,   var_idx + 1, x_last[i_q + 3, i_ser].item()]
        row_f2   = [rt_val, y_new[3, i_ser].item(), d_f2,   var_idx + 1, x_last[i_q + 6, i_ser].item()]
        row_f3   = [rt_val, y_new[4, i_ser].item(), d_f3,   var_idx + 1, x_last[i_q + 9, i_ser].item()]

        current_output[t][0].append(row_back)
        current_output[t][1].append(row_now)
        current_output[t][2].append(row_f1)
        current_output[t][3].append(row_f2)
        current_output[t][4].append(row_f3)

    for h_idx in range(5):
        csvwrite_debug(target_dir, f'current_output_t{t + 1}_h{h_idx + 1}.csv', current_output[t][h_idx])

    return current_output


# Legacy alias
ml_nowcast_current_output = nowcast_current_output


def get_current_quarter(name_vintage, t, dates_v, debug_dir=None):
    """
    Replicates the MATLAB ML_CurrentQuarter function.
    Identifies the end-month of the current quarter based on the vintage date.
    Note: 't' should be passed as a 0-based Python loop index.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    target_dir = init_debug_dir('ML_CurrentQuarter', t=t, debug_dir=debug_dir)
    
    val = name_vintage[t]
    dt = pd.to_datetime(val - 693960, unit='D', origin='1899-12-30')
    
    temp_initial = np.array([[dt.year, dt.month]], dtype=float)
    
    month = temp_initial[0, 1]
    if month % 3 == 1:
        month += 2
    elif month % 3 == 2:
        month += 1
        
    qnews = np.array([[dt.year, month]], dtype=float)

    idx_arr = np.where((dates_v[:, 0] == qnews[0, 0]) & (dates_v[:, 1] == qnews[0, 1]))[0]
    
    if len(idx_arr) > 0:
        iq = idx_arr[0]
        iq_export = iq + 1
    else:
        iq = np.nan
        iq_export = np.nan

    export_t = t + 1 
    
    csvwrite_debug(target_dir, f'temp_initial_t{export_t}.csv', temp_initial)
    csvwrite_debug(target_dir, f'Qnews_t{export_t}.csv', qnews)
    csvwrite_debug(target_dir, f'iQ_t{export_t}.csv', iq_export)

    return int(iq) if not np.isnan(iq) else np.nan


# Legacy alias
ml_current_quarter = get_current_quarter


def estimate_breach_probability(res_smooth, R_new, i_ser_idx, target_idx, threshold, X_new=None, dist='gaussian', direction='lower'):
    """
    Estimates the probability of a target variable (i_ser_idx) breaching a threshold at target_idx.
    Uses the smoothed mean and variance computed from the Kalman smoother.
    
    Parameters:
        res_smooth (dict): The dictionary returned by para_constdg.
        R_new (dict): The estimated DFM parameters dictionary.
        i_ser_idx (int): The index of the indicator variable.
        target_idx (int): The time index/horizon.
        threshold (float): The threshold value.
        X_new (np.ndarray, optional): The dataset matrix containing actual values to fit non-Gaussian distributions.
        dist (str): The distribution to use ('gaussian', 'student-t', 'skew-normal', 'johnsonsu', 'empirical').
        direction (str): 'lower' (value < threshold) or 'higher' (value > threshold) (default: 'lower').
        
    Returns:
        prob (float): The probability of the value breaching the threshold.
        mean (float): The mean of the distribution in original units.
        std_dev (float): The standard deviation of the distribution in original units.
    """
    C = R_new['C']
    R = R_new['R']
    Wx = R_new['Wx']
    
    # 1. Mean in original units
    X_sm = res_smooth['X_sm']
    mean = X_sm[target_idx, i_ser_idx]
    
    # 2. Variance in standardized units: C_i * P_{h+1} * C_i^T + R_{i,i}
    Vsmooth = res_smooth['P']
    P_h = Vsmooth[:, :, target_idx + 1]
    c_i = C[i_ser_idx, :]
    var_std = c_i @ P_h @ c_i.T + R[i_ser_idx, i_ser_idx]
    
    # 3. Standard deviation in original units
    std_dev = Wx[0, i_ser_idx] * np.sqrt(var_std)
    
    # 4. Probability
    if std_dev < 1e-12:
        if direction == 'lower':
            prob = 1.0 if mean < threshold else 0.0
        else:
            prob = 1.0 if mean > threshold else 0.0
    else:
        z = (threshold - mean) / std_dev
        cdf_val = compute_non_gaussian_cdf(z, dist, X_new, R_new, i_ser_idx)
        if direction == 'lower':
            prob = cdf_val
        else:
            prob = 1.0 - cdf_val
        
    return prob, mean, std_dev


def estimate_probability_below(res_smooth, R_new, i_ser_idx, target_idx, threshold, X_new=None, dist='gaussian'):
    """
    Deprecated: use estimate_breach_probability instead.
    """
    return estimate_breach_probability(res_smooth, R_new, i_ser_idx, target_idx, threshold, X_new, dist, direction='lower')


def compute_filter_innovations(X_new, R_new, i_ser_idx):
    """
    Computes historical standardized filter innovations (one-step-ahead forecasting errors)
    for the series at i_ser_idx.
    """
    Z_0, V_0 = R_new['Z_0'], R_new['V_0']
    A, C = R_new['A'], R_new['C']
    Q, R = R_new['Q'], R_new['R']
    Mx, Wx = R_new['Mx'], R_new['Wx']
    
    from dfm_nowcast.data import DFMPreprocessor
    preprocessor = DFMPreprocessor()
    preprocessor.Mx = Mx
    preprocessor.Wx = Wx
    xNaN = preprocessor.transform(X_new)
    y = xNaN.T
    
    n, m = C.shape
    nobs = y.shape[1]
    
    Au = Z_0.copy().reshape(-1, 1)
    Pu = V_0.copy()
    
    c_i = C[i_ser_idx, :]
    r_ii = R[i_ser_idx, i_ser_idx]
    scale = Wx[0, i_ser_idx]
    
    innovations = []
    
    def miss_data(y_vec, C_mat, R_mat):
        ix = ~np.isnan(y_vec).flatten()
        return y_vec[ix].reshape(-1, 1), C_mat[ix, :], R_mat[ix][:, ix]
    
    for t in range(nobs):
        # 1-step ahead prediction
        A_pred = A @ Au
        P_pred = A @ Pu @ A.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        
        # Target variable observed?
        y_val = y[i_ser_idx, t]
        if not np.isnan(y_val):
            y_pred = c_i @ A_pred
            var_pred = c_i @ P_pred @ c_i.T + r_ii
            std_pred = scale * np.sqrt(max(1e-12, var_pred))
            
            Y_val = X_new[t, i_ser_idx]
            Y_pred = scale * y_pred.item() + Mx[0, i_ser_idx]
            
            u_t = (Y_val - Y_pred) / std_pred
            innovations.append(u_t)
            
        # Update step
        y_t, Z_t, R_t = miss_data(y[:, t:t+1], C, R)
        if y_t.size == 0:
            Au = A_pred
            Pu = P_pred
        else:
            PZ = P_pred @ Z_t.T
            iF = np.linalg.inv(Z_t @ PZ + R_t)
            PZF = PZ @ iF
            V = y_t - Z_t @ A_pred
            Au = A_pred + PZF @ V
            Pu = P_pred - PZF @ PZ.T
            Pu = 0.5 * (Pu + Pu.T)
            
    innov = np.array(innovations)
    return innov[~np.isnan(innov) & ~np.isinf(innov)]


def compute_non_gaussian_cdf(z, dist, X_new, R_new, i_ser_idx):
    """
    Fits the requested non-Gaussian distribution to the historical filter innovations
    and returns the CDF at z.
    """
    import scipy.stats as stats
    
    if dist == 'gaussian':
        return stats.norm.cdf(z)
        
    if X_new is None:
        raise ValueError(f"X_new must be provided to fit the non-Gaussian distribution '{dist}' to innovations.")
        
    innovations = compute_filter_innovations(X_new, R_new, i_ser_idx)
    if len(innovations) < 3:
        warnings.warn("Not enough historical observations to fit non-Gaussian distribution. Falling back to Gaussian.")
        return stats.norm.cdf(z)
        
    if dist == 'student-t':
        t_df, t_loc, t_scale = stats.t.fit(innovations)
        return stats.t.cdf(z, df=t_df, loc=t_loc, scale=t_scale)
        
    elif dist == 'skew-normal':
        sn_a, sn_loc, sn_scale = stats.skewnorm.fit(innovations)
        return stats.skewnorm.cdf(z, a=sn_a, loc=sn_loc, scale=sn_scale)
        
    elif dist == 'johnsonsu':
        jsu_a, jsu_b, jsu_loc, jsu_scale = stats.johnsonsu.fit(innovations)
        return stats.johnsonsu.cdf(z, a=jsu_a, b=jsu_b, loc=jsu_loc, scale=jsu_scale)
        
    elif dist == 'empirical':
        return np.mean(innovations <= z)
        
    elif dist == 'auto':
        best_dist = 'gaussian'
        best_aic = float('inf')
        best_cdf_val = stats.norm.cdf(z)
        
        # Gaussian (k = 2)
        try:
            g_loc, g_scale = stats.norm.fit(innovations)
            g_loglik = np.sum(stats.norm.logpdf(innovations, loc=g_loc, scale=g_scale))
            g_aic = 2 * 2 - 2 * g_loglik
            if not np.isnan(g_aic) and not np.isinf(g_aic) and g_aic < best_aic:
                best_aic = g_aic
                best_dist = 'gaussian'
                best_cdf_val = stats.norm.cdf(z, loc=g_loc, scale=g_scale)
        except Exception:
            pass
            
        # Student's-t (k = 3)
        try:
            t_df, t_loc, t_scale = stats.t.fit(innovations)
            t_loglik = np.sum(stats.t.logpdf(innovations, df=t_df, loc=t_loc, scale=t_scale))
            t_aic = 2 * 3 - 2 * t_loglik
            if not np.isnan(t_aic) and not np.isinf(t_aic) and t_aic < best_aic:
                best_aic = t_aic
                best_dist = 'student-t'
                best_cdf_val = stats.t.cdf(z, df=t_df, loc=t_loc, scale=t_scale)
        except Exception:
            pass
            
        # Skew-Normal (k = 3)
        try:
            sn_a, sn_loc, sn_scale = stats.skewnorm.fit(innovations)
            sn_loglik = np.sum(stats.skewnorm.logpdf(innovations, a=sn_a, loc=sn_loc, scale=sn_scale))
            sn_aic = 2 * 3 - 2 * sn_loglik
            if not np.isnan(sn_aic) and not np.isinf(sn_aic) and sn_aic < best_aic:
                best_aic = sn_aic
                best_dist = 'skew-normal'
                best_cdf_val = stats.skewnorm.cdf(z, a=sn_a, loc=sn_loc, scale=sn_scale)
        except Exception:
            pass
            
        # Johnson's SU (k = 4)
        try:
            jsu_a, jsu_b, jsu_loc, jsu_scale = stats.johnsonsu.fit(innovations)
            jsu_loglik = np.sum(stats.johnsonsu.logpdf(innovations, a=jsu_a, b=jsu_b, loc=jsu_loc, scale=jsu_scale))
            jsu_aic = 2 * 4 - 2 * jsu_loglik
            if not np.isnan(jsu_aic) and not np.isinf(jsu_aic) and jsu_aic < best_aic:
                best_aic = jsu_aic
                best_dist = 'johnsonsu'
                best_cdf_val = stats.johnsonsu.cdf(z, a=jsu_a, b=jsu_b, loc=jsu_loc, scale=jsu_scale)
        except Exception:
            pass
            
        return best_cdf_val
        
    else:
        raise ValueError(f"Unknown distribution type: {dist}")


def estimate_annual_breach_probability(res_smooth, R_new, i_ser_idx, target_year, dates, data_input, threshold, X_new=None, dist='gaussian', direction='lower'):
    """
    Estimates the probability that the annual growth rate of a target indicator in target_year breaches a threshold.
    Correctly calculates the annual growth rate as:
        Growth_Y = Sum(GDP_Y,q) / Sum(GDP_Y-1,q) - 1
    where unobserved quarterly levels are estimated using (1 + growth%) * level_Y-1,q.
    
    Parameters:
        res_smooth (dict): Smoother output dictionary from para_constdg.
        R_new (dict): Estimated DFM parameter dictionary.
        i_ser_idx (int): Index of the target indicator in the series list.
        target_year (int): Calendar year (e.g. 2026).
        dates (np.ndarray): 1D array of MATLAB datenums corresponding to data rows.
        data_input (str or dict): Path to input Excel/CSV data.
        threshold (float): Target threshold value (e.g. 5.0).
        X_new (np.ndarray, optional): Vintage data matrix.
        dist (str): Distribution type ('gaussian', etc.).
        direction (str): 'lower' (value < threshold) or 'higher' (value > threshold) (default: 'lower').
        
    Returns:
        prob (float): Probability that the annual rate breaches the threshold.
        mean_annual (float): Expected annual rate in percent.
        std_dev_annual (float): Standard deviation of the annual rate in percent.
        observed (list of bool): True if the quarter was observed in the raw data, False otherwise.
        levels_used (list of float): GDP levels used for each of the four quarters of target_year.
    """
    from scipy.stats import norm
    from dfm_nowcast.data import load_data_input
    
    C = R_new['C']
    R = R_new['R']
    Wx = R_new['Wx']
    A = R_new['A']
    P_smooth = res_smooth['P']
    X_sm = res_smooth['X_sm']
    
    # 1. Load QuarterlyData to get levels
    data_dict = load_data_input(data_input)
    df_q = data_dict['QuarterlyData']
    
    # Parse dates of QuarterlyData
    dates_qv_excel = df_q.iloc[:, 0]
    if not pd.api.types.is_numeric_dtype(dates_qv_excel) and not pd.api.types.is_datetime64_any_dtype(dates_qv_excel):
        dates_qv_dt = pd.to_datetime(dates_qv_excel)
    elif pd.api.types.is_datetime64_any_dtype(dates_qv_excel):
        dates_qv_dt = dates_qv_excel
    else:
        dates_qv_dt = pd.to_datetime(dates_qv_excel, unit='D', origin='1899-12-30')
        
    df_q['Year'] = dates_qv_dt.dt.year
    df_q['Month'] = dates_qv_dt.dt.month
    
    # Extract levels for previous year (target_year - 1)
    df_prev = df_q[df_q['Year'] == (target_year - 1)].sort_values('Month')
    if len(df_prev) < 4:
        raise ValueError(f"Could not find 4 quarters of GDP level data for year {target_year - 1}")
    levels_prev = df_prev['RGDP_level'].values
    sum_prev = np.sum(levels_prev)
    
    # Extract levels for target_year
    df_curr = df_q[df_q['Year'] == target_year].sort_values('Month')
    levels_curr = df_curr['RGDP_level'].values
    
    # Find matching time indices in the model's monthly date vector
    dates_flat = dates.flatten()
    dates_dt = pd.to_datetime(dates_flat - 693960, unit='D', origin='1899-12-30')
    
    q_months = [3, 6, 9, 12]
    q_indices = []
    for m in q_months:
        idx_arr = np.where((dates_dt.year == target_year) & (dates_dt.month == m))[0]
        if len(idx_arr) > 0:
            q_indices.append(idx_arr[0])
        else:
            raise ValueError(f"Quarter ending in month {m} of year {target_year} not found in dates.")
            
    # 2. Determine which quarters are observed vs forecasted
    observed = []
    levels_used = []
    
    for i, idx in enumerate(q_indices):
        is_obs = False
        if X_new is not None:
            if idx < X_new.shape[0] and not np.isnan(X_new[idx, i_ser_idx]):
                is_obs = True
        else:
            val_level = levels_curr[i] if i < len(levels_curr) else np.nan
            if not np.isnan(val_level):
                is_obs = True
                
        if is_obs:
            observed.append(True)
            val_level = levels_curr[i] if i < len(levels_curr) else np.nan
            levels_used.append(val_level)
        else:
            observed.append(False)
            y_forecast = X_sm[idx, i_ser_idx]
            levels_used.append(levels_prev[i] * (1.0 + y_forecast / 100.0))
            
    sum_curr_mean = np.sum(levels_used)
    mean_annual_growth = (sum_curr_mean / sum_prev - 1.0) * 100.0
    
    # 3. Calculate weights for the unobserved quarterly growth rates
    weights = levels_prev / sum_prev
    
    # 4. Compute covariance of unobserved quarters in standardized units
    cov_std = np.zeros((4, 4))
    for j in range(4):
        for k in range(4):
            if observed[j] or observed[k]:
                cov_std[j, k] = 0.0
                continue
                
            idx_j = q_indices[j]
            idx_k = q_indices[k]
            
            P_j = P_smooth[:, :, idx_j + 1]
            c_i = C[i_ser_idx, :]
            
            if j == k:
                cov_std[j, k] = c_i @ P_j @ c_i.T + R[i_ser_idx, i_ser_idx]
            elif j < k:
                steps = idx_k - idx_j
                A_steps = np.linalg.matrix_power(A, steps)
                P_jk = P_j @ A_steps.T
                cov_std[j, k] = c_i @ P_jk @ c_i.T
            else:
                cov_std[j, k] = cov_std[k, j]
                
    # 5. Convert variance to original units and compute standard deviation
    scale = Wx[0, i_ser_idx]
    var_growth = 0.0
    for j in range(4):
        for k in range(4):
            var_growth += weights[j] * weights[k] * cov_std[j, k]
            
    var_growth = (scale ** 2) * var_growth
    std_dev_growth = np.sqrt(var_growth)
    
    # 6. Compute probability
    if std_dev_growth < 1e-12:
        if direction == 'lower':
            prob = 1.0 if mean_annual_growth < threshold else 0.0
        else:
            prob = 1.0 if mean_annual_growth > threshold else 0.0
    else:
        z = (threshold - mean_annual_growth) / std_dev_growth
        cdf_val = compute_non_gaussian_cdf(z, dist, X_new, R_new, i_ser_idx)
        if direction == 'lower':
            prob = cdf_val
        else:
            prob = 1.0 - cdf_val
        
    return prob, mean_annual_growth, std_dev_growth, observed, levels_used


def estimate_annual_probability_below(res_smooth, R_new, i_ser_idx, target_year, dates, data_input, threshold, X_new=None, dist='gaussian'):
    """
    Deprecated: use estimate_annual_breach_probability instead.
    """
    return estimate_annual_breach_probability(res_smooth, R_new, i_ser_idx, target_year, dates, data_input, threshold, X_new, dist, direction='lower')


def estimate_sequential_breach_probability(
    res_smooth, R_new, i_ser_idx, current_idx, dates,
    threshold=0.0, direction='lower', n_consecutive=2,
    horizon=4, k_offset=0, n_simulations=100000
):
    """
    Estimates the probability of sequential growth rates breaching a threshold.
    Useful for technical recession forecasting (e.g. 2 consecutive quarters of negative growth).
    
    Parameters:
        res_smooth (dict): Kalman smoother output from para_constdg.
        R_new (dict): Estimated DFM parameters.
        i_ser_idx (int): Index of the target variable (e.g. RGDP_growth).
        current_idx (int): The monthly date index corresponding to the current quarter (iQ).
        dates (np.ndarray): 1D array of MATLAB datenums.
        threshold (float): Breach threshold value (default: 0.0).
        direction (str): 'lower' (value < threshold) or 'higher' (value > threshold) (default: 'lower').
        n_consecutive (int): Number of consecutive quarters required for a breach (default: 2).
        horizon (int): Number of future quarters to evaluate (default: 4).
        k_offset (int): Number of quarters to skip before starting the evaluation (default: 0).
        n_simulations (int): Number of Monte Carlo simulations to run (default: 100,000).
        
    Returns:
        prob (float): The probability of the sequential breach event occurring.
        means (list of float): Predicted growth rate means for each evaluated quarter.
        cov_matrix (np.ndarray): Covariance matrix of the evaluated quarters in original units.
    """
    import warnings
    
    # Target quarter indices in the monthly series dates are spaced 3 months apart
    q_indices = [current_idx + 3 * (k_offset + i) for i in range(horizon)]
    
    # Filter indices that are within the valid date range of the dataset
    valid_q_indices = []
    for idx in q_indices:
        if 0 <= idx < res_smooth['X_sm'].shape[0]:
            valid_q_indices.append(idx)
            
    if len(valid_q_indices) < n_consecutive:
        warnings.warn(f"Not enough future forecasting quarters ({len(valid_q_indices)}) "
                      f"available to compute sequential breach of length {n_consecutive}.")
        return 0.0, [], np.array([])
        
    # 1. Compute means
    X_sm = res_smooth['X_sm']
    means = np.array([X_sm[idx, i_ser_idx] for idx in valid_q_indices])
    
    # 2. Compute covariance matrix
    C = R_new['C']
    R = R_new['R']
    Wx = R_new['Wx']
    A = R_new['A']
    P_smooth = res_smooth['P']
    
    m = len(valid_q_indices)
    cov_std = np.zeros((m, m))
    for j in range(m):
        for k in range(m):
            idx_j = valid_q_indices[j]
            idx_k = valid_q_indices[k]
            P_j = P_smooth[:, :, idx_j + 1]
            c_i = C[i_ser_idx, :]
            
            if j == k:
                cov_std[j, k] = c_i @ P_j @ c_i.T + R[i_ser_idx, i_ser_idx]
            elif j < k:
                steps = idx_k - idx_j
                A_steps = np.linalg.matrix_power(A, steps)
                P_jk = P_j @ A_steps.T
                cov_std[j, k] = c_i @ P_jk @ c_i.T
            else:
                cov_std[j, k] = cov_std[k, j]
                
    scale = Wx[0, i_ser_idx]
    cov_orig = (scale ** 2) * cov_std
    
    # 3. Monte Carlo Simulation from Multivariate Normal
    try:
        draws = np.random.multivariate_normal(means, cov_orig, size=n_simulations)
    except Exception:
        # Fallback with a tiny diagonal ridge for numerical stability
        cov_ridge = cov_orig + np.eye(m) * 1e-8
        draws = np.random.multivariate_normal(means, cov_ridge, size=n_simulations)
        
    # 4. Check consecutive breaches
    if direction == 'lower':
        breached = draws < threshold
    else:
        breached = draws > threshold
        
    success_count = 0
    for row in breached:
        consecutive = 0
        has_breach = False
        for val in row:
            if val:
                consecutive += 1
                if consecutive >= n_consecutive:
                    has_breach = True
                    break
            else:
                consecutive = 0
        if has_breach:
            success_count += 1
            
def compute_probability_bins(mean, std_dev, bins=(5.0, 5.35), dist='gaussian'):
    """
    Computes mutually exclusive probability distribution bins for a given mean and standard deviation.
    
    Parameters:
        mean (float): Point estimate (e.g. Nowcast or Annual Nowcast).
        std_dev (float): Standard deviation of the estimate.
        bins (tuple): Threshold boundaries, default (5.0, 5.35).
        dist (str): Distribution type (default 'gaussian').
        
    Returns:
        dict: Probabilities for '< 5.0', '5.0 - 5.35', and '> 5.35'.
    """
    from scipy.stats import norm
    b0, b1 = bins[0], bins[1]
    if std_dev < 1e-12:
        p_below = 1.0 if mean < b0 else 0.0
        p_between = 1.0 if b0 <= mean <= b1 else 0.0
        p_above = 1.0 if mean > b1 else 0.0
    else:
        p_below = norm.cdf(b0, loc=mean, scale=std_dev)
        p_between = norm.cdf(b1, loc=mean, scale=std_dev) - norm.cdf(b0, loc=mean, scale=std_dev)
        p_above = 1.0 - norm.cdf(b1, loc=mean, scale=std_dev)
        
    return {
        f'< {b0}': float(p_below),
        f'{b0} - {b1}': float(p_between),
        f'> {b1}': float(p_above)
    }

