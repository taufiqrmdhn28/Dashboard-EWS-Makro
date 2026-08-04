import os
import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_rmse(actual, pred):
    mask = ~np.isnan(actual) & ~np.isnan(pred)
    a = actual[mask]
    p = pred[mask]
    if len(a) == 0:
        return np.nan
    return float(np.sqrt(np.mean((a - p)**2)))

def calculate_mae(actual, pred):
    mask = ~np.isnan(actual) & ~np.isnan(pred)
    a = actual[mask]
    p = pred[mask]
    if len(a) == 0:
        return np.nan
    return float(np.mean(np.abs(a - p)))

def calculate_mape(actual, pred):
    mask = ~np.isnan(actual) & ~np.isnan(pred) & (actual != 0)
    a = actual[mask]
    p = pred[mask]
    if len(a) == 0:
        return np.nan
    return float(np.mean(np.abs((a - p) / a)) * 100)

def calculate_gaussian_crps(actual, mean, std):
    mask = ~np.isnan(actual) & ~np.isnan(mean) & ~np.isnan(std) & (std > 0)
    a = actual[mask]
    m = mean[mask]
    s = std[mask]
    if len(a) == 0:
        return np.nan
        
    z = (a - m) / s
    crps = s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))

def calculate_gaussian_log_score(actual, mean, std):
    mask = ~np.isnan(actual) & ~np.isnan(mean) & ~np.isnan(std) & (std > 0)
    a = actual[mask]
    m = mean[mask]
    s = std[mask]
    if len(a) == 0:
        return np.nan
        
    log_score = -0.5 * np.log(2 * np.pi * s**2) - ((a - m)**2) / (2 * s**2)
    return float(np.mean(log_score))

def calculate_brier_score(prob, actual_binary):
    mask = ~np.isnan(prob) & ~np.isnan(actual_binary)
    p = prob[mask]
    a = actual_binary[mask]
    if len(p) == 0:
        return np.nan
    return float(np.mean((p - a)**2))

def calculate_log_loss(prob, actual_binary, eps=1e-15):
    mask = ~np.isnan(prob) & ~np.isnan(actual_binary)
    p = np.clip(prob[mask], eps, 1 - eps)
    a = actual_binary[mask]
    if len(p) == 0:
        return np.nan
    return float(-np.mean(a * np.log(p) + (1 - a) * np.log(1 - p)))

def calculate_ece(prob, actual_binary, n_bins=5):
    mask = ~np.isnan(prob) & ~np.isnan(actual_binary)
    p = prob[mask]
    a = actual_binary[mask]
    if len(p) == 0:
        return np.nan
        
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i+1]
        
        # Find indices in bin range
        in_bin = (p >= bin_lower) & (p < bin_upper) if i < n_bins - 1 else (p >= bin_lower) & (p <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(a[in_bin])
            confidence_in_bin = np.mean(p[in_bin])
            ece += prop_in_bin * np.abs(accuracy_in_bin - confidence_in_bin)
            
    return float(ece)

def generate_calibration_plot(prob, actual_binary, save_path, n_bins=5):
    """
    Generates a reliability diagram and saves it.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is not installed. Skipping calibration plot.")
        return False
        
    mask = ~np.isnan(prob) & ~np.isnan(actual_binary)
    p = prob[mask]
    a = actual_binary[mask]
    if len(p) == 0:
        print("Warning: No valid data to plot calibration.")
        return False
        
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    accuracies = []
    
    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i+1]
        in_bin = (p >= bin_lower) & (p < bin_upper) if i < n_bins - 1 else (p >= bin_lower) & (p <= bin_upper)
        
        if np.sum(in_bin) > 0:
            bin_centers.append(np.mean(p[in_bin]))
            accuracies.append(np.mean(a[in_bin]))
            
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.plot(bin_centers, accuracies, marker='o', color='blue', label='Model Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Empirical Fraction of Positives')
    plt.title('Probability Calibration Curve (Reliability Diagram)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return True

def evaluate_run_outputs(output_dir, threshold=None, threshold_direction='lower'):
    """
    Reads exported CSV files in output_dir, calculates metrics for each horizon,
    and returns a summary dictionary of metrics.
    """
    horizons = ['Backcast', 'Nowcast', 'Forecast', 'Forecast2S', 'Forecast3S']
    
    # Load actuals and reference dates
    actual_path = os.path.join(output_dir, 'Actual.csv')
    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"Actual.csv not found in {output_dir}")
        
    actuals = np.loadtxt(actual_path, delimiter=",")
    if actuals.ndim == 1:
        actuals = actuals.reshape(-1, 1)
        
    metrics_summary = {}
    
    for idx, horizon in enumerate(horizons):
        pred_path = os.path.join(output_dir, f"{horizon}.csv")
        std_path = os.path.join(output_dir, f"Std_{horizon}.csv")
        prob_path = os.path.join(output_dir, f"Prob_{horizon}.csv")
        
        if not os.path.exists(pred_path):
            continue
            
        preds = np.loadtxt(pred_path, delimiter=",")
        if preds.ndim > 1:
            preds = preds.flatten()
            
        act = actuals[:, idx]
        
        # 1. Point forecast metrics
        rmse = calculate_rmse(act, preds)
        mae = calculate_mae(act, preds)
        mape = calculate_mape(act, preds)
        
        horizon_metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
        # 2. Density forecast metrics
        if os.path.exists(std_path):
            stds = np.loadtxt(std_path, delimiter=",")
            if stds.ndim > 1:
                stds = stds.flatten()
            crps = calculate_gaussian_crps(act, preds, stds)
            log_score = calculate_gaussian_log_score(act, preds, stds)
            
            horizon_metrics['CRPS'] = crps
            horizon_metrics['LogScore'] = log_score
            
        # 3. Probability calibration metrics
        if threshold is not None and os.path.exists(prob_path):
            probs = np.loadtxt(prob_path, delimiter=",")
            if probs.ndim > 1:
                probs = probs.flatten()
                
            # Create binary outcome: 1 if actual is below (or above) threshold
            if threshold_direction == 'lower':
                act_binary = np.array(act < threshold, dtype=float)
            else:
                act_binary = np.array(act > threshold, dtype=float)
                
            # Filter NaNs from binary outcomes (if actual was NaN)
            act_binary[np.isnan(act)] = np.nan
            
            brier = calculate_brier_score(probs, act_binary)
            log_loss = calculate_log_loss(probs, act_binary)
            ece = calculate_ece(probs, act_binary)
            
            horizon_metrics['BrierScore'] = brier
            horizon_metrics['LogLoss'] = log_loss
            horizon_metrics['ECE'] = ece
            
        metrics_summary[horizon] = horizon_metrics
        
    return metrics_summary


def compute_residual_normality(actual, pred):
    """
    Computes skewness, kurtosis, and Jarque-Bera and Shapiro-Wilk test statistics
    for the prediction residuals (actual - pred).
    """
    from scipy.stats import jarque_bera, shapiro, skew, kurtosis
    
    mask = ~np.isnan(actual) & ~np.isnan(pred)
    a = actual[mask]
    p = pred[mask]
    if len(a) == 0:
        return {
            'count': 0, 'skewness': np.nan, 'kurtosis': np.nan,
            'jb_stat': np.nan, 'jb_pval': np.nan,
            'shapiro_stat': np.nan, 'shapiro_pval': np.nan
        }
        
    residuals = a - p
    
    jb_stat, jb_pval = jarque_bera(residuals)
    try:
        shapiro_stat, shapiro_pval = shapiro(residuals)
    except Exception:
        shapiro_stat, shapiro_pval = np.nan, np.nan
        
    skew_val = skew(residuals)
    kurt_val = kurtosis(residuals) # excess
    
    return {
        'count': len(residuals),
        'skewness': float(skew_val),
        'kurtosis': float(kurt_val + 3.0),
        'excess_kurtosis': float(kurt_val),
        'jb_stat': float(jb_stat),
        'jb_pval': float(jb_pval),
        'shapiro_stat': float(shapiro_stat),
        'shapiro_pval': float(shapiro_pval)
    }


def calculate_pit(actual, mean, std):
    """
    Computes Probability Integral Transform (PIT) values: PIT_t = Phi((y_t - mu_t) / std_t).
    If predictive distributions are correct, PIT values should be Uniform(0, 1).
    """
    mask = ~np.isnan(actual) & ~np.isnan(mean) & ~np.isnan(std) & (std > 0)
    a = actual[mask]
    m = mean[mask]
    s = std[mask]
    if len(a) == 0:
        return np.array([])
    z = (a - m) / s
    pit = norm.cdf(z)
    return pit


def generate_pit_histogram(pit_values, save_path, n_bins=10):
    """
    Generates a PIT histogram to diagnose probability calibration.
    For a well-calibrated model, the histogram should be flat (uniform).
    """
    if len(pit_values) == 0:
        print("Warning: No valid PIT values to plot.")
        return False
        
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is not installed. Skipping PIT plot.")
        return False
        
    plt.figure(figsize=(6, 5))
    # Plot uniform reference line
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Perfect Calibration')
    
    # Plot density histogram
    plt.hist(pit_values, bins=n_bins, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Model PIT')
    
    # Run Kolmogorov-Smirnov test for uniformity
    from scipy.stats import kstest
    ks_stat, ks_pval = kstest(pit_values, 'uniform')
    
    plt.xlabel('Probability Integral Transform (PIT) Value')
    plt.ylabel('Density')
    plt.title(f'PIT Histogram (KS p-val: {ks_pval:.4f})')
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return True

