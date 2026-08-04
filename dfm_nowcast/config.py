import os
import json

DEFAULT_CONFIG = {
    "p": 2,
    "r": 2,
    "dyn": 2,
    "max_iter": 300,
    "thresh": 1e-3,
    "hist_eval": 1,
    "start_est": None,
    "start_eval": None,
    "last_eval": None,
    "ser_news": "RGDP_growth",
    "freq_estimation": "quarterly",
    "growth_rate": "yoy",
    "idiosyncratic": "Autoregressive",
    "p_ar": 2,
    "native": False,
    "output_dir": "temp/python",
    "debug_dir": None,
    "threshold": None,
    "threshold_direction": "lower",
    "prob_distribution": "gaussian",
    "seq_threshold": 0.0,
    "seq_direction": "lower",
    "seq_n_consecutive": 2,
    "seq_horizon": 4,
    "seq_k_offset": 0,
    "estimator": "EM",
    "robust_scaling": False,
    "winsorization": False,
    "winsorization_k": 4.0,
    "covariance_regularization": False,
    "ridge_lambda": 1e-4,
    "mcmc_draws": 100,
    "mcmc_burnin": 50
}

def parse_list_int(val):
    if isinstance(val, str):
        if "," in val:
            return [int(x.strip()) for x in val.split(",")]
        try:
            return int(val)
        except ValueError:
            return val
    return val

def parse_start_est(val):
    if isinstance(val, str):
        parts = val.split("-")
        if len(parts) == 2:
            try:
                return [int(parts[0]), int(parts[1])]
            except ValueError:
                pass
    elif isinstance(val, list):
        try:
            return [int(x) for x in val]
        except ValueError:
            pass
    return val

def load_config(config_path=None):
    """
    Loads configuration from config_path (or default config.json if present),
    applies environment variable overrides, and returns a config dict.
    """
    config = DEFAULT_CONFIG.copy()
    
    # 1. Load from file if config_path is provided or default config.json exists
    actual_path = config_path
    if not actual_path:
        if os.path.exists("config.json"):
            actual_path = "config.json"
            
    if actual_path and os.path.exists(actual_path):
        try:
            with open(actual_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
                for k, v in file_config.items():
                    if k in config:
                        config[k] = v
        except Exception as e:
            print(f"Warning: Failed to load config from {actual_path}: {e}")
            
    # Normalize start_est if loaded from config file
    if config["start_est"] is not None:
        config["start_est"] = parse_start_est(config["start_est"])
    if config["r"] is not None:
        config["r"] = parse_list_int(config["r"])
    if config["dyn"] is not None:
        config["dyn"] = parse_list_int(config["dyn"])

    # 2. Environment variable overrides (prefixed with DFM_)
    env_mapping = {
        "DFM_P": ("p", int),
        "DFM_R": ("r", parse_list_int),
        "DFM_DYN": ("dyn", parse_list_int),
        "DFM_MAX_ITER": ("max_iter", int),
        "DFM_THRESH": ("thresh", float),
        "DFM_HIST_EVAL": ("hist_eval", int),
        "DFM_START_EST": ("start_est", parse_start_est),
        "DFM_START_EVAL": ("start_eval", str),
        "DFM_LAST_EVAL": ("last_eval", str),
        "DFM_SER_NEWS": ("ser_news", str),
        "DFM_FREQ_EVALUATION": ("freq_estimation", str),
        "DFM_GROWTH_RATE": ("growth_rate", str),
        "DFM_IDIOSYNCRATIC": ("idiosyncratic", str),
        "DFM_P_AR": ("p_ar", int),
        "DFM_NATIVE": ("native", lambda x: x.lower() in ("true", "1", "yes")),
        "DFM_OUTPUT_DIR": ("output_dir", str),
        "DFM_DEBUG_DIR": ("debug_dir", str),
        "DFM_THRESHOLD": ("threshold", float),
        "DFM_THRESHOLD_DIRECTION": ("threshold_direction", str),
        "DFM_PROB_DISTRIBUTION": ("prob_distribution", str),
        "DFM_SEQ_THRESHOLD": ("seq_threshold", float),
        "DFM_SEQ_DIRECTION": ("seq_direction", str),
        "DFM_SEQ_N_CONSECUTIVE": ("seq_n_consecutive", int),
        "DFM_SEQ_HORIZON": ("seq_horizon", int),
        "DFM_SEQ_K_OFFSET": ("seq_k_offset", int),
        "DFM_ESTIMATOR": ("estimator", str),
        "DFM_ROBUST_SCALING": ("robust_scaling", lambda x: x.lower() in ("true", "1", "yes")),
        "DFM_WINSORIZATION": ("winsorization", lambda x: x.lower() in ("true", "1", "yes")),
        "DFM_WINSORIZATION_K": ("winsorization_k", float),
        "DFM_COVARIANCE_REGULARIZATION": ("covariance_regularization", lambda x: x.lower() in ("true", "1", "yes")),
        "DFM_RIDGE_LAMBDA": ("ridge_lambda", float),
        "DFM_MCMC_DRAWS": ("mcmc_draws", int),
        "DFM_MCMC_BURNIN": ("mcmc_burnin", int)
    }
    
    for env_var, (key, type_cast) in env_mapping.items():
        val = os.environ.get(env_var)
        if val is not None:
            try:
                config[key] = type_cast(val)
            except Exception as e:
                print(f"Warning: Failed to cast env var {env_var}='{val}': {e}")
                
    return config
