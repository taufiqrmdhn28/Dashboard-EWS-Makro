__version__ = "1.0.0"

from .utils import (
    compare_csv_directories,
    matlab_datevec,
    to_excel_serial,
    convert_date,
    get_run_tag
)

from .data import (
    read_data,
    build_pseudo_real_time_vintages,
    build_calendar,
    interpolate_missing_spline,
    check_collinearity,
    update_dataset_from_zaki,
    # Legacy aliases
    ml_pseudo_real_time_vintages,
    ml_build_calendar,
    remnans_spline
)

from .kalman import (
    run_kf,
    run_kalman_smoother,
    reset_kf_run_flag,
    # Legacy alias
    para_constdg
)

from .estimation import (
    fit_dfm_em,
    em_step,
    em_converged,
    init_cond,
    mixed_frequency_restrictions,
    idiosyncratic_law_of_motion,
    evaluate_em_convergence,
    # Legacy aliases
    em_dfm_ss_block_idioqarma_restrmq,
    ml_mixed_frequency_restrictions,
    ml_law_motion_idiosyncratic,
    ml_estimation_yn
)

from .prediction import (
    update_predictions,
    compute_forecast_news,
    compute_annual_nowcast,
    nowcast_current_output,
    get_current_quarter,
    estimate_probability_below,
    estimate_annual_probability_below,
    estimate_breach_probability,
    estimate_annual_breach_probability,
    estimate_sequential_breach_probability,
    compute_probability_bins,
    # Legacy aliases
    ml_update_prediction,
    news_dfm_mldgn3,
    ml_annual_nowcast,
    ml_nowcast_current_output,
    ml_current_quarter
)

from .benchmark import (
    ar_realtime_gdp,
    fit_autoregressive,
    fit_ols,
    update_benchmark_predictions,
    compute_forecast_errors,
    # Legacy aliases
    ml_ar_realtime_gdp,
    ml_autoregressive,
    ml_ols,
    ml_update_benchmark_predictions,
    ml_nowcast_forecast_error
)

from .orchestrator import (
    run_nowcasting_pipeline
)

from .config import (
    load_config
)

from .validation import (
    evaluate_run_outputs,
    generate_calibration_plot,
    compute_residual_normality
)

from .pptx_exporter import (
    generate_probability_pptx,
    build_3slide_deck
)
