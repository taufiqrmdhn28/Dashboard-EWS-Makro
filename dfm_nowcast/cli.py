import os
import argparse
import sys
from .orchestrator import run_nowcasting_pipeline
from .utils import get_run_tag

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        bench_parser = argparse.ArgumentParser(description="Run side-by-side performance competition benchmark.")
        bench_parser.add_argument("data_file", nargs="?", default="data/INO_130726.xlsx", help="Path to the Excel dataset")
        bench_parser.add_argument("--start-est", default="2010-01", help="Start date for sample estimation in YYYY-MM format (e.g. 2010-01)")
        bench_parser.add_argument("--start-eval", default="2016-01-01", help="Starting date for historical evaluation (e.g. 2016-01-01)")
        bench_parser.add_argument("--last-eval", default="2026-03-31", help="Last date for historical evaluation (e.g. 2026-03-31)")
        bench_parser.add_argument("--output-dir", default="output/competition", help="Directory to save output files and plots")
        
        bench_args = bench_parser.parse_args(sys.argv[2:])
        
        start_est_list = None
        if bench_args.start_est:
            parts = bench_args.start_est.split("-")
            if len(parts) != 2:
                bench_parser.error("--start-est must be in YYYY-MM format (e.g. 2010-01)")
            try:
                start_est_list = [int(parts[0]), int(parts[1])]
            except ValueError:
                bench_parser.error("--start-est must contain valid integers (e.g. 2010-01)")
                
        from .benchmark import run_competition_benchmark
        try:
            run_competition_benchmark(
                data_file=bench_args.data_file,
                start_est=start_est_list,
                start_eval=bench_args.start_eval,
                last_eval=bench_args.last_eval,
                output_dir=bench_args.output_dir
            )
        except Exception as e:
            print(f"Error running competition benchmark: {e}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Dynamic Factor Model (DFM) Nowcasting Pipeline CLI."
    )
    parser.add_argument(
        "data_file",
        nargs="?",
        default=None,
        help="Path to the Excel dataset (e.g. data/INO_07112025.xlsx). Optional if --eval-only-dir is specified."
    )
    parser.add_argument(
        "--hist-eval",
        type=int,
        choices=[0, 1],
        default=None,
        help="Set to 1 for historical evaluation mode, or 0 for latest vintage monitoring."
    )
    parser.add_argument(
        "--start-est",
        default=None,
        help="Start date for sample estimation in YYYY-MM format (e.g. 2002-07)."
    )
    parser.add_argument(
        "--start-eval",
        default=None,
        help="Starting date for historical evaluation (e.g. '1-Jan-2025')."
    )
    parser.add_argument(
        "--last-eval",
        default=None,
        help="Last date for historical evaluation (e.g. '31-Dec-2025')."
    )
    parser.add_argument(
        "--ser-news",
        default=None,
        help="Name of the target variable to nowcast."
    )
    parser.add_argument(
        "--freq-estimation",
        default=None,
        choices=["quarterly", "annual"],
        help="Estimation frequency."
    )
    parser.add_argument(
        "--growth-rate",
        default=None,
        choices=["yoy", "qoq"],
        help="Growth rate type."
    )
    parser.add_argument(
        "--idiosyncratic",
        default=None,
        choices=["Autoregressive", "WhiteNoise"],
        help="Idiosyncratic component type."
    )
    parser.add_argument(
        "--p-ar",
        type=int,
        default=None,
        help="Number of lags for the autoregressive benchmark model."
    )
    parser.add_argument(
        "--native",
        action="store_true",
        default=None,
        help="Use modern native numpy OLS solver instead of legacy matrix inversion."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save final output CSV files."
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Directory to save intermediate step debug files (optional)."
    )
    parser.add_argument(
        "--tag-run",
        action="store_true",
        help="Append active git branch and timestamp sub-folder to the output/debug directory."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold value to calculate breach probability (e.g. 5.0)."
    )
    parser.add_argument(
        "--threshold-direction",
        default=None,
        choices=["lower", "higher"],
        help="Direction of threshold breach: 'lower' (prob < threshold) or 'higher' (prob > threshold)."
    )
    parser.add_argument(
        "--prob-distribution",
        default=None,
        choices=["gaussian", "student-t", "skew-normal", "johnsonsu", "empirical", "auto"],
        help="Distribution to use for probability calculations. 'auto' selects the best fit based on AIC."
    )
    parser.add_argument(
        "--prob-bins",
        default=None,
        help="Comma-separated boundary thresholds for 3-bin probability intervals (e.g. '5.0,5.35')."
    )
    parser.add_argument(
        "--no-pptx",
        action="store_true",
        help="Disable automatic generation of native PowerPoint presentation (.pptx) reports."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to external config.json file containing model hyperparameters."
    )
    parser.add_argument(
        "--eval-metrics",
        action="store_true",
        help="Compute and print historical forecasting validation metrics (RMSE, MAE, MAPE, CRPS, Log Score)."
    )
    parser.add_argument(
        "--plot-calibration",
        default=None,
        help="Path to save the probability calibration plot (e.g. calibration.png)."
    )
    parser.add_argument(
        "--plot-pit",
        default=None,
        help="Path to save the PIT calibration histogram plot (e.g. pit_histogram.png)."
    )
    parser.add_argument(
        "--eval-only-dir",
        default=None,
        help="Skip running the pipeline and only compute validation metrics on the specified directory of exported CSV files."
    )
 
    args = parser.parse_args()

    if not args.data_file and not args.eval_only_dir:
        parser.error("Must specify either data_file (to run pipeline) or --eval-only-dir (to evaluate existing run).")
 
    # Parse --start-est into [YYYY, MM]
    start_est_list = None
    if args.start_est:
        parts = args.start_est.split("-")
        if len(parts) != 2:
            parser.error("--start-est must be in YYYY-MM format (e.g. 2002-07)")
        try:
            start_est_list = [int(parts[0]), int(parts[1])]
        except ValueError:
            parser.error("--start-est must contain valid integers (e.g. 2002-07)")
 
    from .config import load_config
    config = load_config(args.config)
 
    output_dir = args.output_dir if args.output_dir is not None else config.get("output_dir", "temp/python")
    debug_dir = args.debug_dir if args.debug_dir is not None else config.get("debug_dir", None)
 
    if args.tag_run:
        tag = get_run_tag()
        if output_dir:
            output_dir = os.path.join(output_dir, tag)
        if debug_dir:
            debug_dir = os.path.join(debug_dir, tag)
 
    # Parse --prob-bins string if provided
    prob_bins_tuple = None
    if args.prob_bins:
        try:
            parts = [float(x.strip()) for x in args.prob_bins.split(",")]
            if len(parts) != 2:
                parser.error("--prob-bins must be two comma-separated floats (e.g. '5.0,5.35')")
            prob_bins_tuple = (parts[0], parts[1])
        except ValueError:
            parser.error("--prob-bins must contain valid numbers (e.g. '5.0,5.35')")

    # If run_pipeline is True (i.e. data_file is specified)
    if args.data_file:
        try:
            run_nowcasting_pipeline(
                data_file=args.data_file,
                hist_eval=args.hist_eval,
                start_est=start_est_list,
                start_eval=args.start_eval,
                last_eval=args.last_eval,
                ser_news=args.ser_news,
                freq_estimation=args.freq_estimation,
                growth_rate=args.growth_rate,
                idiosyncratic=args.idiosyncratic,
                p_ar=args.p_ar,
                native=args.native,
                output_dir=output_dir,
                debug_dir=debug_dir,
                threshold=args.threshold,
                threshold_direction=args.threshold_direction,
                prob_distribution=args.prob_distribution,
                config_path=args.config,
                prob_bins=prob_bins_tuple,
                generate_pptx=not args.no_pptx
            )
        except Exception as e:
            print(f"Error running pipeline: {e}", file=sys.stderr)
            sys.exit(1)

    # Validation and Evaluation logic
    if args.eval_metrics or args.plot_calibration or args.eval_only_dir:
        import numpy as np
        eval_dir = args.eval_only_dir if args.eval_only_dir else output_dir
        threshold = args.threshold if args.threshold is not None else config.get("threshold", None)
        threshold_direction = args.threshold_direction if args.threshold_direction is not None else config.get("threshold_direction", "lower")
        
        from .validation import evaluate_run_outputs, generate_calibration_plot
        
        try:
            metrics = evaluate_run_outputs(
                eval_dir,
                threshold=threshold,
                threshold_direction=threshold_direction
            )
            
            if args.eval_metrics or args.eval_only_dir:
                # Print nicely
                print_metrics_table(metrics, threshold, threshold_direction)
                
            if args.plot_calibration:
                # We need Prob_Nowcast.csv and Actual.csv
                prob_path = os.path.join(eval_dir, 'Prob_Nowcast.csv')
                actual_path = os.path.join(eval_dir, 'Actual.csv')
                
                if os.path.exists(prob_path) and os.path.exists(actual_path):
                    probs = np.loadtxt(prob_path, delimiter=",")
                    actuals = np.loadtxt(actual_path, delimiter=",")
                    if probs.ndim > 1:
                        probs = probs.flatten()
                    act_now = actuals[:, 1] # Nowcast actual
                    
                    if threshold_direction == 'lower':
                        act_binary = np.array(act_now < threshold, dtype=float)
                    else:
                        act_binary = np.array(act_now > threshold, dtype=float)
                        
                    act_binary[np.isnan(act_now)] = np.nan
                    
                    success = generate_calibration_plot(
                        probs,
                        act_binary,
                        args.plot_calibration
                    )
                    if success:
                        print(f"Calibration plot successfully saved to: {args.plot_calibration}")
                else:
                    print("Error: Could not generate calibration plot. Ensure the run was executed with a threshold so that Prob_Nowcast.csv is available.")
                    
            if args.plot_pit:
                pred_path = os.path.join(eval_dir, 'Nowcast.csv')
                std_path = os.path.join(eval_dir, 'Std_Nowcast.csv')
                actual_path = os.path.join(eval_dir, 'Actual.csv')
                
                if os.path.exists(pred_path) and os.path.exists(std_path) and os.path.exists(actual_path):
                    from .validation import calculate_pit, generate_pit_histogram
                    preds = np.loadtxt(pred_path, delimiter=",")
                    stds = np.loadtxt(std_path, delimiter=",")
                    actuals = np.loadtxt(actual_path, delimiter=",")
                    if preds.ndim > 1:
                        preds = preds.flatten()
                    if stds.ndim > 1:
                        stds = stds.flatten()
                    act_now = actuals[:, 1] # Nowcast actual
                    
                    pit_vals = calculate_pit(act_now, preds, stds)
                    success = generate_pit_histogram(pit_vals, args.plot_pit)
                    if success:
                        print(f"PIT histogram successfully saved to: {args.plot_pit}")
                else:
                    print("Error: Could not generate PIT histogram. Ensure the run was executed with forecast standard deviations so that Std_Nowcast.csv is available.")
                    
        except Exception as e:
            print(f"Error running evaluation: {e}", file=sys.stderr)
            sys.exit(1)

def print_metrics_table(metrics_summary, threshold=None, threshold_direction='lower'):
    import numpy as np
    print("\n" + "="*80)
    print("                      HISTORICAL FORECAST EVALUATION")
    print("="*80)
    headers = ["Horizon", "RMSE", "MAE", "MAPE", "CRPS", "Log Score"]
    print(f"{headers[0]:<12} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10} {headers[5]:<10}")
    print("-"*80)
    for horizon, m in metrics_summary.items():
        rmse = f"{m.get('RMSE', np.nan):.4f}" if not np.isnan(m.get('RMSE', np.nan)) else "NaN"
        mae = f"{m.get('MAE', np.nan):.4f}" if not np.isnan(m.get('MAE', np.nan)) else "NaN"
        mape = f"{m.get('MAPE', np.nan):.2f}%" if not np.isnan(m.get('MAPE', np.nan)) else "NaN"
        crps = f"{m.get('CRPS', np.nan):.4f}" if not np.isnan(m.get('CRPS', np.nan)) else "NaN"
        log_score = f"{m.get('LogScore', np.nan):.4f}" if not np.isnan(m.get('LogScore', np.nan)) else "NaN"
        print(f"{horizon:<12} {rmse:<10} {mae:<10} {mape:<10} {crps:<10} {log_score:<10}")
    print("="*80)
    
    # If probability metrics exist, print them
    has_prob = any('BrierScore' in m for m in metrics_summary.values())
    if has_prob:
        print("\n" + "="*80)
        print(f"                     PROBABILITY CALIBRATION METRICS")
        print(f"                  (Threshold: {threshold}, Direction: {threshold_direction})")
        print("="*80)
        p_headers = ["Horizon", "Brier Score", "Log Loss", "ECE"]
        print(f"{p_headers[0]:<12} {p_headers[1]:<15} {p_headers[2]:<15} {p_headers[3]:<10}")
        print("-"*80)
        for horizon, m in metrics_summary.items():
            brier = f"{m.get('BrierScore', np.nan):.4f}" if not np.isnan(m.get('BrierScore', np.nan)) else "NaN"
            log_loss = f"{m.get('LogLoss', np.nan):.4f}" if not np.isnan(m.get('LogLoss', np.nan)) else "NaN"
            ece = f"{m.get('ECE', np.nan):.4f}" if not np.isnan(m.get('ECE', np.nan)) else "NaN"
            print(f"{horizon:<12} {brier:<15} {log_loss:<15} {ece:<10}")
        print("="*80)

if __name__ == "__main__":
    main()
