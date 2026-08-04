import os
import unittest
import tempfile
import json
import shutil
import numpy as np
import pandas as pd

from dfm_nowcast.data import read_data, ml_build_calendar

class TestDataInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.excel_path = "data/INO_130726.xlsx"
        if not os.path.exists(cls.excel_path):
            raise FileNotFoundError(f"Test requires {cls.excel_path} to run.")
            
        cls.start_est = [2010, 1]
        cls.start_eval = "2024-01-01"
        cls.last_eval = "2024-03-31"
        cls.n_q = 1
        
        # 1. Run baseline using Excel file
        cls.ex_data, cls.ex_series, cls.ex_release_day, cls.ex_series_q, cls.ex_dates, cls.ex_nq = read_data(
            cls.excel_path, cls.start_est
        )
        cls.ex_vintage, cls.ex_pmv, cls.ex_mm, cls.ex_r = ml_build_calendar(
            cls.excel_path, cls.start_eval, cls.last_eval, cls.n_q
        )
        
        # Load sheets as DataFrames for building other formats
        xls = pd.ExcelFile(cls.excel_path)
        cls.dfs = {
            'InfoM': pd.read_excel(xls, 'InfoM'),
            'MonthlyData': pd.read_excel(xls, 'MonthlyData'),
            'InfoQ': pd.read_excel(xls, 'InfoQ'),
            'QuarterlyData': pd.read_excel(xls, 'QuarterlyData'),
            'Calendar': pd.read_excel(xls, 'Calendar', header=None)
        }
        
        # Create a temp directory for other test files
        cls.temp_dir = tempfile.mkdtemp()
        
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)
        
    def assert_read_data_parity(self, data, series, release_day, series_q, dates, n_q):
        # Numerical arrays comparison
        np.testing.assert_allclose(data, self.ex_data, equal_nan=True, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(dates, self.ex_dates, equal_nan=True, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(release_day, self.ex_release_day, equal_nan=True, rtol=1e-10, atol=1e-10)
        
        # Meta info comparison
        self.assertEqual(series, self.ex_series)
        self.assertEqual(series_q, self.ex_series_q)
        self.assertEqual(n_q, self.ex_nq)
        
    def assert_calendar_parity(self, vintage, pmv, mm, r):
        np.testing.assert_allclose(vintage, self.ex_vintage, equal_nan=True, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(pmv, self.ex_pmv, equal_nan=True, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(mm, self.ex_mm, equal_nan=True, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(r, self.ex_r, equal_nan=True, rtol=1e-10, atol=1e-10)

    def test_dict_of_dataframes(self):
        # Test direct DataFrame dictionary input
        data, series, release_day, series_q, dates, n_q = read_data(self.dfs, self.start_est)
        self.assert_read_data_parity(data, series, release_day, series_q, dates, n_q)
        
        vintage, pmv, mm, r = ml_build_calendar(self.dfs, self.start_eval, self.last_eval, self.n_q)
        self.assert_calendar_parity(vintage, pmv, mm, r)

    def test_csv_directory_iso_dates(self):
        # Create CSV folder representation with standard ISO date strings
        csv_dir = os.path.join(self.temp_dir, "csv_iso")
        os.makedirs(csv_dir, exist_ok=True)
        
        for name, df in self.dfs.items():
            # If it's a data table, convert its first date column to string
            df_write = df.copy()
            if name in ['MonthlyData', 'QuarterlyData']:
                # The first sheet column is date
                dt_col = df_write.iloc[:, 0]
                if pd.api.types.is_numeric_dtype(dt_col):
                    parsed_dt = pd.to_datetime(dt_col, unit='D', origin='1899-12-30')
                else:
                    parsed_dt = pd.to_datetime(dt_col)
                df_write.iloc[:, 0] = parsed_dt.dt.strftime('%Y-%m-%d')
            
            # Write to CSV
            header = False if name == 'Calendar' else True
            index = False
            df_write.to_csv(os.path.join(csv_dir, f"{name}.csv"), index=index, header=header)
            
        data, series, release_day, series_q, dates, n_q = read_data(csv_dir, self.start_est)
        self.assert_read_data_parity(data, series, release_day, series_q, dates, n_q)
        
        vintage, pmv, mm, r = ml_build_calendar(csv_dir, self.start_eval, self.last_eval, self.n_q)
        self.assert_calendar_parity(vintage, pmv, mm, r)

    def test_json_file_iso_dates(self):
        # Create JSON representation with ISO date strings
        json_path = os.path.join(self.temp_dir, "data_iso.json")
        
        json_data = {}
        for name, df in self.dfs.items():
            df_write = df.copy()
            if name in ['MonthlyData', 'QuarterlyData']:
                dt_col = df_write.iloc[:, 0]
                if pd.api.types.is_numeric_dtype(dt_col):
                    parsed_dt = pd.to_datetime(dt_col, unit='D', origin='1899-12-30')
                else:
                    parsed_dt = pd.to_datetime(dt_col)
                df_write.iloc[:, 0] = parsed_dt.dt.strftime('%Y-%m-%d')
            
            # Convert DataFrame to records (list of dicts) or list of lists for Calendar
            if name == 'Calendar':
                # Calendar needs to be list of lists since it has no headers
                json_data[name] = df_write.values.tolist()
            else:
                json_data[name] = df_write.to_dict(orient='records')
                
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f)
            
        data, series, release_day, series_q, dates, n_q = read_data(json_path, self.start_est)
        self.assert_read_data_parity(data, series, release_day, series_q, dates, n_q)
        
        vintage, pmv, mm, r = ml_build_calendar(json_path, self.start_eval, self.last_eval, self.n_q)
        self.assert_calendar_parity(vintage, pmv, mm, r)

    def test_json_file_excel_dates(self):
        # Create JSON representation with raw Excel datenum/serial values
        json_path = os.path.join(self.temp_dir, "data_excel.json")
        
        json_data = {}
        for name, df in self.dfs.items():
            if name == 'Calendar':
                # Save as lists of lists
                json_data[name] = df.values.tolist()
            else:
                json_data[name] = df.to_dict(orient='records')
                
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f)
            
        data, series, release_day, series_q, dates, n_q = read_data(json_path, self.start_est)
        self.assert_read_data_parity(data, series, release_day, series_q, dates, n_q)
        
        vintage, pmv, mm, r = ml_build_calendar(json_path, self.start_eval, self.last_eval, self.n_q)
        self.assert_calendar_parity(vintage, pmv, mm, r)

    def test_dynamic_defaults(self):
        # 1. Test read_data dynamic start_est (first row where all monthly columns are not NaN)
        data, series, release_day, series_q, dates, n_q = read_data(self.excel_path)
        # Verify first date is indeed 732494.0 (June 2005 as MATLAB datenum)
        self.assertEqual(dates[0], 732494.0)
        
        # 2. Test ml_build_calendar dynamic defaults
        # hist_eval = True: should default to 1-Jan-2025 and 31-Dec-2025, and n_q = 2
        v_def, pmv_def, mm_def, r_def = ml_build_calendar(self.excel_path, hist_eval=True)
        v_exp, pmv_exp, mm_exp, r_exp = ml_build_calendar(self.excel_path, "1-Jan-2025", "31-Dec-2025", 2)
        
        np.testing.assert_allclose(v_def, v_exp, equal_nan=True)
        np.testing.assert_allclose(pmv_def, pmv_exp, equal_nan=True)
        np.testing.assert_allclose(mm_def, mm_exp, equal_nan=True)
        np.testing.assert_allclose(r_def, r_exp, equal_nan=True)

    def test_date_conversions(self):
        from dfm_nowcast import convert_date
        
        # Test single date conversion
        dt = convert_date(735600.0, to_format='python')
        self.assertEqual(dt.year, 2014)
        self.assertEqual(dt.month, 1)
        self.assertEqual(dt.day, 1)
        
        dn = convert_date("2014-01-02", to_format='matlab')
        self.assertEqual(dn, 735601.0)
        
        ex = convert_date("2014-01-02", to_format='excel')
        self.assertEqual(ex, 41641.0)
        
        # Test array/list conversion
        list_dn = [735600.0, 735631.0]
        list_dt = convert_date(list_dn, to_format='python')
        self.assertEqual(list_dt[0].year, 2014)
        self.assertEqual(list_dt[1].month, 2)
        
        # Test read_data with Python and Excel date formats
        data, series, release_day, series_q, dates_py, n_q = read_data(
            self.excel_path, self.start_est, date_format='python'
        )
        self.assertTrue(isinstance(dates_py[0], pd.Timestamp))
        self.assertEqual(dates_py[0].year, 2010)
        self.assertEqual(dates_py[0].month, 1)
        
        data, series, release_day, series_q, dates_ex, n_q = read_data(
            self.excel_path, self.start_est, date_format='excel'
        )
        self.assertEqual(dates_ex[0], 40179.0)

    def test_native_ols(self):
        from dfm_nowcast import ml_ols
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = np.array([2.0, 4.0, 5.0, 8.0, 10.0])
        
        # Test original OLS
        beta_orig, _, _, _, _, _ = ml_ols(y, x, det=1, native=False)
        
        # Test native OLS
        beta_native, _, _, _, _, _ = ml_ols(y, x, det=1, native=True)
        
        # Verify output is identical within tolerance
        np.testing.assert_allclose(beta_orig, beta_native)

    def test_probability_estimation(self):
        from dfm_nowcast import estimate_probability_below
        
        # Create dummy inputs
        res_smooth = {
            'X_sm': np.array([[5.0]]),
            'P': np.zeros((1, 1, 2))
        }
        res_smooth['P'][0, 0, 1] = 1.0  # State variance = 1.0 at target_idx = 0 (t+1 = 1)
        
        R_new = {
            'C': np.array([[1.0]]),
            'R': np.array([[0.0]]),
            'Wx': np.array([[1.0]])
        }
        
        # Mean = 5.0, std dev = 1.0. For threshold = 5.0, P(X < 5) should be 0.5 (50%)
        prob, mean, std_dev = estimate_probability_below(res_smooth, R_new, i_ser_idx=0, target_idx=0, threshold=5.0)
        self.assertAlmostEqual(prob, 0.5)
        self.assertAlmostEqual(mean, 5.0)
        self.assertAlmostEqual(std_dev, 1.0)
        
        # For threshold = 5.0 + 1.95996... (1.96 standard deviations), P(X < 5 + 1.96) should be ~0.975 (97.5%)
        prob2, _, _ = estimate_probability_below(res_smooth, R_new, i_ser_idx=0, target_idx=0, threshold=5.0 + 1.95996)
        self.assertAlmostEqual(prob2, 0.975, places=3)

    def test_non_gaussian_probability_estimation(self):
        from dfm_nowcast.prediction import estimate_probability_below
        
        # Create dummy inputs
        res_smooth = {
            'X_sm': np.array([[5.0]]),
            'P': np.zeros((1, 1, 2))
        }
        res_smooth['P'][0, 0, 1] = 0.5  # State variance = 0.5 at target_idx = 0
        
        R_new = {
            'Z_0': np.zeros(1),
            'V_0': np.eye(1),
            'A': np.eye(1),
            'Q': np.eye(1),
            'C': np.array([[1.0]]),
            'R': np.array([[0.5]]), # Total variance in standardized units = c @ P @ c + R = 0.5 + 0.5 = 1.0
            'Wx': np.array([[1.0]]),
            'Mx': np.array([[0.0]])
        }
        
        # Create a mock X_new to compute innovations.
        np.random.seed(42)
        X_new = np.random.randn(20, 1)
        
        # Test student-t distribution
        prob_t, _, _ = estimate_probability_below(
            res_smooth, R_new, i_ser_idx=0, target_idx=0, threshold=5.0, X_new=X_new, dist='student-t'
        )
        self.assertTrue(0.0 <= prob_t <= 1.0)
        
        # Test skew-normal
        prob_sn, _, _ = estimate_probability_below(
            res_smooth, R_new, i_ser_idx=0, target_idx=0, threshold=5.0, X_new=X_new, dist='skew-normal'
        )
        self.assertTrue(0.0 <= prob_sn <= 1.0)

        # Test Johnsons SU
        prob_jsu, _, _ = estimate_probability_below(
            res_smooth, R_new, i_ser_idx=0, target_idx=0, threshold=5.0, X_new=X_new, dist='johnsonsu'
        )
        self.assertTrue(0.0 <= prob_jsu <= 1.0)
        
        # Test empirical
        prob_emp, _, _ = estimate_probability_below(
            res_smooth, R_new, i_ser_idx=0, target_idx=0, threshold=5.0, X_new=X_new, dist='empirical'
        )
        self.assertTrue(0.0 <= prob_emp <= 1.0)

        # Test auto selection
        prob_auto, _, _ = estimate_probability_below(
            res_smooth, R_new, i_ser_idx=0, target_idx=0, threshold=5.0, X_new=X_new, dist='auto'
        )
        self.assertTrue(0.0 <= prob_auto <= 1.0)

    def test_octave_python_parity(self):
        import subprocess
        import shutil
        
        # Check if octave is installed
        if not shutil.which("octave"):
            self.skipTest("GNU Octave is not installed/available in PATH.")
            
        test_dir = os.path.join(self.temp_dir, "parity_test")
        os.makedirs(test_dir, exist_ok=True)
        
        # Write a custom Nowcasting_test.m script
        octave_script = """
        clear all; close all; clc
        
        isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
        if isOctave
            pkg load io;
            if exist('setproperties', 'file') || exist('setproperties', 'builtin')
                setproperties("OCTINTERP", "parse_overrides", true);
            end
        end
        
        addpath ([pwd '/FUNCTIONS']);
        ML_graph_options

        HistEval = 0;
        StartEst = [2007 2];
        StartEval = '1-Jan-2026';
        LastEval = '31-Jul-2026';
        SerNews = 'RGDP_growth';
        DataFile = 'data/INO_130726.xlsx';

        NowcastingEstimation
        """
        
        script_path = os.path.join(test_dir, "Nowcasting_test.m")
        with open(script_path, "w") as f:
            f.write(octave_script)
            
        # Run Octave
        res_oct = subprocess.run(
            ["octave", "--no-gui", script_path],
            cwd=os.getcwd(),
            capture_output=True,
            text=True
        )
        if res_oct.returncode != 0:
            self.fail(f"Octave script failed with stderr:\\n{res_oct.stderr}\\nstdout:\\n{res_oct.stdout}")
            
        # Run Python
        from dfm_nowcast.orchestrator import run_nowcasting_pipeline
        py_output_dir = os.path.join(test_dir, "python_output")
        run_nowcasting_pipeline(
            data_file="data/INO_130726.xlsx",
            hist_eval=1,
            start_est=[2007, 2],
            start_eval="1-Jan-2026",
            last_eval="31-Jul-2026",
            ser_news="RGDP_growth",
            output_dir=py_output_dir,
            prob_distribution="gaussian"
        )
        
        # Compare output Nowcast, Forecast, and Backcast
        oct_nowcast_file = "temp/octave/NowcastingEstimation/Nowcast.csv"
        py_nowcast_file = os.path.join(py_output_dir, "Nowcast.csv")
        
        self.assertTrue(os.path.exists(oct_nowcast_file), "Octave output Nowcast.csv not found")
        self.assertTrue(os.path.exists(py_nowcast_file), "Python output Nowcast.csv not found")
        
        oct_nowcast = np.loadtxt(oct_nowcast_file, delimiter=",")
        py_nowcast = np.loadtxt(py_nowcast_file, delimiter=",")
        
        # Verify shapes are equal
        self.assertEqual(oct_nowcast.shape, py_nowcast.shape)
        
        # Verify numerical closeness
        np.testing.assert_allclose(py_nowcast, oct_nowcast, rtol=1e-2, atol=1e-2)

    def test_config_management(self):
        from dfm_nowcast.config import load_config
        
        # 1. Test DEFAULT_CONFIG
        config = load_config()
        self.assertEqual(config["p"], 2)
        self.assertEqual(config["thresh"], 1e-3)
        self.assertEqual(config["native"], False)
        
        # 2. Test JSON loading
        temp_config_path = os.path.join(self.temp_dir, "test_config.json")
        with open(temp_config_path, "w", encoding="utf-8") as f:
            json.dump({
                "p": 4,
                "r": [2, 1],
                "thresh": 0.005,
                "native": True,
                "start_est": "2005-06"
            }, f)
            
        config_loaded = load_config(temp_config_path)
        self.assertEqual(config_loaded["p"], 4)
        self.assertEqual(config_loaded["r"], [2, 1])
        self.assertEqual(config_loaded["thresh"], 0.005)
        self.assertEqual(config_loaded["native"], True)
        self.assertEqual(config_loaded["start_est"], [2005, 6])
        
        # 3. Test Env Var Override
        os.environ["DFM_P"] = "5"
        os.environ["DFM_THRESH"] = "0.01"
        os.environ["DFM_R"] = "3,2"
        os.environ["DFM_NATIVE"] = "True"
        
        config_override = load_config(temp_config_path)
        self.assertEqual(config_override["p"], 5)
        self.assertEqual(config_override["thresh"], 0.01)
        self.assertEqual(config_override["r"], [3, 2])
        self.assertEqual(config_override["native"], True)
        
        # Clean up env
        del os.environ["DFM_P"]
        del os.environ["DFM_THRESH"]
        del os.environ["DFM_R"]
        del os.environ["DFM_NATIVE"]

    def test_validation_metrics(self):
        from dfm_nowcast.validation import (
            calculate_rmse, calculate_mae, calculate_mape,
            calculate_gaussian_crps, calculate_gaussian_log_score,
            calculate_brier_score, calculate_log_loss, calculate_ece,
            generate_calibration_plot, calculate_pit, generate_pit_histogram
        )
        
        # Test basic math on simple inputs
        actual = np.array([2.0, 4.0, np.nan, 8.0])
        pred = np.array([1.5, 4.5, 6.0, 7.0])
        std = np.array([0.5, 1.0, 2.0, 1.5])
        
        self.assertAlmostEqual(calculate_rmse(actual, pred), np.sqrt((0.5**2 + 0.5**2 + 1.0**2)/3.0))
        self.assertAlmostEqual(calculate_mae(actual, pred), (0.5 + 0.5 + 1.0)/3.0)
        self.assertAlmostEqual(calculate_mape(actual, pred), (0.5/2.0 + 0.5/4.0 + 1.0/8.0)/3.0 * 100)
        
        # Test CRPS & Log Score
        crps = calculate_gaussian_crps(actual, pred, std)
        log_score = calculate_gaussian_log_score(actual, pred, std)
        self.assertTrue(crps > 0)
        self.assertTrue(log_score < 0)
        
        # Test Probability metrics
        prob = np.array([0.1, 0.9, np.nan, 0.4])
        act_binary = np.array([0.0, 1.0, 1.0, 0.0])
        
        self.assertAlmostEqual(calculate_brier_score(prob, act_binary), (0.1**2 + 0.1**2 + 0.4**2)/3.0)
        self.assertTrue(calculate_log_loss(prob, act_binary) > 0)
        self.assertTrue(calculate_ece(prob, act_binary) >= 0)
        
        # Test PIT
        pit_vals = calculate_pit(actual, pred, std)
        self.assertEqual(len(pit_vals), 3)
        self.assertTrue(all(p >= 0 and p <= 1 for p in pit_vals))
        
        # Test PIT plotting
        pit_plot_path = os.path.join(self.temp_dir, "test_pit.png")
        pit_success = generate_pit_histogram(pit_vals, pit_plot_path)
        try:
            import matplotlib
            self.assertTrue(pit_success)
            self.assertTrue(os.path.exists(pit_plot_path))
        except ImportError:
            self.assertFalse(pit_success)

    def test_generalized_and_sequential_probabilities(self):
        from dfm_nowcast.prediction import (
            estimate_breach_probability,
            estimate_sequential_breach_probability
        )
        
        # Mock DFM parameters
        R_new = {
            'C': np.array([[0.5, 0.2]]),
            'R': np.array([[0.01]]),
            'Wx': np.array([[1.5]]),
            'A': np.array([[0.8, 0.0], [0.0, 0.5]])
        }
        
        # Mock Kalman smoother outputs
        T = 20
        res_smooth = {
            'X_sm': np.zeros((T, 1)),
            'P': np.zeros((2, 2, T + 1))
        }
        # Populate smooth state means
        res_smooth['X_sm'][:, 0] = -1.0
        # Set state covariance to identity
        for t in range(T + 1):
            res_smooth['P'][:, :, t] = np.eye(2)
            
        # Test estimate_breach_probability with direction='lower'
        prob_lower, mean, std = estimate_breach_probability(
            res_smooth, R_new, i_ser_idx=0, target_idx=10,
            threshold=0.0, dist='gaussian', direction='lower'
        )
        self.assertAlmostEqual(mean, -1.0)
        self.assertAlmostEqual(std, 1.5 * np.sqrt(0.3))
        self.assertTrue(prob_lower > 0.5)
        
        # Test direction='higher'
        prob_higher, _, _ = estimate_breach_probability(
            res_smooth, R_new, i_ser_idx=0, target_idx=10,
            threshold=0.0, dist='gaussian', direction='higher'
        )
        self.assertAlmostEqual(prob_lower + prob_higher, 1.0)
        
        # Test estimate_sequential_breach_probability
        dates = np.arange(730000, 730000 + T * 30).astype(float)
        prob_seq, seq_means, cov_matrix = estimate_sequential_breach_probability(
            res_smooth, R_new, i_ser_idx=0, current_idx=5, dates=dates,
            threshold=0.0, direction='lower', n_consecutive=2,
            horizon=3, k_offset=0, n_simulations=1000
        )
        self.assertEqual(len(seq_means), 3)
        self.assertEqual(cov_matrix.shape, (3, 3))
        self.assertTrue(0.0 <= prob_seq <= 1.0)


if __name__ == '__main__':
    unittest.main()



