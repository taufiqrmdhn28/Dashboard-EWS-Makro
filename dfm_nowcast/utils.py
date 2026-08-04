import os
import shutil
import subprocess
import datetime
import numpy as np
import pandas as pd

def init_debug_dir(dir_name, t=None, debug_dir=None):
    """
    Initializes a debug directory under debug_dir if provided.
    """
    if debug_dir is None:
        debug_dir = os.environ.get("DFM_NOWCAST_DEBUG_DIR")
        
    if debug_dir is None:
        return None
    
    target_dir = os.path.join(debug_dir, dir_name)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def csvwrite_debug(target_dir, filename, data):
    """
    Writes data as a CSV file to target_dir if target_dir is not None.
    Matches MATLAB's NaN casing.
    """
    if target_dir is None:
        return
    
    filepath = os.path.join(target_dir, filename)
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        with open(filepath, 'w') as f:
            for item in data:
                f.write(f"{item}\n")
        return
        
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.to_numpy()
        
    data = np.array(data, dtype=float)
    if data.ndim == 0:
        data = data.reshape(1, 1)
    elif data.ndim == 1:
        data = data.reshape(-1, 1)
        
    np.savetxt(filepath, data, delimiter=',', fmt='%.16g')
    
    with open(filepath, 'r') as f:
        content = f.read()
    with open(filepath, 'w') as f:
        f.write(content.replace('nan', 'NaN'))

def get_run_tag():
    """
    Detects the current active git branch and generates a timestamped string
    in the format '{branch}_{timestamp}' (e.g., 'master_20260629_013000').
    """
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL)
        branch_name = branch.decode("utf-8").strip()
    except Exception:
        branch_name = "unknown-branch"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{branch_name}_{timestamp}"

def to_matlab_datenum(date_series):
    """Converts a pandas Series/DatetimeIndex to MATLAB datenums, supporting string conversion."""
    if not pd.api.types.is_numeric_dtype(date_series) and not pd.api.types.is_datetime64_any_dtype(date_series):
        date_series = pd.to_datetime(date_series)
        
    if pd.api.types.is_datetime64_any_dtype(date_series):
        dt_index = pd.DatetimeIndex(date_series)
        return dt_index.map(lambda x: x.toordinal() + 366).values.astype(float)
    else:
        return np.array(date_series) + 693960

def to_excel_serial(val):
    """Converts explicit dates/numbers/strings to raw Excel serial floats."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (pd.Timestamp, datetime.datetime)):
        return (val.toordinal() + 366) - 693960 + (val.hour*3600 + val.minute*60 + val.second)/86400.0
    if isinstance(val, (int, float, np.number)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            pass
        try:
            dt = pd.to_datetime(val)
            if not pd.isna(dt):
                return (dt.toordinal() + 366) - 693960 + (dt.hour*3600 + dt.minute*60 + dt.second)/86400.0
        except (ValueError, TypeError):
            pass
    return np.nan

def _single_to_datetime(val, from_format=None):
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, (pd.Timestamp, datetime.datetime)):
        return pd.Timestamp(val)
    if isinstance(val, datetime.date):
        return pd.Timestamp(val)
    if isinstance(val, str):
        try:
            return pd.to_datetime(val)
        except (ValueError, TypeError):
            try:
                val = float(val)
            except ValueError:
                return pd.NaT
                
    if isinstance(val, (int, float, np.number)):
        val_f = float(val)
        if np.isnan(val_f):
            return pd.NaT
        if from_format == 'matlab':
            return pd.to_datetime(val_f - 693960, unit='D', origin='1899-12-30')
        elif from_format == 'excel':
            return pd.to_datetime(val_f, unit='D', origin='1899-12-30')
        else:
            if val_f > 100000:
                return pd.to_datetime(val_f - 693960, unit='D', origin='1899-12-30')
            else:
                return pd.to_datetime(val_f, unit='D', origin='1899-12-30')
                
    return pd.NaT

def convert_date(val, to_format='python', from_format=None):
    """
    Converts a date value or array/Series/Index of date values between:
    - 'python' (datetime.datetime / pd.Timestamp / pd.DatetimeIndex)
    - 'matlab' (MATLAB datenum float)
    - 'excel' (Excel serial float)
    
    Parameters:
        val: A single date, string, float, list, NumPy array, or pandas Series/Index.
        to_format (str): The target format, one of 'python', 'matlab', or 'excel'.
        from_format (str, optional): The source format. If None, it will be automatically inferred.
    """
    to_format = to_format.lower()
    if to_format not in ['python', 'matlab', 'excel']:
        raise ValueError("to_format must be one of: 'python', 'matlab', 'excel'")
        
    if from_format is not None:
        from_format = from_format.lower()
        if from_format not in ['python', 'matlab', 'excel']:
            raise ValueError("from_format must be one of: 'python', 'matlab', 'excel'")

    is_iterable = isinstance(val, (list, np.ndarray, pd.Series, pd.Index))
    
    if not is_iterable:
        dt = _single_to_datetime(val, from_format=from_format)
        if to_format == 'python':
            return dt
        elif to_format == 'matlab':
            if pd.isna(dt):
                return np.nan
            return float(dt.toordinal() + 366) + (dt.hour*3600 + dt.minute*60 + dt.second)/86400.0
        elif to_format == 'excel':
            if pd.isna(dt):
                return np.nan
            return float((dt.toordinal() + 366) - 693960) + (dt.hour*3600 + dt.minute*60 + dt.second)/86400.0
            
    if isinstance(val, pd.Series):
        dt_series = pd.to_datetime(val.map(lambda x: _single_to_datetime(x, from_format=from_format)))
        if to_format == 'python':
            return dt_series
        elif to_format == 'matlab':
            return dt_series.map(lambda x: np.nan if pd.isna(x) else float(x.toordinal() + 366) + (x.hour*3600 + x.minute*60 + x.second)/86400.0).values
        elif to_format == 'excel':
            return dt_series.map(lambda x: np.nan if pd.isna(x) else float((x.toordinal() + 366) - 693960) + (x.hour*3600 + x.minute*60 + x.second)/86400.0).values
            
    elif isinstance(val, pd.Index):
        dt_idx = pd.to_datetime([_single_to_datetime(x, from_format=from_format) for x in val])
        if to_format == 'python':
            return dt_idx
        elif to_format == 'matlab':
            return np.array([np.nan if pd.isna(x) else float(x.toordinal() + 366) + (x.hour*3600 + x.minute*60 + x.second)/86400.0 for x in dt_idx])
        elif to_format == 'excel':
            return np.array([np.nan if pd.isna(x) else float((x.toordinal() + 366) - 693960) + (x.hour*3600 + x.minute*60 + x.second)/86400.0 for x in dt_idx])
            
    else:
        arr = np.array(val)
        shape = arr.shape
        flat = arr.flatten()
        dt_list = [_single_to_datetime(x, from_format=from_format) for x in flat]
        
        if to_format == 'python':
            out = np.array(dt_list, dtype=object).reshape(shape)
        elif to_format == 'matlab':
            out = np.array([np.nan if pd.isna(x) else float(x.toordinal() + 366) + (x.hour*3600 + x.minute*60 + x.second)/86400.0 for x in dt_list]).reshape(shape)
        elif to_format == 'excel':
            out = np.array([np.nan if pd.isna(x) else float((x.toordinal() + 366) - 693960) + (x.hour*3600 + x.minute*60 + x.second)/86400.0 for x in dt_list]).reshape(shape)
            
        if isinstance(val, list):
            return out.tolist()
        return out

def get_matlab_xlsread_matrix_from_df(df):
    """Emulates MATLAB's `a` matrix bounding box extraction from a DataFrame."""
    if hasattr(df, 'map'):
        num_mat = df.map(to_excel_serial).values 
    else:
        num_mat = df.applymap(to_excel_serial).values 
        
    valid_mask = ~np.isnan(num_mat)
    valid_rows = np.where(np.any(valid_mask, axis=1))[0]
    valid_cols = np.where(np.any(valid_mask, axis=0))[0]
    
    if len(valid_rows) == 0 or len(valid_cols) == 0:
        return np.array([[]])
        
    row_start, row_end = valid_rows[0], valid_rows[-1]
    col_start, col_end = valid_cols[0], valid_cols[-1]
    
    return num_mat[row_start:row_end+1, col_start:col_end+1]

def get_matlab_xlsread_matrix(data_input, sheet_name):
    """Emulates MATLAB's `a` matrix bounding box extraction from Excel or DataFrames/dicts."""
    if isinstance(data_input, pd.DataFrame):
        df = data_input
    elif isinstance(data_input, dict):
        df = data_input.get(sheet_name)
        if df is None:
            raise ValueError(f"Sheet/Key '{sheet_name}' not found in data_input dict.")
    elif isinstance(data_input, str) and os.path.exists(data_input):
        if data_input.endswith('.xlsx'):
            df = pd.read_excel(data_input, sheet_name=sheet_name, header=None)
        elif data_input.endswith('.json'):
            import json
            with open(data_input, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            df = pd.DataFrame(data_dict[sheet_name])
        elif os.path.isdir(data_input):
            csv_path = os.path.join(data_input, f"{sheet_name}.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(data_input, f"{sheet_name.lower()}.csv")
            df = pd.read_csv(csv_path, header=None)
        else:
            raise ValueError(f"Unsupported file format: {data_input}")
    else:
        raise TypeError(f"data_input must be a path string or dict/DataFrame, got {type(data_input)}")
        
    return get_matlab_xlsread_matrix_from_df(df)

def matlab_datevec(dn_array):
    """Emulates MATLAB datevec() extracting [Y, M, D, H, MI, S] from datenums."""
    flat_dn = np.array(dn_array).flatten()
    out = np.full((len(flat_dn), 6), np.nan)
    valid = ~np.isnan(flat_dn)
    
    dt_index = pd.to_datetime(flat_dn[valid] - 693960, unit='D', origin='1899-12-30')
    
    out[valid, 0] = dt_index.year
    out[valid, 1] = dt_index.month
    out[valid, 2] = dt_index.day
    out[valid, 3] = dt_index.hour
    out[valid, 4] = dt_index.minute
    out[valid, 5] = dt_index.second
    
    return out

def compare_csv_directories(base_dir="temp", matlab_base=None, python_base=None):
    if matlab_base is None:
        matlab_base = os.path.join(base_dir, "matlab")
        if not os.path.exists(matlab_base):
            matlab_base = os.path.join(base_dir, "octave")
    if python_base is None:
        python_base = os.path.join(base_dir, "python")
        
    if not os.path.exists(matlab_base) or not os.path.exists(python_base):
        print(f"Error: Ensure both '{matlab_base}' and '{python_base}' exist.")
        return

    results = []
    
    # Walk through the Python directory to find all CSVs
    for root, _, files in sorted(os.walk(python_base)):
        for file in sorted(files):
            if file.endswith('.csv') and file not in ['comparison_report.log', 'comparison_report.csv']:
                rel_path = os.path.relpath(os.path.join(root, file), python_base)
                parts = rel_path.split(os.sep)
                component = parts[0] if len(parts) > 1 else "Root"
                filename = parts[-1]
                
                matlab_file = os.path.join(matlab_base, rel_path)
                python_file = os.path.join(python_base, rel_path)
                
                res = {
                    'rel_path': rel_path,
                    'component': component,
                    'filename': filename,
                    'status': 'MATCH',
                    'detail': '',
                    'log': []
                }
                
                res['log'].append(f"\nComparing: {rel_path}")
                
                if not os.path.exists(matlab_file):
                    res['status'] = 'MISSING'
                    res['detail'] = 'Missing in MATLAB dir'
                    res['log'].append(f"  ❌ Missing in MATLAB/Octave directory: {matlab_file}")
                    results.append(res)
                    continue
                
                try:
                    # Check empty files
                    if os.path.getsize(matlab_file) <= 2 and os.path.getsize(python_file) <= 2:
                        with open(matlab_file, 'r') as fm, open(python_file, 'r') as fp:
                            if not fm.read().strip() and not fp.read().strip():
                                res['status'] = 'MATCH'
                                res['detail'] = 'Both empty files'
                                res['log'].append("  ✅ MATCH (both empty files)")
                                results.append(res)
                                continue
                                
                    df_matlab = pd.read_csv(matlab_file, header=None)
                    df_python = pd.read_csv(python_file, header=None)
                    
                    if df_matlab.shape != df_python.shape:
                        res['status'] = 'SHAPE_MISMATCH'
                        res['detail'] = f"Shape Mismatch: M{df_matlab.shape} vs P{df_python.shape}"
                        res['log'].append(f"  ❌ Shape Mismatch! MATLAB: {df_matlab.shape}, Python: {df_python.shape}")
                        results.append(res)
                        continue
                    
                    try:
                        is_close = np.allclose(df_matlab.to_numpy(), df_python.to_numpy(), rtol=1e-5, atol=1e-6, equal_nan=True)
                        if is_close:
                            res['status'] = 'MATCH'
                            res['detail'] = 'Match (within tolerance)'
                            res['log'].append("  ✅ MATCH (within numerical tolerance)")
                        else:
                            max_diff = np.nanmax(np.abs(df_matlab.to_numpy() - df_python.to_numpy()))
                            res['status'] = 'MISMATCH'
                            res['detail'] = f"Max Diff: {max_diff:.6g}"
                            res['log'].append(f"  ❌ MISMATCH! Max absolute difference: {max_diff}")
                    except TypeError:
                        if df_matlab.equals(df_python):
                            res['status'] = 'MATCH'
                            res['detail'] = 'Match (text/exact)'
                            res['log'].append("  ✅ MATCH (text/exact)")
                        else:
                            res['status'] = 'MISMATCH'
                            res['detail'] = 'Text mismatch'
                            res['log'].append("  ❌ MISMATCH (text contents do not match)")
                            
                except Exception as e:
                    res['status'] = 'ERROR'
                    res['detail'] = f"Error: {str(e)}"
                    res['log'].append(f"  ⚠️ Error processing file: {e}")
                    
                results.append(res)
                
    errors = [r for r in results if r['status'] != 'MATCH']
    
    if errors:
        print("=" * 100)
        print("                                    MISMATCH SUMMARY                                    ")
        print("=" * 100)
        print(f"| {'Component / Step':<28} | {'File Name':<38} | {'Error Type / Detail':<25} |")
        print("-" * 100)
        for err in errors:
            comp = err['component'][:28]
            fname = err['filename'][:38]
            det = err['detail'][:25]
            print(f"| {comp:<28} | {fname:<38} | {det:<25} |")
        print("=" * 100)
        print("\n")
    else:
        print("=" * 100)
        print("                          NO MISMATCHES DETECTED (ALL MATCH)                            ")
        print("=" * 100)
        print("\n")
        
    # Write full detailed report to log file
    log_file_path = os.path.join(base_dir, "comparison_report.log")
    try:
        with open(log_file_path, "w", encoding="utf-8") as lf:
            lf.write("================================================================================\n")
            lf.write("                           FULL DETAILED COMPARISON REPORT                      \n")
            lf.write("================================================================================\n")
            for res in results:
                for line in res['log']:
                    lf.write(line + "\n")
    except Exception as e:
        print(f"Warning: Could not write full log file to '{log_file_path}': {e}")
        
    print("=" * 100)
    print("                      DETAILED COMPARISON REPORT (MISMATCHES ONLY)              ")
    print("=" * 100)
    if errors:
        limit = 50
        for err in errors[:limit]:
            for line in err['log']:
                print(line)
        if len(errors) > limit:
            print(f"\n... and {len(errors) - limit} more mismatches. See full log at: {log_file_path}")
    else:
        print("All files matched successfully!")
            
    matches_count = sum(1 for r in results if r['status'] == 'MATCH')
    mismatches_count = len(results) - matches_count
    print(f"\nComparison complete: {matches_count} MATCHES, {mismatches_count} MISMATCHES.")
    print(f"Full detailed log has been written to: {log_file_path}")
