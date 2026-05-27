import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
import os
import pickle
import warnings
import hashlib

# Abaikan warning agar terminal bersih
warnings.filterwarnings('ignore')

file_makro = "Makro Indikator AI.xlsx"
file_adb = "INO_02022026.xlsx"
CACHE_FILE = "policy_cache.pkl"

# ==========================================
# 0. INISIALISASI & CACHE
# ==========================================
try:
    USER_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    USER_API_KEY = ""

if 'policy_cache' not in st.session_state:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            st.session_state.policy_cache = pickle.load(f)
    else:
        st.session_state.policy_cache = {}

# Inisialisasi State Global agar aman dipanggil antar Tab
if 'scen_val' not in st.session_state: st.session_state.scen_val = 'med'
if 'yr_val' not in st.session_state: st.session_state.yr_val = 2026
if 'nt_val' not in st.session_state: st.session_state.nt_val = 16700
if 'oil_val' not in st.session_state: st.session_state.oil_val = 65
if 'current_avg' not in st.session_state: st.session_state.current_avg = 5.2
if 'monthly_summary_str' not in st.session_state: st.session_state.monthly_summary_str = "Data belum tersedia."
if 'heatmap_summary_str' not in st.session_state: st.session_state.heatmap_summary_str = "Data belum tersedia."
if 'daily_summary_str' not in st.session_state: st.session_state.daily_summary_str = "Data belum tersedia."
if 'sim_gdp' not in st.session_state: st.session_state.sim_gdp = 5.4
if 'sim_ca' not in st.session_state: st.session_state.sim_ca = -4.51
if 'sim_defpdb' not in st.session_state: st.session_state.sim_defpdb = -2.76
if 'base_gdp' not in st.session_state: st.session_state.base_gdp = 5.4

def make_signature(view, avg, target, monthly_info, daily_info, ext_info):
    raw_str = f"{view}_{avg:.2f}_{target}_{monthly_info}_{daily_info}_{ext_info}"
    return hashlib.md5(raw_str.encode()).hexdigest()

# ==========================================
# 1. FUNGSI LOADING DATA
# ==========================================
@st.cache_data
def load_data():
    try:
        df_target = pd.read_excel(file_makro, sheet_name=0)
        df_triwulan = pd.read_excel(file_makro, sheet_name=1)
        df_makro = pd.read_excel(file_makro, sheet_name=2)
        df_hist_gdp = pd.read_excel(file_adb, sheet_name=2)
        return df_target, df_triwulan, df_makro, df_hist_gdp
    except Exception as e:
        st.error(f"Error Loading Data: {e}")
        return None, None, None, None

@st.cache_data(ttl=3600)
def load_daily_data():
    try:
        url = "https://docs.google.com/spreadsheets/d/1wM0lHYqNTgf4Jo4AMCDakWnwqF1lVg-7/export?format=xlsx&gid=1981545536"
        df_daily = pd.read_excel(url, engine="openpyxl")
        date_col = 'Tanggal' if 'Tanggal' in df_daily.columns else df_daily.columns[0]
        df_daily[date_col] = pd.to_datetime(df_daily[date_col])
        df_daily = df_daily.sort_values(by=date_col)
        return df_daily, date_col
    except Exception as e:
        return None, None

df_target, df_triwulan, df_makro, df_hist_gdp = load_data()
df_daily, date_col_daily = load_daily_data()

# ==========================================
# 2. ENGINE DFM NOWCASTING (MAKRO NASIONAL)
# ==========================================
def apply_matlab_transformation(series, j1, j2, j3, freq='M'):
    out = series.copy().astype(float)
    if j1 == 1:
        out = out.mask(out <= 0, np.nan)
        out = 100 * np.log(out)
    if j2 == 1: out = out.diff(1)
    elif j3 == 1:
        lags = 12 if freq == 'M' else 4
        out = out.diff(lags)
    return out

def build_ragged_vintage(data_full, df_cal, indicator_col, vintage_cols, v_date, obs_cutoff):
    vintage = data_full[data_full.index <= obs_cutoff].copy()
    vcols_sorted = sorted([c for c in vintage_cols if c <= obs_cutoff])
    v_col_key = vcols_sorted[-1] if vcols_sorted else None
    if v_col_key is None: return vintage
    for _, row in df_cal.iterrows():
        ind = row[indicator_col]
        if ind not in vintage.columns: continue
        rd = pd.to_datetime(row[v_col_key], errors="coerce")
        if pd.notna(rd) and rd > v_date:
            mask_from = rd.replace(day=1) - pd.DateOffset(months=1)
            vintage.loc[vintage.index >= mask_from, ind] = np.nan
    return vintage

def get_prediction_value(pred_means, target, quarter):
    if target not in pred_means.columns: return np.nan
    q_end = quarter.to_timestamp(how="end").normalize()
    q_start = quarter.to_timestamp(how="start")
    for candidate in [q_start, q_end, q_start.replace(day=1), q_end.replace(day=1)]:
        if candidate in pred_means.index:
            return float(pred_means.loc[candidate, target])
    return np.nan

def calculate_annual_nowcast(pred_means, target_var, cutoff):
    year = cutoff.year
    vals = [get_prediction_value(pred_means, target_var, pd.Period(year=year, quarter=q, freq='Q')) for q in range(1, 5)]
    vals = [v for v in vals if pd.notna(v)]
    return np.mean(vals) if len(vals) == 4 else np.nan

@st.cache_data(show_spinner="⚙️ DFM Engine: Menghasilkan Histori Prediksi & Sinkronisasi Data Actual...")
def run_full_dfm_replication():
    try:
        df_m_raw = pd.read_excel(file_adb, sheet_name='MonthlyData', index_col=0, parse_dates=True)
        df_q_raw = pd.read_excel(file_adb, sheet_name='QuarterlyData', index_col=0, parse_dates=True)
        df_cal = pd.read_excel(file_adb, sheet_name='Calendar')
        info_m = pd.read_excel(file_adb, sheet_name='InfoM')
        info_q = pd.read_excel(file_adb, sheet_name='InfoQ')
        
        if "INCLUDE" in df_cal.columns: df_cal = df_cal[df_cal["INCLUDE"] == 1].reset_index(drop=True)
        indicator_col = df_cal.columns[0]
        vintage_cols = [pd.to_datetime(c) for c in df_cal.columns[2:]]

        processed_data = {}
        for _, row in info_m[info_m['INCLUDED'] == 1].iterrows():
            name = row['Indicator Code']
            if name in df_m_raw.columns:
                processed_data[name] = apply_matlab_transformation(df_m_raw[name], row['log'], row['MoM'], row['YoY'], 'M')
        for _, row in info_q[info_q['INCLUDED'] == 1].iterrows():
            name = row['Indicator Code']
            if name in df_q_raw.columns:
                s = apply_matlab_transformation(df_q_raw[name], row['log'], row['QoQ'], row['YoY'], 'Q')
                processed_data[name] = s

        data_full = pd.DataFrame(processed_data).replace([np.inf, -np.inf], np.nan).sort_index()
        data_full_resampled = data_full.resample('MS').first() 
        target_var = 'RGDP_growth'

        def get_actual_value(ref_period):
            target_date = ref_period.to_timestamp(how='end').replace(day=1).normalize()
            if target_date in data_full.index:
                val = data_full.loc[target_date, target_var]
                return val if pd.notna(val) else np.nan
            return np.nan

        jobs, seen, hari_ini = [], set(), pd.Timestamp.today().normalize()
        for vc in vintage_cols:
            col_name = vc.strftime('%Y-%m-%d 00:00:00') if vc.strftime('%Y-%m-%d 00:00:00') in df_cal.columns else df_cal.columns[2 + vintage_cols.index(vc)]
            release_dates = pd.to_datetime(df_cal[col_name], errors="coerce").dropna().unique()
            for rd in sorted(release_dates):
                if 2023 <= rd.year <= 2026 and rd <= hari_ini and (rd, vc) not in seen:
                    seen.add((rd, vc)); jobs.append((rd, vc)) 
        jobs.sort(key=lambda x: x[0])

        results_table = []
        for actual_v_date, v_date_base in jobs:
            obs_cutoff = v_date_base.replace(day=1)
            ref_q = pd.Period(actual_v_date, freq='Q')
            v_data = build_ragged_vintage(data_full_resampled, df_cal, indicator_col, vintage_cols, actual_v_date, obs_cutoff).dropna(axis=1, how='all')
            end_m = v_data.drop(columns=[target_var], errors='ignore')
            q_freq = "QE" if pd.__version__ >= "2.2.0" else "Q"
            if target_var in v_data.columns: end_q = v_data[[target_var]].resample(q_freq).last()
            else: end_q = data_full_resampled.loc[data_full_resampled.index <= obs_cutoff, [target_var]].resample(q_freq).last()
            
            model = DynamicFactorMQ(endog=end_m, endog_quarterly=end_q, k_factors=1, factor_orders=1, idiosyncratic_ar=1, standardize=True)
            res = model.fit(method='em', maxiter=500, tolerance=1e-5, disp=False)
            means = res.get_prediction(end=res.model.nobs + 24).predicted_mean
            
            results_table.append({
                'Day Prediction': actual_v_date, 'Reference Quarter': ref_q.strftime('%YQ%q'), 'Actual': get_actual_value(ref_q), 
                'Backcast': get_prediction_value(means, target_var, ref_q - 1), 'Nowcast': get_prediction_value(means, target_var, ref_q),
                'Forecast': get_prediction_value(means, target_var, ref_q + 1), '2-step': get_prediction_value(means, target_var, ref_q + 2),
                '3-step': get_prediction_value(means, target_var, ref_q + 3), 'Annual Nowcast': calculate_annual_nowcast(means, target_var, actual_v_date)
            })
        return pd.DataFrame(results_table)
    except Exception as e:
        return pd.DataFrame()


# ===============================================================================
# 3. ENGINE SEKTOR EKSTERNAL & FISKAL
# ===============================================================================
SCEN = {
    "med": {
        "nt":         {2026: 16700, 2027: 16500, 2028: 16300, 2029: 16200},
        "icp":        {2026: 65,    2027: 65,    2028: 65,    2029: 65},
        "ca":         {2026: -4.512,   2027: -19.096,  2028: 30.655,   2029: 3.750},
        "tradebal":   {2026: 47.226,   2027: 39.341,   2028: 47.772,   2029: 23.016},
        "exp":        {2026: 302.146,  2027: 343.369,  2028: 367.146,  2029: 381.850},
        "imp":        {2026: -254.920, 2027: -304.028, 2028: -319.374, 2029: -358.834},
        "svcbal":     {2026: -20.581,  2027: -18.602,  2028: -8.188,   2029: -10.202},
        "primbal":    {2026: -39.004,  2027: -49.990,  2028: -14.082,  2029: -14.907},
        "secbal":     {2026: 7.847,    2027: 10.155,   2028: 5.153,    2029: 5.842},
        "capbal":     {2026: 0.352,    2027: 0.352,    2028: 0.352,    2029: 0.352},
        "finbal":     {2026: 10.882,   2027: 25.073,   2028: 19.737,   2029: 6.940},
        "total":      {2026: 6.721,    2027: 6.329,    2028: 50.744,   2029: 11.042},
        "reserves":   {2026: 160.997,  2027: 165.131,  2028: 213.679,  2029: 222.526},
        "bulan_imp":  {2026: 6.008,    2027: 5.160,    2028: 6.810,    2029: 6.333},
        "capdb":      {2026: -0.272,   2027: -1.059,   2028: 1.499,    2029: 0.163},
        "gdpnom_usd": {2026: 1659.5,   2027: 1803.9,   2028: 2044.6,   2029: 2299.6},
        "gdp":        {2026: 5.4,    2027: 5.9,    2028: 7.7,    2029: 8.0},
        "cons":       {2026: 5.105,  2027: 8.254,  2028: 7.996,  2029: 8.008},
        "gov":        {2026: 7.854,  2027: 28.496, 2028: 7.099,  2029: 16.055},
        "inv":        {2026: 7.059,  2027: 9.841,  2028: 10.324, 2029: 11.149},
        "gexp":       {2026: 17.310, 2027: 5.132,  2028: -4.789, 2029: 1.538},
        "gimp":       {2026: 8.461,  2027: 13.907, 2028: -4.232, 2029: 7.928},
        "rev":        {2026: 3223.6,  2027: 3332.2,  2028: 5015.6,  2029: 6286.5},
        "bel":        {2026: 3936.8,  2027: 3816.0,  2028: 5526.9,  2029: 6396.4},
        "def":        {2026: -713.1,  2027: -483.8,  2028: -511.3,  2029: -109.9},
        "defpdb":     {2026: -2.765,  2027: -1.715,  2028: -1.635,  2029: -0.315},
        "sube":       {2026: 213.0,   2027: 186.4,   2028: 197.8,   2029: 234.9},
        "subnon":     {2026: 109.0,   2027: 112.1,   2028: 164.5,   2029: 203.1},
        "bunga":      {2026: 599.2,   2027: 622.5,   2028: 785.7,   2029: 785.7},
        "pajak":      {2026: 2725.7,  2027: 2825.1,  2028: 4311.0,  2029: 5466.6},
        "pnbp":       {2026: 497.3,   2027: 506.6,   2028: 699.0,   2029: 813.6},
        "migas":      {2026: 114.0,   2027: 131.7,   2028: 167.8,   2029: 194.8},
        "pdb":        {2026: 25788.9, 2027: 28211.7, 2028: 31283.1, 2029: 34938.3},
    },
    "high": {
        "nt":         {2026: 16700, 2027: 16500, 2028: 16300, 2029: 16200},
        "icp":        {2026: 65,    2027: 75,    2028: 75,    2029: 75},
        "ca":         {2026: -4.512,   2027: -6.802,   2028: 18.853,   2029: 10.329},
        "tradebal":   {2026: 47.226,   2027: 46.285,   2028: 63.627,   2029: 51.137},
        "exp":        {2026: 302.146,  2027: 341.304,  2028: 397.482,  2029: 427.061},
        "imp":        {2026: -254.920, 2027: -295.019, 2028: -333.856, 2029: -375.924},
        "svcbal":     {2026: -19.363,  2027: -19.060,  2028: -13.579,  2029: -13.500},
        "primbal":    {2026: -39.719,  2027: -42.193,  2028: -37.201,  2029: -35.580},
        "secbal":     {2026: 7.266,    2027: 8.421,    2028: 7.725,    2029: 8.239},
        "capbal":     {2026: 0.352,    2027: 0.352,    2028: 0.352,    2029: 0.352},
        "finbal":     {2026: 8.358,    2027: 13.112,   2028: 16.630,   2029: 14.697},
        "total":      {2026: 4.198,    2027: 6.662,    2028: 35.835,   2029: 25.378},
        "reserves":   {2026: 172.208,  2027: 186.145,  2028: 219.162,  2029: 208.666},
        "bulan_imp":  {2026: 6.008,    2027: 5.160,    2028: 6.810,    2029: 6.333},
        "capdb":      {2026: 0.602,    2027: 0.825,    2028: 1.151,    2029: 0.563},
        "gdpnom_usd": {2026: 1532.5,   2027: 1695.6,   2028: 1982.1,   2029: 2247.0},
        "gdp":        {2026: 5.4,    2027: 7.5,    2028: 7.7,    2029: 8.0},
        "cons":       {2026: 5.105,  2027: 5.727,  2028: 8.512,  2029: 8.525},
        "gov":        {2026: 7.796,  2027: 7.807,  2028: 9.557,  2029: 10.901},
        "inv":        {2026: 5.340,  2027: 8.845,  2028: 11.601, 2029: 12.410},
        "gexp":       {2026: 17.310, 2027: 8.779,  2028: -0.811, 2029: 3.454},
        "gimp":       {2026: 6.321,  2027: 9.746,  2028: 8.635,  2029: 8.000},
        "rev":        {2026: 3225.4,  2027: 3569.4,  2028: 5047.0,  2029: 6417.0},
        "bel":        {2026: 3938.4,  2027: 4210.9,  2028: 4977.3,  2029: 6314.1},
        "def":        {2026: -713.1,  2027: -641.5,  2028: 69.7,    2029: 102.9},
        "defpdb":     {2026: -2.765,  2027: -2.226,  2028: 0.219,   2029: 0.290},
        "sube":       {2026: 284.5,   2027: 192.8,   2028: 195.4,   2029: 194.1},
        "subnon":     {2026: 107.1,   2027: 114.3,   2028: 142.3,   2029: 293.9},
        "bunga":      {2026: 599.1,   2027: 634.4,   2028: 808.3,   2029: 947.1},
        "pajak":      {2026: 2725.0,  2027: 3025.6,  2028: 4356.1,  2029: 5558.6},
        "pnbp":       {2026: 499.7,   2027: 542.9,   2028: 685.2,   2029: 852.0},
        "migas":      {2026: 150.5,   2027: 136.0,   2028: 146.1,   2029: 222.7},
        "pdb":        {2026: 25788.9, 2027: 28824.4, 2028: 31810.0, 2029: 35524.1},
    },
}

EL = {
    "bop_exp_nt":      0.15,   "bop_imp_nt":      -0.25,
    "bop_exp_oil":     0.80,   "bop_imp_oil":      0.95,
    "bop_svc_nt":      0.05,   "bop_prim_nt":     -0.03,
    "share_exp_migas": 0.06,   "share_imp_migas":  0.16,
    "gexp_nt":         0.08,   "gimp_nt":         -0.06,
    "gexp_oil":        0.025,  "gimp_oil":         0.018,
    "w_exp":           0.27,   "w_imp":            0.22,
    "sube_oil":        0.95,   "sube_nt":         -0.15,
    "bunga_nt":        0.006,
}

YEARS = [2026, 2027, 2028, 2029]

C = {
    "blue":    "#2563eb", "green":   "#16a34a", "red":     "#dc2626", "red2":    "#b91c1c",
    "amber":   "#d97706", "purple":  "#7c3aed", "orange":  "#ea580c", "orange2": "#c2410c",
    "teal":    "#0891b2", "gray":    "#6e7681",
}

def simulate_eksternal(nt: float, oil: float, year: int, scen: str):
    D = SCEN[scen]
    b = {k: D[k].get(year, 0) for k in D}
    dNT      = (nt  - b["nt"])  / b["nt"]
    dOil     = (oil - b["icp"]) / b["icp"]
    dNT_pct  = dNT  * 100
    dOil_pct = dOil * 100
    s = {}

    exp_migas = b["exp"] * EL["share_exp_migas"]
    exp_non   = b["exp"] * (1 - EL["share_exp_migas"])
    imp_migas = b["imp"] * EL["share_imp_migas"]
    imp_non   = b["imp"] * (1 - EL["share_imp_migas"])

    s["exp"]        = exp_migas * (1 + EL["bop_exp_oil"] * dOil) + exp_non * (1 + EL["bop_exp_nt"] * dNT)
    s["imp"]        = imp_migas * (1 + EL["bop_imp_oil"] * dOil) + imp_non * (1 + EL["bop_imp_nt"] * dNT)
    s["tradebal"]   = s["exp"] + s["imp"]
    s["svcbal"]     = b["svcbal"]  * (1 + EL["bop_svc_nt"]  * dNT)
    s["primbal"]    = b["primbal"] * (1 + EL["bop_prim_nt"] * dNT)
    s["secbal"]     = b["secbal"]
    s["capbal"]     = b["capbal"]
    s["finbal"]     = b["finbal"]
    s["ca"]         = s["tradebal"] + s["svcbal"] + s["primbal"] + s["secbal"]
    s["total"]      = s["ca"] + s["capbal"] + s["finbal"]
    s["reserves"]   = b["reserves"] + (s["total"] - b["total"])
    s["gdpnom_usd"] = b["gdpnom_usd"] * (b["nt"] / nt)
    s["capdb"]      = (s["ca"] / s["gdpnom_usd"]) * 100

    base_imp_mo = abs(b["imp"]) / 12
    sim_imp_mo  = abs(s["imp"]) / 12
    if b.get("bulan_imp"):
        adj            = b["bulan_imp"] / (b["reserves"] / base_imp_mo)
        s["bulan_imp"] = (s["reserves"] / sim_imp_mo) * adj
    else:
        b["bulan_imp"] = b["reserves"] / base_imp_mo
        s["bulan_imp"] = s["reserves"] / sim_imp_mo

    dGexp_nt  = EL["gexp_nt"]  * dNT_pct
    dGimp_nt  = EL["gimp_nt"]  * dNT_pct
    dGexp_oil = EL["gexp_oil"] * dOil_pct
    dGimp_oil = EL["gimp_oil"] * dOil_pct

    s["gexp"] = b["gexp"] + dGexp_nt + dGexp_oil
    s["gimp"] = b["gimp"] + dGimp_nt + dGimp_oil
    
    # 🔥 LOGIKA SENSITIVITAS REVISI: NT/Minyak naik -> PDB Turun
    s["cons"] = b["cons"] - (0.04 * abs(dNT_pct)) - (0.02 * abs(dOil_pct)) if dNT_pct > 0 or dOil_pct > 0 else b["cons"]
    s["gov"]  = b["gov"]
    s["inv"]  = b["inv"] - (0.03 * abs(dNT_pct)) - (0.01 * abs(dOil_pct)) if dNT_pct > 0 or dOil_pct > 0 else b["inv"]
    s["gdp"]  = b["gdp"] - (0.015 * dNT_pct) - (0.01 * dOil_pct)

    dRevMigas = b["migas"] * EL["bop_exp_oil"] * dOil
    dPPH      = 55.6 * 0.7 * dOil
    dBea      = b["pajak"] * 0.019 * (dOil + dNT * 0.3)
    dRevTotal = dRevMigas + dPPH + dBea

    dSubsE    = b["sube"]  * (EL["sube_oil"] * dOil + EL["sube_nt"] * dNT)
    dBunga    = b["bunga"] * EL["bunga_nt"] * dNT
    dBelTotal = dSubsE + dBunga

    s["rev"]    = b["rev"]   + dRevTotal
    s["bel"]    = b["bel"]   + dBelTotal
    s["def"]    = s["rev"]   - s["bel"]
    s["defpdb"] = (s["def"]  / b["pdb"]) * 100
    s["sube"]   = b["sube"]  + dSubsE
    s["bunga"]  = b["bunga"] + dBunga
    s["pajak"]  = b["pajak"] + dBea
    s["migas"]  = b["migas"] + dRevMigas
    s["pnbp"]   = b["pnbp"]  + dRevMigas * 0.5
    s["subnon"] = b["subnon"]
    s["pdb"]    = b["pdb"]

    s["tx"] = {
        "expNT":  dGexp_nt, "impNT":  dGimp_nt, "netNT": -0.015 * dNT_pct,
        "expICP": dGexp_oil, "impICP": dGimp_oil, "netICP": -0.01 * dOil_pct,
    }

    s["ax"] = {
        "pph":   dPPH, "sda":   dRevMigas, "bea":   dBea, "rev":   dRevTotal,
        "sube":  dSubsE, "bunga": dBunga, "bel":   dBelTotal,
    }
    return b, s

# --- HELPER CHARTS ---
_BASE_LAYOUT = dict(
    paper_bgcolor="white", plot_bgcolor="#f8f9fa", font=dict(family="monospace", size=11, color="#4b5563"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=10,),
    margin=dict(l=40, r=20, t=50, b=40), height=270,
    xaxis=dict(gridcolor="#e5e7eb", showgrid=True), yaxis=dict(gridcolor="#e5e7eb", showgrid=True),
)
def fig_layout(title: str, barmode: str = None, **kwargs) -> dict:
    layout = {**_BASE_LAYOUT, "title": dict(text=title, font=dict(size=13)), **kwargs}
    if barmode: layout["barmode"] = barmode
    return layout
def bar_trace(name, x, y, color, opacity=0.75):
    return go.Bar(name=name, x=x, y=y, marker_color=color, marker_line_width=0, opacity=opacity)
def line_trace(name, x, y, color, fill=None, fillcolor=None, width=2, size=6):
    return go.Scatter(name=name, x=x, y=y, mode="lines+markers", line=dict(color=color, width=width), marker=dict(size=size, color=color), fill=fill, fillcolor=fillcolor)
def line_base_trace(name, x, y):
    return go.Scatter(name=name, x=x, y=y, mode="lines+markers", line=dict(color=C["gray"], dash="dash", width=1.5), marker=dict(size=4, color=C["gray"]))
def dot_trace(name, x, y):
    return go.Scatter(name=name, x=x, y=y, mode="markers", marker=dict(size=11, color=C["amber"], line=dict(color="white", width=2)))
def delta_color(val: float) -> str:
    if val > 0.005:   return "green"
    if val < -0.005:  return "red"
    return "gray"
def metric_delta_color(val: float) -> str:
    dc = delta_color(val)
    if dc == "green": return "normal"
    if dc == "red":   return "inverse"
    return "off"

# ==========================================
# 4. TAMPILAN HEADER & UI
# ==========================================
st.set_page_config(page_title="Macro AI Command Center", layout="wide", page_icon="🇮🇩", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background: radial-gradient(circle at 10% 20%, rgb(242, 243, 247) 0%, rgb(215, 221, 232) 90.2%); }
    .glass-card {
        background: rgba(255, 255, 255, 0.65);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.7);
        padding: 24px; margin-bottom: 24px;
    }
    .main-header {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        color: white; padding: 14px 20px; border-radius: 10px;
        margin-bottom: 16px; display: flex; align-items: center; gap: 14px;
    }
    .logo-box {
        background: white; color: #1d4ed8; width: 38px; height: 38px;
        border-radius: 7px; display: flex; align-items: center;
        justify-content: center; font-weight: 800; font-size: 14px; flex-shrink: 0;
    }
    .hdr-title { font-size: 17px; font-weight: 700; }
    .hdr-sub   { font-size: 12px; opacity: 0.8; margin-top: 2px; }
    .card-title { font-size: 13px; color: #444; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .card-value { font-size: 26px; color: #111; font-weight: 800; margin: 4px 0; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 8px; font-size: 11px; font-weight: 700; margin-right: 6px; }
    .badge-green { background: rgba(212, 237, 218, 0.8); color: #155724; }
    .badge-red { background: rgba(248, 215, 218, 0.8); color: #721c24; }
    .badge-neutral { background: rgba(226, 227, 229, 0.8); color: #383d41; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div class="logo-box">ID</div>
    <div>
        <div class="hdr-title">Dashboard Makroekonomi & Pembangunan RI</div>
        <div class="hdr-sub">BOP &middot; Pertumbuhan Ekonomi &middot; Defisit APBN &mdash; Sistem Pendukung Keputusan Strategis</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 5. STRUKTUR TABS UTAMA (NAVIGASI ATAS)
# ==========================================
tab_makro, tab_eksternal, tab_daerah, tab_ai = st.tabs([
    "📊 Modul 1: Makro Nasional (DFM)", 
    "🌍 Modul 2: Sektor Eksternal & Fiskal", 
    "📍 Modul 3: Ekonomi Daerah",
    "🧠 Modul 4: AI Strategic Executive Brief"
])

# ===============================================================================
# TAB 1: MAKRO NASIONAL
# ===============================================================================
with tab_makro:
    if df_target is not None:
        t_2025 = df_target[df_target['Tahun'] == 2025]['Target'].values[0]
        row_2025 = df_triwulan[df_triwulan['Tahun'] == 2025].iloc[0]

        real_2025, now_2025, combined_2025 = [], [], []
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            r = row_2025.get(f'Realisasi {q}', np.nan)
            n = row_2025.get(f'Nowcasting {q}', np.nan)
            real_2025.append(r if pd.notna(r) else None)
            now_2025.append(n if pd.notna(n) else None)
            val = r if pd.notna(r) else (n if pd.notna(n) else 5.0)
            combined_2025.append(val)

        t_2026 = df_target[df_target['Tahun'] == 2026]['Target'].values[0] if 2026 in df_target['Tahun'].values else 5.4
        
        df_full_results = run_full_dfm_replication()
        
        if not df_full_results.empty:
            preds_2026 = []
            latest_row = df_full_results.sort_values('Day Prediction').iloc[-1]
            ref_q_str = latest_row['Reference Quarter'] 
            ref_year = int(ref_q_str[:4])
            ref_q_num = int(ref_q_str[-1])
            
            for target_q in [1, 2, 3, 4]:
                distance = (2026 - ref_year) * 4 + (target_q - ref_q_num)
                mapping_kolom = {-1: 'Backcast', 0: 'Nowcast', 1: 'Forecast', 2: '2-step', 3: '3-step'}
                nama_kolom = mapping_kolom.get(distance)
                if nama_kolom and nama_kolom in latest_row and pd.notna(latest_row[nama_kolom]):
                    preds_2026.append(float(latest_row[nama_kolom]))
                else:
                    fallback_df = df_full_results[df_full_results['Reference Quarter'] == f"2026Q{target_q}"]
                    if not fallback_df.empty: preds_2026.append(float(fallback_df.sort_values('Day Prediction').iloc[-1]['Nowcast']))
                    else: preds_2026.append(np.nan)
            s_preds = pd.Series(preds_2026)
            preds_2026 = s_preds.ffill().bfill().fillna(5.2).tolist()
        else:
            preds_2026 = [5.1, 5.2, 5.3, 5.4]

        real_2026 = [None, None, None, None]
        df_triwulan['Tahun_str'] = df_triwulan['Tahun'].astype(str).str.strip().str.replace('.0', '', regex=False)
        
        if '2026' in df_triwulan['Tahun_str'].values:
            row_2026 = df_triwulan[df_triwulan['Tahun_str'] == '2026'].iloc[0]
            for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
                col_name = f'Realisasi {q}'
                if col_name in row_2026.index:
                    r = row_2026[col_name]
                    try: real_2026[i] = float(r) if pd.notna(r) and str(r).strip() != '' else None
                    except: pass

        now_2026 = preds_2026
        header_ui = st.container()

        selected_view = st.radio("Pilih Rentang Waktu Analisis:", ["2026", "2010 - 2026"], horizontal=True, index=0)

        final_x, final_real, final_now, final_target = [], [], [], []
        current_avg, current_target = 0, t_2026

        x_2026 = ['2026-Q1', '2026-Q2', '2026-Q3', '2026-Q4']
        valid_x_2026_real = [x_2026[i] for i, r in enumerate(real_2026) if r is not None]
        valid_y_2026_real = [r for r in real_2026 if r is not None]

        if selected_view == "2026":
            final_x = ['Q1', 'Q2', 'Q3', 'Q4']
            final_real, final_now, final_target = real_2026, now_2026, [t_2026]*4
            current_avg = np.mean(now_2026)
        else: 
            df_h = df_hist_gdp.copy()
            try:
                if pd.api.types.is_numeric_dtype(df_h.iloc[:, 0]): df_h.iloc[:, 0] = pd.to_datetime(df_h.iloc[:, 0], unit='D', origin='1899-12-30')
                else: df_h.iloc[:, 0] = pd.to_datetime(df_h.iloc[:, 0])
            except: pass
            df_h.set_index(df_h.columns[0], inplace=True)
            col_target = 'RGDP_growth' if 'RGDP_growth' in df_h.columns else df_h.columns[1]
            series_hist = df_h[col_target].dropna()
            series_hist = series_hist[series_hist.index >= '2010-01-01']
            try: x_hist = [f"{d.year}-Q{(d.month-1)//3 + 1}" for d in series_hist.index]
            except: x_hist = [str(i) for i in range(len(series_hist))]
            y_hist = series_hist.values.tolist()
            x_2025 = ['2025-Q1', '2025-Q2', '2025-Q3', '2025-Q4']
            full_x_real = x_hist + x_2025 + valid_x_2026_real
            full_y_real = y_hist + combined_2025 + valid_y_2026_real
            
            if valid_y_2026_real:
                last_real_x = valid_x_2026_real[-1]
                last_real_y = valid_y_2026_real[-1]
                sisa_q = len(valid_y_2026_real) 
                full_x_proj = [last_real_x] + x_2026[sisa_q:]
                full_y_proj = [last_real_y] + preds_2026[sisa_q:]
            else:
                full_x_proj = [x_2025[-1]] + x_2026
                full_y_proj = [combined_2025[-1]] + preds_2026
            current_avg = np.mean(preds_2026)

        st.session_state.current_avg = current_avg

        title_text = f"Outlook Ekonomi: {selected_view}"
        if selected_view == "2026": title_text += " (Model: Dynamic Factor MQ)"
        else: title_text = "Historis & Proyeksi Ekonomi (DFM Model)"

        with header_ui:
            st.markdown(f"### {title_text}")
            realisasi_bps_ctc = 5.61 
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Target Acuan", f"{current_target}%")
            if realisasi_bps_ctc is not None:
                gap_realisasi = realisasi_bps_ctc - current_target
                c2.metric("Realisasi BPS (c-t-c)", f"{realisasi_bps_ctc:.2f}%", delta=f"{gap_realisasi:.2f}%")
                c2.caption("Capaian Triwulan I-2026")
            else:
                c2.metric("Realisasi BPS (c-t-c)", "Belum Rilis", delta="-", delta_color="off")
                
            gap_proyeksi = current_avg - current_target
            c3.metric("Proyeksi DFM (Avg)", f"{current_avg:.2f}%", delta=f"{gap_proyeksi:.2f}%")
            angka_acuan_status = realisasi_bps_ctc if realisasi_bps_ctc is not None else current_avg
            gap_status = angka_acuan_status - current_target
            status = "✅ SESUAI TARGET" if gap_status >= -0.1 else "❌ BELOW TARGET"
            c4.metric("Status Capaian", status, delta_color="normal" if gap_status >= -0.1 else "inverse")

        fig = go.Figure()
        if selected_view == "2010 - 2026":
            latest_q_real = valid_x_2026_real[-1].split('-')[-1] if valid_x_2026_real else "Q4 2025"
            legend_realisasi = f"Realisasi (Q1 2010-{latest_q_real} 2026)"
            fig.add_trace(go.Scatter(x=full_x_real, y=full_y_real, name=legend_realisasi, mode='lines', line=dict(color='#f1c40f', width=2.5)))
            fig.add_trace(go.Scatter(x=full_x_proj, y=full_y_proj, name='Proyeksi DFM 2026', mode='lines', line=dict(color='#27ae60', width=2.5, dash='dot')))
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1), height=450)
        else:
            fig.add_trace(go.Bar(
                x=final_x, y=final_real, name='Realisasi (BPS)', marker_color='#2980b9', 
                text=[f"{v:.2f}%" if v else "" for v in final_real], 
                textposition='inside', insidetextanchor='middle', textfont=dict(color='white', size=14)
            ))
            fig.add_trace(go.Scatter(x=final_x, y=final_now, name='DFM Nowcasting', mode='lines+markers', line=dict(color='#f39c12', width=4, shape='spline'), text=[f"{v:.2f}%" for v in final_now], textposition='top center'))
            fig.add_trace(go.Scatter(x=final_x, y=final_target, name='Target APBN', mode='lines', line=dict(color='#c0392b', width=3, dash='dash')))
            fig.update_layout(barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1), height=450)

        for trace in fig.data:
            trace_name = getattr(trace, 'name', '')
            if "Realisasi" in trace_name and "2010-" in trace_name:
                text_labels, marker_sizes, text_pos = [], [], []
                if trace.x is not None and trace.y is not None:
                    for i, y_val in enumerate(trace.y):
                        if i == len(trace.x) - 1 and pd.notna(y_val): 
                            text_labels.append(f"<b>{float(y_val):.2f}%</b>")
                            marker_sizes.append(10)
                            text_pos.append("top center") 
                        else:
                            text_labels.append(""); marker_sizes.append(0); text_pos.append("top center")
                    trace.mode = "lines+markers+text"
                    trace.text = text_labels
                    trace.textposition = text_pos 
                    trace.textfont = dict(size=13, color="#0f172a") 
                    if not hasattr(trace, 'marker') or trace.marker is None: trace.marker = dict()
                    trace.marker.size = marker_sizes
                    trace.marker.symbol = "circle"
                    trace.marker.color = "#f1c40f"
                    trace.marker.line = dict(width=2, color="white")
                    
            elif trace_name == 'Proyeksi DFM 2026':
                text_labels, marker_sizes, text_pos = [], [], []
                if trace.x is not None and trace.y is not None:
                    pos_toggle = True
                    for i, (x_val, y_val) in enumerate(zip(trace.x, trace.y)):
                        if i > 0 and '2026' in str(x_val) and pd.notna(y_val):
                            text_labels.append(f"<b>{float(y_val):.2f}%</b>")
                            marker_sizes.append(10)
                            text_pos.append("bottom center" if pos_toggle else "top center")
                            pos_toggle = not pos_toggle
                        else:
                            text_labels.append(""); marker_sizes.append(0); text_pos.append("top center")
                    trace.mode = "lines+markers+text"
                    trace.text = text_labels
                    trace.textposition = text_pos 
                    trace.textfont = dict(size=13, color="#0f172a")
                    if not hasattr(trace, 'marker') or trace.marker is None: trace.marker = dict()
                    trace.marker.size = marker_sizes
                    trace.marker.symbol = "circle"
                    trace.marker.color = "#27ae60"
                    trace.marker.line = dict(width=2, color="white")
                    
            elif trace_name == 'DFM Nowcasting':
                text_labels, marker_sizes = [], []
                if trace.x is not None and trace.y is not None:
                    for y_val in trace.y:
                        if pd.notna(y_val):
                            text_labels.append(f"<b>{float(y_val):.2f}%</b>")
                            marker_sizes.append(11)
                        else:
                            text_labels.append(""); marker_sizes.append(0)
                    trace.mode = "lines+markers+text"
                    trace.text = text_labels
                    trace.textposition = "top center" 
                    trace.textfont = dict(size=14, color="#0f172a")
                    if not hasattr(trace, 'marker') or trace.marker is None: trace.marker = dict()
                    trace.marker.size = marker_sizes
                    trace.marker.symbol = "circle"
                    trace.marker.color = "#f39c12" 
                    trace.marker.line = dict(width=2, color="white")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 📈 Monitoring Data Harian")
        selected_daily_view = st.radio("Pilih Mode Tampilan Pasar:", ["Data Berjalan", "Data Rata-Rata"], horizontal=True, key="daily_view_toggle")
        
        daily_summary_list, daily_berjalan_list, daily_rata_list = [], [], []
        
        if 'df_daily' in locals() and df_daily is not None:
            daily_cols = st.columns(4)
            daily_indicators = ['IHSG', 'Saham Daily', 'Obligasi Daily', 'Brent', 'WTI', 'CPO', 'Emas', 'Batubara', 'Natural Gas', 'Nikel']
            idx = 0
            for col in daily_indicators:
                if col not in df_daily.columns: continue
                valid_series = df_daily[[date_col_daily, col]].dropna()
                if valid_series.empty: continue
                latest_row = valid_series.iloc[-1]
                val = latest_row[col] 
                date_obj = latest_row[date_col_daily]
                date_str = date_obj.strftime("%d %b %Y")
                current_year = date_obj.year
                
                if len(valid_series) > 1:
                    prev_row = valid_series.iloc[-2]
                    val_prev = prev_row[col]
                    dtd = ((val - val_prev) / val_prev) * 100 if val_prev != 0 else 0
                else: dtd = 0
                    
                prev_year_data = valid_series[valid_series[date_col_daily].dt.year == current_year - 1]
                if not prev_year_data.empty:
                    ytd_base_val = prev_year_data.iloc[-1][col]
                    ytd = ((val - ytd_base_val) / ytd_base_val) * 100 if ytd_base_val != 0 else 0
                    ytd_str = f"YTD: {ytd:+.2f}%"
                else: ytd, ytd_str = 0, "YTD: -"
                
                current_year_data = valid_series[valid_series[date_col_daily].dt.year == current_year]
                avg_current = current_year_data[col].mean() if not current_year_data.empty else val
                avg_prev = prev_year_data[col].mean() if not prev_year_data.empty else 0
                avg_growth = ((avg_current - avg_prev) / avg_prev) * 100 if avg_prev != 0 else 0

                disp_val_b = f"{val:,.2f}" if val > 10 else f"{val:.2f}"
                daily_berjalan_list.append(f"{col}: {disp_val_b} (DTD: {dtd:+.2f}%, {ytd_str})")
                disp_val_r = f"{avg_current:,.2f}" if avg_current > 10 else f"{avg_current:.2f}"
                daily_rata_list.append(f"{col}: Avg {current_year} = {disp_val_r} (Perubahan vs Avg 2025: {avg_growth:+.2f}%)")

                if "Berjalan" in selected_daily_view:
                    disp_val = disp_val_b
                    color_1 = "badge-red" if dtd < 0 else "badge-green"
                    color_2 = "badge-red" if ytd < 0 else "badge-green"
                    badge_1_str, badge_2_str = f"DTD: {dtd:+.2f}%", ytd_str
                    subtitle_str = f"Data Spot: {date_str}"
                    daily_summary_list.append(f"{col}: {disp_val_b} (DTD: {dtd:+.2f}%)")
                else:
                    disp_val = disp_val_r
                    color_1 = "badge-neutral" 
                    color_2 = "badge-red" if avg_growth < 0 else "badge-green"
                    avg_prev_disp = f"{avg_prev:,.2f}" if avg_prev > 10 else f"{avg_prev:.2f}"
                    badge_1_str, badge_2_str = f"Avg '25: {avg_prev_disp}", f"Δ {avg_growth:+.2f}%"
                    subtitle_str = f"Rata-rata YTD {current_year}"
                    daily_summary_list.append(f"{col}: Avg {current_year} = {disp_val_r} (Perubahan vs Avg 2025: {avg_growth:+.2f}%)")

                html = f"""
                <div class="glass-card" style="padding: 15px; margin-bottom: 10px;">
                    <div class="card-title">{col}</div>
                    <div class="card-value">{disp_val}</div>
                    <div style="font-size: 11px; color: #666; margin-bottom: 8px; font-style: italic;">{subtitle_str}</div>
                    <span class="badge {color_1}">{badge_1_str}</span>
                    <span class="badge {color_2}">{badge_2_str}</span>
                </div>
                """
                with daily_cols[idx % 4]: st.markdown(html, unsafe_allow_html=True)
                idx += 1
                
            if daily_summary_list: st.session_state.daily_summary_str = " | ".join(daily_summary_list)
                
        st.markdown("<br>", unsafe_allow_html=True)

        ATURAN_WARNA = {
            'PMI Manufaktur Negara Berkembang': True, 'Jumlah Uang Yang Beredar': True, 
            'Penjualan Mobil': True, 'Penjualan semen': True, 'Ekspor Barang': True, 
            'Impor Barang Modal': True, 'Impor Bahan Baku': True, 'Kredit Perbankan': True, 
            'Penjualan Motor': True, 'Indeks Keyakinan Konsumen': True, 'Impor Barang Konsumsi': True, 
            'Inflasi': False, 'Nilai Tukar terhadap Dolar AS': False, 'Suku Bunga': False
        }

        st.markdown("### 🔍 Deep Dive: Indikator Makro (Real Sector)")
        df_makro['Tanggal'] = pd.to_datetime(df_makro['Tanggal'])
        df_makro = df_makro.sort_values(by='Tanggal')
        
        cols = st.columns(4)
        monthly_summary_list = [] 
        indicator_cols = [c for c in df_makro.columns if c != 'Tanggal']

        for i, col in enumerate(indicator_cols):
            valid_series = df_makro[['Tanggal', col]].dropna()
            if valid_series.empty: continue
            latest_row = valid_series.iloc[-1]
            val = latest_row[col]
            date_obj = latest_row['Tanggal']
            date_str = date_obj.strftime("%b %Y")
            
            if len(valid_series) > 1:
                prev_row = valid_series.iloc[-2]
                val_prev_mtm = prev_row[col]
                mtm_diff = val - val_prev_mtm
                mtm_pct = (mtm_diff / abs(val_prev_mtm)) * 100 if val_prev_mtm != 0 else 0
            else: mtm_diff, mtm_pct = 0, 0

            target_date_yoy = date_obj - pd.DateOffset(years=1)
            row_yoy = df_makro[(df_makro['Tanggal'].dt.year == target_date_yoy.year) & (df_makro['Tanggal'].dt.month == target_date_yoy.month)]
            
            if not row_yoy.empty and pd.notna(row_yoy.iloc[0][col]):
                val_yoy = row_yoy.iloc[0][col]
                yoy_diff = val - val_yoy
                yoy_pct = (yoy_diff / abs(val_yoy)) * 100 if val_yoy != 0 else 0
                has_yoy = True
            else: yoy_diff, yoy_pct, has_yoy = 0, 0, False

            rule_naik_bagus = ATURAN_WARNA.get(col, True)
            is_level_indicator = any(k in col for k in ["PMI", "Inflasi", "Suku Bunga", "Nilai Tukar", "Indeks Keyakinan Konsumen"])

            if is_level_indicator:
                disp = f"{val:,.2f}" if val > 1000 else f"{val:.2f}"
                badge_1 = f"MtM: {mtm_diff:+.2f}"
                badge_2 = f"YoY: {yoy_diff:+.2f}" if has_yoy else "YoY: -"
                is_bad_mtm = (rule_naik_bagus and mtm_diff < 0) or (not rule_naik_bagus and mtm_diff > 0)
                is_bad_yoy = (rule_naik_bagus and yoy_diff < 0) or (not rule_naik_bagus and yoy_diff > 0)
            else:
                disp = f"{val:,.2f}"
                badge_1 = f"MtM: {mtm_pct:+.2f}%"
                badge_2 = f"YoY: {yoy_pct:+.2f}%" if has_yoy else "YoY: -"
                is_bad_mtm = (rule_naik_bagus and mtm_pct < 0) or (not rule_naik_bagus and mtm_pct > 0)
                is_bad_yoy = (rule_naik_bagus and yoy_pct < 0) or (not rule_naik_bagus and yoy_pct > 0)

            if "PMI" in col and val < 50: is_bad_mtm, is_bad_yoy = True, True
            color_1 = "badge-red" if is_bad_mtm else "badge-green"
            color_2 = "badge-red" if is_bad_yoy else "badge-green"
            status_mtm = "Melemah" if is_bad_mtm else "Membaik"
            status_yoy = "Melemah" if is_bad_yoy else "Membaik"
            monthly_summary_list.append(f"[{col}] Data: {disp} | {badge_1} ({status_mtm}) | {badge_2} ({status_yoy})")

            html = f"""
            <div class="glass-card" style="padding: 15px; margin-bottom: 10px;">
                <div class="card-title">{col}</div>
                <div class="card-value">{disp}</div>
                <div style="font-size: 11px; color: #666; margin-bottom: 8px; font-style: italic;">Data: {date_str}</div>
                <span class="badge {color_1}">{badge_1}</span>
                <span class="badge {color_2}">{badge_2}</span>
            </div>
            """
            with cols[i%4]: st.markdown(html, unsafe_allow_html=True)
            
        if monthly_summary_list: st.session_state.monthly_summary_str = "\n".join(monthly_summary_list)

        st.markdown("### 🗺️ Heatmap Tracker (Tren YoY & Threshold Target)")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        df_hm = df_makro[df_makro['Tanggal'] >= '2025-01-01'].copy()
        heatmap_summary_list = [] 
        
        if not df_hm.empty:
            dates_hm = df_hm['Tanggal'].tolist()
            x_labels = df_hm['Tanggal'].dt.strftime('%b %Y').tolist()
            z_data, text_data = [], []
            for col in indicator_cols:
                col_z, col_text = [], []
                rule_naik_bagus = ATURAN_WARNA.get(col.strip(), True)
                is_level_indicator = any(k in col for k in ["PMI", "Inflasi", "Suku Bunga", "Nilai Tukar", "Indeks Keyakinan Konsumen"])
                    
                for d in dates_hm:
                    curr_row = df_makro[df_makro['Tanggal'] == d]
                    val = curr_row[col].values[0] if not curr_row.empty else np.nan
                    prev_d = d - pd.DateOffset(years=1)
                    prev_row = df_makro[(df_makro['Tanggal'].dt.year == prev_d.year) & (df_makro['Tanggal'].dt.month == prev_d.month)]
                    val_prev = prev_row[col].values[0] if not prev_row.empty else np.nan
                    prev_prev_d = prev_d - pd.DateOffset(years=1)
                    prev_prev_row = df_makro[(df_makro['Tanggal'].dt.year == prev_prev_d.year) & (df_makro['Tanggal'].dt.month == prev_prev_d.month)]
                    val_prev_prev = prev_prev_row[col].values[0] if not prev_prev_row.empty else np.nan
                    
                    if pd.isna(val) or pd.isna(val_prev):
                        col_z.append(0); col_text.append("-")
                    else:
                        if is_level_indicator:
                            txt = f"{val:,.2f}" if val > 1000 else f"{val:.2f}"
                            diff = val - val_prev
                        else:
                            yoy_curr = (val - val_prev) / abs(val_prev) * 100 if val_prev != 0 else 0
                            txt = f"{yoy_curr:+.2f}%"
                            if pd.isna(val_prev_prev): diff = yoy_curr
                            else:
                                yoy_prev = (val_prev - val_prev_prev) / abs(val_prev_prev) * 100 if val_prev_prev != 0 else 0
                                diff = yoy_curr - yoy_prev
                            
                        is_green, is_special_indicator = False, False
                        if "PMI" in col: is_special_indicator, is_green = True, val >= 50.0
                        elif "Inflasi" in col: is_special_indicator, is_green = True, 1.5 <= val <= 3.5
                        elif "Nilai Tukar" in col: is_special_indicator, is_green = True, val <= 16900
                            
                        if is_special_indicator: col_z.append(1 if is_green else -1)
                        else:
                            if diff == 0: col_z.append(0) 
                            elif rule_naik_bagus: is_green = diff > 0; col_z.append(1 if is_green else -1)
                            else: is_green = diff < 0; col_z.append(1 if is_green else -1)
                            
                        col_text.append(txt)
                        if d == dates_hm[-1]:
                            sentimen = "Positif/Aman (Hijau)" if is_green else "Negatif/Waspada (Merah)"
                            heatmap_summary_list.append(f"{col}: Kondisi {sentimen} ({txt})")
                
                z_data.append(col_z); text_data.append(col_text)
                
            if heatmap_summary_list: st.session_state.heatmap_summary_str = " | ".join(heatmap_summary_list)
                
            fig_hm = go.Figure(data=go.Heatmap(
                z=z_data, x=x_labels, y=indicator_cols, text=text_data, texttemplate="<b>%{text}</b>", 
                textfont=dict(size=14, color='#111'), colorscale=[[0.0, '#e74c3c'], [0.5, '#ecf0f1'], [1.0, '#2ecc71']], 
                zmin=-1, zmax=1, showscale=False, xgap=3, ygap=3
            ))
            fig_hm.update_layout(height=150 + len(indicator_cols)*35, margin=dict(l=220, r=20, t=30, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis=dict(autorange="reversed", tickfont=dict(size=12, color='#333', weight='bold')))
            st.plotly_chart(fig_hm, use_container_width=True)
            
            st.markdown("""
            <div style='font-size: 11.5px; color: #475569; background: #f8fafc; padding: 12px 15px; border-radius: 10px; border: 1px solid #cbd5e1; line-height: 1.6;'>
                <strong>Keterangan Momentum Umum:</strong> 🟩 Mengalami Perbaikan Momentum (vs Tahun Lalu) | 🟥 Mengalami Perlambatan Momentum | ⬜ Stagnan / Belum Rilis <br>
                <strong>Keterangan Threshold Khusus:</strong> 
                <span style='background:#dcfce7; color:#166534; padding:2px 6px; border-radius:4px;'>🟩 PMI Manufaktur (≥ 50)</span> | 
                <span style='background:#dcfce7; color:#166534; padding:2px 6px; border-radius:4px;'>🟩 Inflasi (1.5% - 3.5%)</span> | 
                <span style='background:#dcfce7; color:#166534; padding:2px 6px; border-radius:4px;'>🟩 Nilai Tukar (< 16.900)</span>
                <br><em>*Khusus 3 indikator di atas, warna merah 🟥 menandakan realisasi keluar dari batas rentang sasaran wajar (Threshold).</em>
            </div>
            """, unsafe_allow_html=True)
        else: st.info("Belum ada data bulanan untuk ditampilkan.")
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# TAB 2: SEKTOR EKSTERNAL & FISKAL
# =========================================================================
with tab2:
    st.markdown("### ⚙️ Parameter Simulasi Skenario")
    
    col_st1, col_st2, col_st3 = st.columns([1, 1, 2])
    scen_v = col_st1.radio("Skenario Baseline:", ["med", "high"], horizontal=True, format_func=lambda x: "Med" if x == "med" else "High", key="scen_v_input")
    yr_v = col_st2.select_slider("Tahun Proyeksi:", options=YEARS, value=2026, key="yr_v_input")
    preset_v = col_st3.selectbox("Preset Simulasi Cepat:", ["-- pilih --", "📊 Base Med", "📈 Base High", "📉 Depresiasi (NT 19.500)", "🛢 Minyak Rendah ($40)", "🔥 Minyak Tinggi ($100)", "⚡ Twin Shock (NT 20.000, ICP $100)"], key="preset_v_input")

    D = SCEN[scen_v]
    nt_default = D["nt"][yr_v]
    icp_default = D["icp"][yr_v]

    if   "Depresiasi" in preset_v: nt_init, oil_init = 19_500,       icp_default
    elif "Rendah"     in preset_v: nt_init, oil_init = nt_default,  40
    elif "Tinggi"     in preset_v: nt_init, oil_init = nt_default,  100
    elif "Twin"       in preset_v: nt_init, oil_init = 20_000,      100
    else:                          nt_init, oil_init = nt_default,  icp_default

    st.markdown("---")
    
    col_inp1, col_inp2 = st.columns(2)
    nt_v = col_inp1.number_input("Nilai Tukar (Rp/USD)", min_value=10_000, max_value=30_000, value=nt_init, step=50, key="nt_v_input")
    col_inp1.caption(f"Baseline {scen_v.upper()} {yr_v}: Rp{nt_default:,} — nilai naik = depresiasi Rupiah")
    
    oil_v = col_inp2.number_input("Harga Minyak ICP (USD/bbl)", min_value=20, max_value=150, value=oil_init, step=1, key="oil_v_input")
    col_inp2.caption(f"Baseline {scen_v.upper()} {yr_v}: ${icp_default} — kenaikan ICP meningkatkan penerimaan & subsidi")

    # Save to state for AI
    st.session_state.scen_val, st.session_state.yr_val, st.session_state.nt_val, st.session_state.oil_val = scen_v, yr_v, nt_v, oil_v

    # Jalankan simulasi
    b, s = simulate_eksternal_v2(nt_v, oil_v, yr_v, scen_v)
    
    # Save to state for AI
    st.session_state.sim_gdp, st.session_state.sim_ca, st.session_state.sim_defpdb = s["gdp"], s["ca"], s["defpdb"]
    st.session_state.base_gdp = b["gdp"]

    st.markdown(f"**DELTA VS BASELINE {yr_v}**")
    delta_rows = [
        ("Neraca Berjalan", s["ca"]       - b["ca"],       " Miliar USD"),
        ("Cadangan Devisa", s["reserves"] - b["reserves"], " Miliar USD"),
        ("PDB Growth",      s["gdp"]      - b["gdp"],      " pp"),
        ("Ekspor Riil",     s["gexp"]     - b["gexp"],     " pp"),
        ("Defisit APBN",    s["def"]      - b["def"],      " T"),
        ("Defisit/PDB",     s["defpdb"]   - b["defpdb"],   " pp"),
    ]
    
    d_cols = st.columns(6)
    for i, (lbl, dv, sfx) in enumerate(delta_rows):
        with d_cols[i]:
            sign  = "+" if dv >= 0 else ""
            color = delta_color(dv)
            st.markdown(f"<div style='font-size:12px; color:#475569;'>{lbl}</div><div style='font-weight:bold; color:{'#16a34a' if color == 'green' else '#dc2626' if color == 'red' else '#64748b'};'>{sign}{dv:.2f}{sfx}</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    keys_sim = ["ca", "exp", "imp", "reserves", "gdp", "gexp", "gimp", "def", "rev", "bel", "sube", "bunga", "pajak"]
    R = {k: {"b": [], "s": []} for k in keys_sim}

    for y in YEARS:
        bb, ss = simulate_eksternal_v2(nt_v, oil_v, y, scen_v)
        for k in keys_sim:
            R[k]["b"].append(round(bb[k], 2))
            R[k]["s"].append(round(ss[k], 2))

    YL = [str(y) for y in YEARS]

    tab_bop, tab_gdp2, tab_apbn, tab_table = st.tabs([
        "📊 Neraca Pembayaran (BOP)",
        "📈 Pertumbuhan Ekonomi (GDP)",
        "🏛 Defisit APBN",
        "📋 Tabel Lengkap",
    ])

    with tab_bop:
        bop_kpis = [
            ("🔵 Transaksi Berjalan",  f"{s['ca']:.2f}",       "Miliar USD", s["ca"]-b["ca"],              " Miliar USD"),
            ("🟢 Ekspor Barang (fob)", f"{s['exp']:.1f}",      "Miliar USD", s["exp"]-b["exp"],             " Miliar USD"),
            ("🔴 Impor Barang (fob)",  f"{s['imp']:.1f}",      "Miliar USD", s["imp"]-b["imp"],             " Miliar USD"),
            ("🟠 Cadangan Devisa",     f"{s['reserves']:.1f}", "Miliar USD", s["reserves"]-b["reserves"],  " Miliar USD"),
            ("🩵 Bulan Impor",         f"{s['bulan_imp']:.1f}","Bulan",      s["bulan_imp"]-b["bulan_imp"]," bln"),
            ("🟡 CA / PDB",            f"{s['capdb']:.2f}%",   "%",          s["capdb"]-b["capdb"],        " pp"),
        ]
        cols = st.columns(6)
        for col, (lbl, val, unit, dv, sfx) in zip(cols, bop_kpis):
            with col:
                st.metric(lbl, val, f"{'+' if dv>=0 else ''}{dv:.2f}{sfx}", delta_color=metric_delta_color(dv))

        c1, c2, c3 = st.columns(3)
        with c1:
            ca_bar_colors = ["rgba(22,163,74,0.7)" if v >= 0 else "rgba(220,38,38,0.7)" for v in R["ca"]["s"]]
            fig = go.Figure([
                bar_trace("Baseline", YL, R["ca"]["b"], C["blue"], opacity=0.35),
                go.Bar(name="Simulasi", x=YL, y=R["ca"]["s"], marker_color=ca_bar_colors, marker_line_width=0),
            ])
            fig.update_layout(**fig_layout("Transaksi Berjalan (Miliar USD)", barmode="group"))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure([
                bar_trace("Exp Baseline", YL, R["exp"]["b"], C["blue"], opacity=0.35),
                bar_trace("Exp Simulasi", YL, R["exp"]["s"], C["teal"], opacity=0.85),
                bar_trace("Imp Baseline", YL, R["imp"]["b"], C["red"],  opacity=0.25),
                bar_trace("Imp Simulasi", YL, R["imp"]["s"], C["red2"], opacity=0.75),
            ])
            fig.update_layout(**fig_layout("Ekspor vs Impor Barang (Miliar USD)", barmode="group"))
            st.plotly_chart(fig, use_container_width=True)

        with c3:
            fig = go.Figure([
                line_base_trace("Baseline", YL, R["reserves"]["b"]),
                line_trace("Simulasi", YL, R["reserves"]["s"], C["orange"], fill="tozeroy", fillcolor="rgba(234,88,12,0.08)"),
            ])
            fig.update_layout(**fig_layout("Cadangan Devisa (Miliar USD)"))
            st.plotly_chart(fig, use_container_width=True)

    with tab_gdp2:
        gdp_kpis = [
            ("🟢 PDB Growth",     f"{s['gdp']:.2f}%",  s["gdp"]-b["gdp"],  " pp"),
            ("🔵 Konsumsi RT",    f"{s['cons']:.2f}%", s["cons"]-b["cons"], ""),
            ("🩵 PMTB/Investasi", f"{s['inv']:.2f}%",  s["inv"]-b["inv"], ""),
            ("🟢 Ekspor B&J",     f"{s['gexp']:.2f}%", s["gexp"]-b["gexp"]," pp"),
            ("🔴 Impor B&J",      f"{s['gimp']:.2f}%", s["gimp"]-b["gimp"]," pp"),
        ]
        cols = st.columns(5)
        for col, (lbl, val, dv, sfx) in zip(cols, gdp_kpis):
            with col:
                delta_str = f"{'+' if dv>=0 else ''}{dv:.2f}{sfx}" if sfx else None
                st.metric(lbl, val, delta_str, delta_color=metric_delta_color(dv))

        c1, c2, c3 = st.columns(3)
        with c1:
            fig = go.Figure([
                line_base_trace("Baseline", YL, R["gdp"]["b"]),
                line_trace("Simulasi", YL, R["gdp"]["s"], C["green"], fill="tozeroy", fillcolor="rgba(22,163,74,0.08)"),
            ])
            fig.update_layout(**fig_layout("Pertumbuhan PDB Riil (%)"))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure([
                bar_trace("Exp B&J Baseline", YL, R["gexp"]["b"], C["blue"], opacity=0.35),
                bar_trace("Exp B&J Simulasi", YL, R["gexp"]["s"], C["teal"], opacity=0.85),
                bar_trace("Imp B&J Baseline", YL, R["gimp"]["b"], C["red"],  opacity=0.25),
                bar_trace("Imp B&J Simulasi", YL, R["gimp"]["s"], C["red2"], opacity=0.75),
            ])
            fig.update_layout(**fig_layout("Ekspor & Impor Riil B&J (%)", barmode="group"))
            st.plotly_chart(fig, use_container_width=True)

        with c3:
            nt_range2 = list(range(12_000, 26_500, 500))
            gdp_sens  = [round(simulate_eksternal_v2(n, oil_v, yr_v, scen_v)[1]["gdp"], 2) for n in nt_range2]
            fig = go.Figure([
                go.Scatter(
                    x=[n / 1000 for n in nt_range2], y=gdp_sens,
                    mode="lines", name="PDB Growth (%)",
                    line=dict(color=C["green"], width=2),
                    fill="tozeroy", fillcolor="rgba(22,163,74,0.08)",
                ),
                dot_trace("Posisi kini", [nt_v / 1000], [round(s["gdp"], 2)]),
            ])
            fig.update_layout(**fig_layout("Sensitivitas PDB vs Nilai Tukar", xaxis_title="NT (ribu Rp)", yaxis_title="PDB Growth (%)"))
            st.plotly_chart(fig, use_container_width=True)

        tx = s["tx"]
        col_nt, col_icp = st.columns(2)
        with col_nt:
            st.markdown("##### 🔵 Jalur Nilai Tukar (NT) → PDB")
            st.dataframe(pd.DataFrame({
                "Komponen":   ["Delta Ekspor Riil via NT", "Delta Impor Riil via NT", "Delta Net PDB via NT"],
                "Nilai (pp)": [round(tx["expNT"], 3), round(tx["impNT"], 3), round(tx["netNT"], 3)],
            }), hide_index=True, use_container_width=True)

        with col_icp:
            st.markdown("##### 🟠 Jalur Harga Minyak (ICP) → PDB")
            st.dataframe(pd.DataFrame({
                "Komponen":   ["Delta Ekspor Riil via ICP", "Delta Impor Riil via ICP", "Delta Net PDB via ICP"],
                "Nilai (pp)": [round(tx["expICP"], 3), round(tx["impICP"], 3), round(tx["netICP"], 3)],
            }), hide_index=True, use_container_width=True)

    with tab_apbn:
        apbn_kpis = [
            ("🔵 Pendapatan Negara", f"Rp {s['rev']:.0f} T",  s["rev"]-b["rev"],       " T"),
            ("🔴 Belanja Negara",    f"Rp {s['bel']:.0f} T",  s["bel"]-b["bel"],       " T"),
            ("🟣 Defisit/Surplus",   f"Rp {s['def']:.0f} T",  s["def"]-b["def"],       " T"),
            ("🟡 Defisit/PDB",       f"{s['defpdb']:.2f}%",   s["defpdb"]-b["defpdb"], " pp"),
            ("🟠 Subsidi Energi",    f"Rp {s['sube']:.0f} T", s["sube"]-b["sube"],     " T"),
        ]
        cols = st.columns(5)
        for col, (lbl, val, dv, sfx) in zip(cols, apbn_kpis):
            with col:
                st.metric(lbl, val, f"{'+' if dv>=0 else ''}{dv:.2f}{sfx}", delta_color=metric_delta_color(dv))

        st.markdown("##### 📊 Posisi Defisit APBN vs Batas 3% PDB")
        limit = 3.0
        col_b, col_s = st.columns(2)
        with col_b:
            alert_b = "🔴" if abs(b["defpdb"]) > limit else "🟢"
            st.markdown(f"**Baseline {alert_b}:** `{b['defpdb']:.2f}% PDB`")
            st.progress(min(abs(b["defpdb"]) / limit, 1.0))
        with col_s:
            alert_s = "🔴" if abs(s["defpdb"]) > limit else "🟢"
            st.markdown(f"**Simulasi {alert_s}:** `{s['defpdb']:.2f}% PDB`")
            st.progress(min(abs(s["defpdb"]) / limit, 1.0))

        c1, c2, c3 = st.columns(3)
        with c1:
            def_colors = ["rgba(22,163,74,0.7)" if v >= 0 else "rgba(220,38,38,0.7)" for v in R["def"]["s"]]
            fig = go.Figure([
                bar_trace("Baseline", YL, R["def"]["b"], C["purple"], opacity=0.35),
                go.Bar(name="Simulasi", x=YL, y=R["def"]["s"], marker_color=def_colors, marker_line_width=0),
            ])
            fig.update_layout(**fig_layout("Defisit APBN (Rp T)", barmode="group"))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure([
                bar_trace("Pendapatan Baseline", YL, R["rev"]["b"], C["blue"], opacity=0.35),
                bar_trace("Pendapatan Simulasi", YL, R["rev"]["s"], C["teal"], opacity=0.85),
                bar_trace("Belanja Baseline",    YL, R["bel"]["b"], C["red"],  opacity=0.25),
                bar_trace("Belanja Simulasi",    YL, R["bel"]["s"], C["red2"], opacity=0.75),
            ])
            fig.update_layout(**fig_layout("Pendapatan vs Belanja (Rp T)", barmode="group"))
            st.plotly_chart(fig, use_container_width=True)

        with c3:
            icp_range = list(range(30, 135, 5))
            def_sens  = [round(simulate_eksternal_v2(nt_v, ic, yr_v, scen_v)[1]["def"], 0) for ic in icp_range]
            fig = go.Figure([
                go.Scatter(
                    x=[f"${ic}" for ic in icp_range], y=def_sens,
                    mode="lines", name="Defisit (Rp T)",
                    line=dict(color=C["purple"], width=2),
                    fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
                ),
                dot_trace("Posisi kini", [f"${oil_v}"], [round(s["def"], 0)]),
            ])
            fig.update_layout(**fig_layout("Sensitivitas Defisit vs ICP", xaxis_title="ICP (USD/bbl)", yaxis_title="Defisit (Rp T)"))
            st.plotly_chart(fig, use_container_width=True)

        ax = s["ax"]
        col_rev, col_bel = st.columns(2)
        with col_rev:
            st.markdown("##### 🔵 Dampak ke Sisi Pendapatan")
            st.dataframe(pd.DataFrame({
                "Komponen": ["PPh Migas (ICP sensitif)", "PNBP SDA Migas (ICP sensitif)", "Bea Keluar (NT + komoditas)", "Delta Total Pendapatan"],
                "Delta (Rp T)": [round(ax["pph"], 2), round(ax["sda"], 2), round(ax["bea"], 2), round(ax["rev"], 2)],
            }), hide_index=True, use_container_width=True)

        with col_bel:
            st.markdown("##### 🔴 Tekanan Sisi Belanja")
            st.dataframe(pd.DataFrame({
                "Komponen": ["Subsidi Energi (ICP naik + NT melemah)", "Bunga Utang Valas (NT melemah)", "Delta Total Belanja"],
                "Delta (Rp T)": [round(ax["sube"], 2), round(ax["bunga"], 2), round(ax["bel"], 2)],
            }), hide_index=True, use_container_width=True)

    with tab_table:
        st.markdown("#### Tabel Lengkap BOP, GDP & APBN — Baseline vs Simulasi")
        table_def = [
            ("I. Transaksi Berjalan",          "ca",        "ca",        2,   "Miliar USD"),
            ("  Neraca Barang",                "tradebal",  "tradebal",  2,   "Miliar USD"),
            ("    Ekspor (fob)",               "exp",       "exp",       1,   "Miliar USD"),
            ("    Impor (fob)",                "imp",       "imp",       1,   "Miliar USD"),
            ("  Neraca Jasa",                  "svcbal",    "svcbal",    2,   "Miliar USD"),
            ("  Pendapatan Primer",            "primbal",   "primbal",   2,   "Miliar USD"),
            ("  Pendapatan Sekunder",          "secbal",    "secbal",    2,   "Miliar USD"),
            ("III. Transaksi Finansial",       "finbal",    "finbal",    2,   "Miliar USD"),
            ("IV. Total BOP",                  "total",     "total",     2,   "Miliar USD"),
            ("Cadangan Devisa",                "reserves",  "reserves",  1,   "Miliar USD"),
            ("  Dalam Bulan Impor",            "bulan_imp", "bulan_imp", 1,   "Bulan"),
            ("CA / PDB (%)",                   "capdb",     "capdb",     2,   "%"),
            ("★ PDB Growth (Riil)",             "gdp",       "gdp",       2,   "%"),
            ("  Konsumsi RT",                  "cons",      "cons",      2,   "%"),
            ("  PMTB/Investasi",               "inv",       "inv",       2,   "%"),
            ("  Ekspor B&J (aktif)",           "gexp",      "gexp",      2,   "%"),
            ("  Impor B&J (aktif)",            "gimp",      "gimp",      2,   "%"),
            ("★ Pendapatan Negara",             "rev",       "rev",       0,   "Rp T"),
            ("  Perpajakan",                   "pajak",     "pajak",     0,   "Rp T"),
            ("  PNBP",                         "pnbp",      "pnbp",      0,   "Rp T"),
            ("★ Belanja Negara",                "bel",       "bel",       0,   "Rp T"),
            ("  Subsidi Energi",               "sube",      "sube",      0,   "Rp T"),
            ("  Bunga Utang",                  "bunga",     "bunga",     0,   "Rp T"),
            ("★ Defisit APBN",                  "def",       "def",       0,   "Rp T"),
            ("  Defisit / PDB",                "defpdb",    "defpdb",    2,   "%"),
            ("  PDB Nominal",                  "pdb",       "pdb",       0,   "Rp T"),
        ]

        rows_tbl = []
        for lbl, bk, sk, dec, unit in table_def:
            bv   = b.get(bk) or 0
            sv   = s.get(sk) or 0
            diff = sv - bv
            pct  = (diff / abs(bv) * 100) if bv != 0 else 0
            sign = "+" if diff >= 0 else ""
            rows_tbl.append({
                "Indikator": lbl,
                "Baseline":  round(bv,   dec),
                "Simulasi":  round(sv,   dec),
                "Delta Abs": f"{sign}{round(diff, dec)}",
                "Delta %":   f"{sign}{pct:.1f}%",
                "Satuan":    unit,
            })

        st.dataframe(pd.DataFrame(rows_tbl), hide_index=True, use_container_width=True, height=720)


# ===============================================================================
# TAB 3: EKONOMI DAERAH
# ===============================================================================
with tab3:
    st.markdown("### 📍 Command Center: Ekonomi Kewilayahan")
    st.info("🚧 Modul analitik data daerah sedang dalam tahap pengembangan.")


# ===============================================================================
# TAB 4: AI EXECUTIVE SUMMARY
# ===============================================================================
with tab4:
    st.markdown("### 🧠 Modul Laporan Eksekutif Utama & Sinkronisasi Multimodul")
    st.caption("Fungsi ini menyatukan indikator jangka pendek dari Modul 1 dan hasil simulasi sensitivitas guncangan global dari Modul 2 menjadi dokumen teknokratis untuk Bapak Menteri.")
    
    signature_super = make_signature("ALL", st.session_state.current_avg, 5.4, st.session_state.heatmap_summary_str, st.session_state.daily_summary_str, f"{st.session_state.nt_val}_{st.session_state.oil_val}_{st.session_state.sim_gdp}")
    editor_key_super = f"super_editor_{signature_super}"
    
    if st.button("🚀 Jalankan Sintesis Laporan Lintas Tab (AI Engine)"):
        genai.configure(api_key=USER_API_KEY)
        with st.spinner("AI Perencana Ahli Utama sedang mensintesis seluruh modul..."):
            try:
                avail_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                selected_model = next((m for m in avail_models if 'flash' in m), avail_models[0] if avail_models else None)
                
                if not selected_model: st.error("Model AI tidak terdeteksi.")
                else:
                    model = genai.GenerativeModel(selected_model)
                    super_prompt = f"""
Anda adalah Perencana Pembangunan Nasional Ahli Utama di Bappenas RI.
Tugas Anda adalah menyusun Catatan Strategis (Policy Brief) komprehensif untuk Menteri PPN/Kepala Bappenas berbasis seluruh data lintas modul di bawah ini.

=======================================================
INPUT DATA MODUL 1 (KONDISI EKSISTING MAKRO)
=======================================================
- Pertumbuhan Proyeksi Jangka Pendek (DFM): {st.session_state.current_avg:.2f}% (Target APBN: 5.4%)
- Sentimen Tracker Riil & Bulanan BPS: {st.session_state.heatmap_summary_str}

=======================================================
INPUT DATA MODUL 2 (RISIKO SENSITIVITAS & FISKAL)
=======================================================
- Skenario Guncangan: Skenario {st.session_state.scen_val.upper()} pada Tahun {st.session_state.yr_val}
- Parameter Parameter Guncangan: Nilai Tukar Rp{st.session_state.nt_val}/USD dan ICP ${st.session_state.oil_val}/bbl
- Dampak Terhadap PDB Sektor Eksternal: Pertumbuhan Ekonomi Terkoreksi menjadi {st.session_state.sim_gdp:.2f}% (Delta vs Baseline: {st.session_state.sim_gdp - st.session_state.base_gdp:.2f} pp)
- Transaksi Berjalan (CA): {st.session_state.sim_ca:.2f} Md USD
- Defisit Fiskal APBN: {st.session_state.sim_defpdb:.2f}% dari PDB

=======================================================
STRUKTUR OUTPUT LAPORAN EKSEKUTIF (MANDATORI RESMI):
=======================================================
1. POSISI STRATEGIS BAPPENAS ATAS PERKEMBANGAN TERKINI
Evaluasi secara tajam kesenjangan antara realisasi makro eksisting di Modul 1 dengan ancaman kejatuhan ekonomi jika parameter guncangan di Modul 2 menetap sepanjang tahun.

2. ANALISIS MATRIKS SENSITIVITAS & TRANSMISI FISKAL-MONETER
Uraikan jalur transmisi mikro bagaimana pelemahan Rupiah dan lonjakan minyak BBM merembes ke inflasi padat karya, melemahkan konsumsi RT, serta membengkakkan defisit fiskal Bappenas hingga mendekati batas aman.

3. DIREKTIF DAN ORKESTRASI LINTAS KEMENTERIAN/LEMBAGA (K/L)
Berikan rekomendasi kebijakan konkret (BUKAN SLOGAN) dengan menyebutkan kementerian spesifik sebagai pelaksana:
- KEMENTERIAN KEUANGAN (Fiskal): Langkah penataan belanja subsidi energi dan realokasi anggaran.
- BANK INDONESIA (Moneter): Kebijakan stabilisasi nilai tukar untuk meredam imported inflation.
- KEMENTERIAN PERINDUSTRIAN & PERDAGANGAN (Sektor Riil/Mikro): Strategi mitigasi bahan baku impor industri padat karya.
- KEMENTERIAN PERTANIAN (Pangan): Stabilisasi harga komoditas pokok nasional di tingkat daerah.
"""
                    response_ai = model.generate_content(super_prompt, generation_config=genai.types.GenerationConfig(temperature=0.7, top_p=0.9))
                    st.session_state.policy_cache[signature_super] = response_ai.text
                    st.session_state[editor_key_super] = response_ai.text
                    st.success("Sintesis Laporan Strategis Selesai!")
                    st.rerun()
            except Exception as e:
                st.error(f"Gagal generate laporan AI: {e}")

    # CONTAINER EDITOR & DOWNOLAD AREA
    if editor_key_super in st.session_state:
        st.markdown("---")
        st.session_state[editor_key_super] = st.text_area("✍️ Ruang Editor Laporan Strategis Kementerian (Dapat Diedit Manual):", value=st.session_state[editor_key_super], height=450)
        
        with st.expander("🔍 Pratinjau Dokumen Cetak Bapak Menteri", expanded=True):
            st.markdown(st.session_state[editor_key_super])
            
        try:
            import markdown
            html_export_content = markdown.markdown(st.session_state[editor_key_super])
            
            final_html_brief = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Executive Brief Sidang Kabinet</title>
                <style>
                    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
                    body {{ font-family: 'Plus Jakarta Sans', sans-serif; color: #1e293b; padding: 40px; background: #fafafa; }}
                    .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 50px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
                    .repub-header {{ border-bottom: 3px solid #002d72; padding-bottom: 20px; margin-bottom: 30px; text-align: center; }}
                    .repub-header h2 {{ color: #002d72; margin: 0; text-transform: uppercase; letter-spacing: 1px; }}
                    h2, h3 {{ color: #1e3a8a; border-left: 5px solid #2563eb; padding-left: 12px; margin-top: 30px; }}
                    p, li {{ font-size: 15px; line-height: 1.8; color: #334155; }}
                    .stamp {{ text-align: right; font-size: 12px; color: #94a3b8; margin-top: 50px; border-top: 1px solid #e2e8f0; padding-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="repub-header">
                        <h2>Kementerian PPN / Bappenas Republik Indonesia</h2>
                        <div style="font-size: 13px; color:#64748b;">DOKUMEN NEGARA &middot; SANGAT RAHASIA</div>
                    </div>
                    {html_export_content}
                    <div class="stamp">Dihasilkan secara otomatis oleh SPKS Bappenas AI Command Center.<br>Tanggal Cetak: {pd.Timestamp.now().strftime('%d %B %Y')}</div>
                </div>
            </body>
            </html>
            """
            
            st.download_button(
                label="📥 Unduh Executive Summary (Siap Cetak / Save as PDF)",
                data=final_html_brief,
                file_name=f"Laporan_Strategis_Bappenas.html",
                mime="text/html",
                type="primary"
            )
            st.caption("💡 *Tips Pejabat:* Setelah mengunduh file .html di atas, buka file tersebut lalu tekan tombol **Ctrl + P** di keyboard komputer Anda, kemudian pilih opsi **'Save as PDF'** untuk menghasilkan file laporan PDF resmi berformat premium.")
        except Exception as e:
            st.warning(f"Gagal menyiapkan bundling dokumen: {e}")
