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
# 0. KONFIGURASI API KEY (SECURE)
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

def make_signature(view, avg, target, monthly_info, daily_info, ext_info):
    raw_str = f"{view}_{avg:.2f}_{target}_{monthly_info}_{daily_info}_{ext_info}"
    return hashlib.md5(raw_str.encode()).hexdigest()

# ==========================================
# 1. SETUP & DESIGN GLOBAL
# ==========================================
st.set_page_config(page_title="Macro AI Command Center", layout="wide", page_icon="🇮🇩", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background: radial-gradient(circle at 10% 20%, rgb(242, 243, 247) 0%, rgb(215, 221, 232) 90.2%); }
    .glass-card {
        background: rgba(255, 255, 255, 0.65);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.7);
        padding: 24px;
        margin-bottom: 24px;
    }
    .card-title { font-size: 13px; color: #444; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .card-value { font-size: 26px; color: #111; font-weight: 800; margin: 4px 0; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 8px; font-size: 11px; font-weight: 700; margin-right: 6px; }
    .badge-green { background: rgba(212, 237, 218, 0.8); color: #155724; }
    .badge-red { background: rgba(248, 215, 218, 0.8); color: #721c24; }
    .badge-neutral { background: rgba(226, 227, 229, 0.8); color: #383d41; }
    h1 { color: #002d72 !important; }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        color: white; padding: 20px; border-radius: 12px;
        margin-bottom: 20px; display: flex; align-items: center; gap: 14px;
    }
    .logo-box {
        background: white; color: #1d4ed8; width: 42px; height: 42px;
        border-radius: 8px; display: flex; align-items: center;
        justify-content: center; font-weight: 800; font-size: 16px; flex-shrink: 0;
    }
    .hdr-title { font-size: 20px; font-weight: 700; }
    .hdr-sub   { font-size: 13px; opacity: 0.8; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div class="logo-box">ID</div>
    <div>
        <div class="hdr-title">🇮🇩 National Economic Command Center — BAPPENAS RI</div>
        <div class="hdr-sub">Sistem Pendukung Keputusan Strategis & Orkestrasi Kebijakan Kementerian/Lembaga</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===============================================================================
# DATA BASELINE SEKTOR EKSTERNAL & FISKAL
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

# --- DATA LOADING FUNGSI ---
df_target, df_triwulan, df_makro, df_hist_gdp = load_data()
df_daily, date_col_daily = load_daily_data()

# --- MODUL SIMULASI SEKTOR EKSTERNAL (SENSITIVITAS TERBALIK) ---
def simulate_eksternal_v2(nt: float, oil: float, year: int, scen: str):
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

    # 🔥 FIX SENSITIVITAS REVISI KOOR: NT Naik (Lemah) & Oil Naik -> Sektor Riil Drop
    dGexp_nt  = EL["gexp_nt"]  * dNT_pct
    dGimp_nt  = EL["gimp_nt"]  * dNT_pct
    dGexp_oil = EL["gexp_oil"] * dOil_pct
    dGimp_oil = EL["gimp_oil"] * dOil_pct

    s["gexp"] = b["gexp"] + dGexp_nt + dGexp_oil
    s["gimp"] = b["gimp"] + dGimp_nt + dGimp_oil
    
    # Efek kontraksi ke sektor riil domestik
    s["cons"] = b["cons"] - (0.04 * abs(dNT_pct)) - (0.02 * abs(dOil_pct)) if dNT_pct > 0 or dOil_pct > 0 else b["cons"]
    s["gov"]  = b["gov"]
    s["inv"]  = b["inv"] - (0.03 * abs(dNT_pct)) - (0.01 * abs(dOil_pct)) if dNT_pct > 0 or dOil_pct > 0 else b["inv"]
    s["gdp"]  = b["gdp"] - (0.03 * dNT_pct) - (0.015 * dOil_pct)

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

    s["tx"] = {"expNT": dGexp_nt, "impNT": dGimp_nt, "netNT": -0.03*dNT_pct, "expICP": dGexp_oil, "impICP": dGimp_oil, "netICP": -0.015*dOil_pct}
    s["ax"] = {"pph": dPPH, "sda": dRevMigas, "bea": dBea, "rev": dRevTotal, "sube": dSubsE, "bunga": dBunga, "bel": dBelTotal}

    return b, s

# --- ENGINE DFM CODES ---
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
        if candidate in pred_means.index: return float(pred_means.loc[candidate, target])
    return np.nan

def calculate_annual_nowcast(pred_means, target_var, cutoff):
    year = cutoff.year
    vals = [get_prediction_value(pred_means, target_var, pd.Period(year=year, quarter=q, freq='Q')) for q in range(1, 5)]
    vals = [v for v in vals if pd.notna(v)]
    return np.mean(vals) if len(vals) == 4 else np.nan

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
            if name in df_m_raw.columns: processed_data[name] = apply_matlab_transformation(df_m_raw[name], row['log'], row['MoM'], row['YoY'], 'M')
        for _, row in info_q[info_q['INCLUDED'] == 1].iterrows():
            name = row['Indicator Code']
            if name in df_q_raw.columns: processed_data[name] = apply_matlab_transformation(df_q_raw[name], row['log'], row['QoQ'], row['YoY'], 'Q')
        data_full = pd.DataFrame(processed_data).replace([np.inf, -np.inf], np.nan).sort_index()
        data_full_resampled = data_full.resample('MS').first()
        target_var = 'RGDP_growth'
        def get_actual_value(ref_period):
            td = ref_period.to_timestamp(how='end').replace(day=1).normalize()
            return data_full.loc[td, target_var] if td in data_full.index else np.nan
        jobs, seen, hari_ini = [], set(), pd.Timestamp.today().normalize()
        for vc in vintage_cols:
            col_name = vc.strftime('%Y-%m-%d 00:00:00') if vc.strftime('%Y-%m-%d 00:00:00') in df_cal.columns else df_cal.columns[2 + vintage_cols.index(vc)]
            rds = pd.to_datetime(df_cal[col_name], errors="coerce").dropna().unique()
            for rd in sorted(rds):
                if 2023 <= rd.year <= 2026 and rd <= hari_ini and (rd, vc) not in seen:
                    seen.add((rd, vc)); jobs.append((rd, vc))
        jobs.sort(key=lambda x: x[0])
        results_table = []
        for avd, vdb in jobs:
            obs_cutoff = vdb.replace(day=1); ref_q = pd.Period(avd, freq='Q')
            v_data = build_ragged_vintage(data_full_resampled, df_cal, indicator_col, vintage_cols, avd, obs_cutoff).dropna(axis=1, how='all')
            end_m = v_data.drop(columns=[target_var], errors='ignore')
            q_freq = "QE" if pd.__version__ >= "2.2.0" else "Q"
            end_q = v_data[[target_var]].resample(q_freq).last() if target_var in v_data.columns else data_full_resampled.loc[data_full_resampled.index <= obs_cutoff, [target_var]].resample(q_freq).last()
            model = DynamicFactorMQ(endog=end_m, endog_quarterly=end_q, k_factors=1, factor_orders=1, idiosyncratic_ar=1, standardize=True)
            res = model.fit(method='em', maxiter=500, tolerance=1e-5, disp=False)
            means = res.get_prediction(end=res.model.nobs + 24).predicted_mean
            results_table.append({'Day Prediction': avd, 'Reference Quarter': ref_q.strftime('%YQ%q'), 'Actual': get_actual_value(ref_q), 'Backcast': get_prediction_value(means, target_var, ref_q - 1), 'Nowcast': get_prediction_value(means, target_var, ref_q), 'Forecast': get_prediction_value(means, target_var, ref_q + 1), '2-step': get_prediction_value(means, target_var, ref_q + 2), '3-step': get_prediction_value(means, target_var, ref_q + 3), 'Annual Nowcast': calculate_annual_nowcast(means, target_var, avd)})
        return pd.DataFrame(results_table)
    except Exception as e:
        return pd.DataFrame()

# =========================================================================
# 🎛️ STRUKTUR NAVIGASI UTAMA 4 TAB DI ATAS
# =========================================================================
tab_makro, tab_eksternal, tab_daerah, tab_ai = st.tabs([
    "📊 Modul 1: Makro Nasional (DFM)", 
    "🌍 Modul 2: Sektor Eksternal & Fiskal", 
    "📍 Modul 3: Ekonomi Daerah (WIP)",
    "🧠 Modul 4: SPKS Executive Brief & Arahan K/L"
])

# WADAH GLOBAL UNTUK KOMUNIKASI DATA ANTAR TAB
monthly_summary_str = "Data tidak tersedia."
daily_summary_str = "Data tidak tersedia."
heatmap_summary_str = "Data tidak tersedia."
current_avg = 5.2
current_target = 5.4
selected_view = "2026"

# =========================================================================
# TAB 1: MAKRO NASIONAL
# =========================================================================
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
            ref_year = int(ref_q_str[:4]); ref_q_num = int(ref_q_str[-1])
            for target_q in [1, 2, 3, 4]:
                distance = (2026 - ref_year) * 4 + (target_q - ref_q_num)
                mapping_kolom = {-1: 'Backcast', 0: 'Nowcast', 1: 'Forecast', 2: '2-step', 3: '3-step'}
                nk = mapping_kolom.get(distance)
                if nk and nk in latest_row and pd.notna(latest_row[nk]): preds_2026.append(float(latest_row[nk]))
                else:
                    fbd = df_full_results[df_full_results['Reference Quarter'] == f"2026Q{target_q}"]
                    if not fbd.empty: preds_2026.append(float(fbd.sort_values('Day Prediction').iloc[-1]['Nowcast']))
                    else: preds_2026.append(np.nan)
            s_preds = pd.Series(preds_2026)
            preds_2026 = s_preds.ffill().bfill().fillna(5.2).tolist()
        else: preds_2026 = [5.1, 5.2, 5.3, 5.4]

        real_2026 = [None, None, None, None]
        df_triwulan['Tahun_str'] = df_triwulan['Tahun'].astype(str).str.strip().str.replace('.0', '', regex=False)
        if '2026' in df_triwulan['Tahun_str'].values:
            row_2026 = df_triwulan[df_triwulan['Tahun_str'] == '2026'].iloc[0]
            for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
                cn = f'Realisasi {q}'
                if cn in row_2026.index:
                    r = row_2026[cn]
                    try: real_2026[i] = float(r) if pd.notna(r) and str(r).strip() != '' else None
                    except: pass
        now_2026 = preds_2026
        header_ui = st.container()
        selected_view = st.radio("Pilih Rentang Waktu Analisis:", ["2026", "2010 - 2026"], horizontal=True, index=0, key="nav_view_makro")
        final_x, final_real, final_now, final_target = [], [], [], []
        current_avg, current_target = 0, t_2026
        x_2026 = ['2026-Q1', '2026-Q2', '2026-Q3', '2026-Q4']
        valid_x_2026_real = [x_2026[i] for i, r in enumerate(real_2026) if r is not None]
        valid_y_2026_real = [r for r in real_2026 if r is not None]

        if selected_view == "2026":
            final_x = ['Q1', 'Q2', 'Q3', 'Q4']; final_real, final_now, final_target = real_2026, now_2026, [t_2026]*4
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
            y_hist = series_hist.values.tolist(); x_2025 = ['2025-Q1', '2025-Q2', '2025-Q3', '2025-Q4']
            full_x_real = x_hist + x_2025 + valid_x_2026_real; full_y_real = y_hist + combined_2025 + valid_y_2026_real
            if valid_y_2026_real:
                last_real_x = valid_x_2026_real[-1]; last_real_y = valid_y_2026_real[-1]; sisa_q = len(valid_y_2026_real) 
                full_x_proj = [last_real_x] + x_2026[sisa_q:]; full_y_proj = [last_real_y] + preds_2026[sisa_q:]
            else:
                full_x_proj = [x_2025[-1]] + x_2026; full_y_proj = [combined_2025[-1]] + preds_2026
            current_avg = np.mean(preds_2026)

        title_text = f"Outlook Ekonomi: {selected_view}"
        with header_ui:
            st.markdown(f"### {title_text}")
            realisasi_bps_ctc = 5.61 
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Target Acuan APBN", f"{current_target}%")
            if realisasi_bps_ctc is not None:
                gap_realisasi = realisasi_bps_ctc - current_target
                c2.metric("Realisasi BPS (c-t-c)", f"{realisasi_bps_ctc:.2f}%", delta=f"{gap_realisasi:.2f}%")
                c2.caption("Capaian Triwulan I-2026")
            gap_proyeksi = current_avg - current_target
            c3.metric("Proyeksi DFM (Avg)", f"{current_avg:.2f}%", delta=f"{gap_proyeksi:.2f}%")
            gap_status = realisasi_bps_ctc - current_target
            c4.metric("Status Capaian", "✅ SESUAI TARGET" if gap_status >= -0.1 else "❌ BELOW TARGET", delta_color="normal" if gap_status >= -0.1 else "inverse")

        # RENDER UTAMA PLOTLY JALUR NASIONAL
        fig = go.Figure()
        if selected_view == "2010 - 2026":
            latest_q_real = valid_x_2026_real[-1].split('-')[-1] if valid_x_2026_real else "Q4 2025"
            fig.add_trace(go.Scatter(x=full_x_real, y=full_y_real, name=f"Realisasi (Q1 2010-{latest_q_real} 2026)", mode='lines', line=dict(color='#f1c40f', width=2.5)))
            fig.add_trace(go.Scatter(x=full_x_proj, y=full_y_proj, name='Proyeksi DFM 2026', mode='lines', line=dict(color='#27ae60', width=2.5, dash='dot')))
        else:
            fig.add_trace(go.Bar(x=final_x, y=final_real, name='Realisasi (BPS)', marker_color='#2980b9', text=[f"{v:.2f}%" if v else "" for v in final_real], textposition='inside', insidetextanchor='middle', textfont=dict(color='white', size=14)))
            fig.add_trace(go.Scatter(x=final_x, y=final_now, name='DFM Nowcasting', mode='lines+markers', line=dict(color='#f39c12', width=4, shape='spline'), text=[f"{v:.2f}%" for v in final_now], textposition='top center'))
            fig.add_trace(go.Scatter(x=final_x, y=final_target, name='Target APBN', mode='lines', line=dict(color='#c0392b', width=3, dash='dash')))
        fig.update_layout(barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1), height=400)
        st.plotly_chart(fig, use_container_width=True)

        # MONITORING BULANAN & HEATMAP TRACKER
        st.markdown("### 🗺️ Tracker Real Sektor & Heatmap Bulanan")
        df_hm = df_makro[df_makro['Tanggal'] >= '2025-01-01'].copy()
        if not df_hm.empty:
            dates_hm = df_hm['Tanggal'].tolist(); x_labels = df_hm['Tanggal'].dt.strftime('%b %Y').tolist()
            z_data, text_data, monthly_summary_list, heatmap_summary_list = [], [], [], []
            indicator_cols = [c for c in df_makro.columns if c != 'Tanggal']
            for col in indicator_cols:
                col_z, col_text = [], []
                for d in dates_hm:
                    curr_row = df_makro[df_makro['Tanggal'] == d]; val = curr_row[col].values[0] if not curr_row.empty else np.nan
                    prev_d = d - pd.DateOffset(years=1); prev_row = df_makro[(df_makro['Tanggal'].dt.year == prev_d.year) & (df_makro['Tanggal'].dt.month == prev_d.month)]; val_prev = prev_row[col].values[0] if not prev_row.empty else np.nan
                    if pd.isna(val) or pd.isna(val_prev): col_z.append(0); col_text.append("-")
                    else:
                        yoy_curr = (val - val_prev) / abs(val_prev) * 100 if val_prev != 0 else 0
                        txt = f"{yoy_curr:+.2f}%"; is_green = yoy_curr >= 0
                        col_z.append(1 if is_green else -1); col_text.append(txt)
                        if d == dates_hm[-1]: heatmap_summary_list.append(f"{col}: {txt} ({'Stabil/Hijau' if is_green else 'Waspada/Merah'})")
                z_data.append(col_z); text_data.append(col_text)
            monthly_summary_str = "Kinerja sektor riil berjalan sesuai tren."
            heatmap_summary_str = " | ".join(heatmap_summary_list)
            
            fig_hm = go.Figure(data=go.Heatmap(z=z_data, x=x_labels, y=indicator_cols, text=text_data, texttemplate="<b>%{text}</b>", colorscale=[[0.0, '#e74c3c'], [0.5, '#ecf0f1'], [1.0, '#2ecc71']], showscale=False, xgap=3, ygap=3))
            fig_hm.update_layout(height=400, margin=dict(l=220, r=20, t=30, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_hm, use_container_width=True)

# =========================================================================
# TAB 2: SEKTOR EKSTERNAL & FISKAL (PLEK KETIPLEK TAMPILAN ASLI REVISI)
# =========================================================================
with tab_eksternal:
    st.markdown("#### 🌍 Simulasi Dampak Sektor Eksternal & Transmisi Fiskal")
    
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    scen_ext = col_ctrl1.radio("Skenario Utama:", ["med", "high"], horizontal=True, format_func=lambda x: "Medium (Med)" if x == "med" else "High Skenario")
    yr_ext = col_ctrl2.select_slider("Horizon Proyeksi:", options=YEARS, value=2026, key="yr_slider_ext")
    preset_ext = col_ctrl3.selectbox("Pilih Preset Simulasi Cepat:", ["-- pilih --", "📊 Base Med Skenario", "📈 Base High Skenario", "📉 Depresiasi Guncangan (NT 19.500)", "🛢 Krisis Minyak Rendah ($40)", "🔥 Booming Minyak Tinggi ($100)", "⚡ Twin Shock (NT 20.000, ICP $100)"])

    D_ext = SCEN[scen_ext]
    nt_default = D_ext["nt"][yr_ext]; icp_default = D_ext["icp"][yr_ext]

    if   "Depresiasi" in preset_ext: nt_init, oil_init = 19_500, icp_default
    elif "Rendah"     in preset_ext: nt_init, oil_init = nt_default, 40
    elif "Tinggi"     in preset_ext: nt_init, oil_init = nt_default, 100
    elif "Twin"       in preset_ext: nt_init, oil_init = 20_000, 100
    else:                        nt_init, oil_init = nt_default, icp_default

    col_inp1, col_inp2 = st.columns(2)
    nt_ext = col_inp1.number_input("Input Parameter Nilai Tukar (Rp/USD):", min_value=10_000, max_value=30_000, value=nt_init, step=50)
    oil_ext = col_inp2.number_input("Input Parameter Harga Minyak ICP (USD/bbl):", min_value=20, max_value=150, value=oil_init, step=1)

    # Jalankan Simulasi Inti 1:1 Sesuai Dokumen Resmi Sektor Eksternal
    b_ext, s_ext = simulate_eksternal(nt_ext, oil_ext, yr_ext, scen_ext)
    
    # RENDER KPI MATRIKS SEKTOR EKSTERNAL
    st.markdown("<br>", unsafe_allow_html=True)
    kpi_ext_cols = st.columns(6)
    kpis_ext_data = [
        ("🔵 Transaksi Berjalan", f"{s_ext['ca']:.2f}", s_ext["ca"]-b_ext["ca"], " Md USD"),
        ("🟢 Ekspor Barang (fob)", f"{s_ext['exp']:.1f}", s_ext["exp"]-b_ext["exp"], " Md USD"),
        ("🔴 Impor Barang (fob)", f"{s_ext['imp']:.1f}", s_ext["imp"]-b_ext["imp"], " Md USD"),
        ("🟠 Cadangan Devisa", f"{s_ext['reserves']:.1f}", s_ext["reserves"]-b_ext["reserves"], " Md USD"),
        ("🩵 Kecukupan Impor", f"{s_ext['bulan_imp']:.1f}", s_ext["bulan_imp"]-b_ext["bulan_imp"], " Bln"),
        ("🟡 Rasio CA/PDB", f"{s_ext['capdb']:.2f}%", s_ext["capdb"]-b_ext["capdb"], " pp")
    ]
    for idx, (lbl, val, dv, sfx) in enumerate(kpis_ext_data):
        with kpi_ext_cols[idx]: st.metric(lbl, val, f"{'+' if dv>=0 else ''}{dv:.2f}{sfx}", delta_color=metric_delta_color(dv))

    # TREN GRAFIK SEKTOR EKSTERNAL
    st.markdown("<br>", unsafe_allow_html=True)
    g_cols = st.columns(3)
    keys_sim = ["ca", "exp", "imp", "reserves", "gdp", "gexp", "gimp", "def"]
    R_ext = {k: {"b": [], "s": []} for k in keys_sim}
    for y in YEARS:
        bb, ss = simulate_eksternal(nt_ext, oil_ext, y, scen_ext)
        for k in keys_sim:
            R_ext[k]["b"].append(round(bb[k], 2)); R_ext[k]["s"].append(round(ss[k], 2))

    with g_cols[0]:
        fig_ca = go.Figure([bar_trace("Baseline", YL, R_ext["ca"]["b"], C["blue"], opacity=0.35), go.Bar(name="Simulasi", x=YL, y=R_ext["ca"]["s"], marker_color=["rgba(22,163,74,0.7)" if v >= 0 else "rgba(220,38,38,0.7)" for v in R_ext["ca"]["s"]])])
        fig_ca.update_layout(**fig_layout("Transaksi Berjalan (Miliar USD)", barmode="group"))
        st.plotly_chart(fig_ca, use_container_width=True)
    with g_cols[1]:
        fig_gdp = go.Figure([line_base_trace("Baseline", YL, R_ext["gdp"]["b"]), line_trace("Simulasi", YL, R_ext["gdp"]["s"], C["green"], fill="tozeroy", fillcolor="rgba(22,163,74,0.08)")])
        fig_gdp.update_layout(**fig_layout("Pertumbuhan PDB Riil (%) — Konsekuensi Transmisi"))
        st.plotly_chart(fig_gdp, use_container_width=True)
    with g_cols[2]:
        fig_def = go.Figure([bar_trace("Baseline", YL, R_ext["def"]["b"], C["purple"], opacity=0.35), go.Bar(name="Simulasi", x=YL, y=R_ext["def"]["s"], marker_color=["rgba(22,163,74,0.7)" if v >= 0 else "rgba(220,38,38,0.7)" for v in R_ext["def"]["s"]])])
        fig_def.update_layout(**fig_layout("Defisit Fiskal APBN (Rp Triliun)", barmode="group"))
        st.plotly_chart(fig_def, use_container_width=True)

    # SENSITIVITAS & TRANSMISI JALUR APBN
    st.markdown("##### 🏛️ Detail Mekanisme Transmisi Fiskal Domestik")
    col_ax1, col_ax2 = st.columns(2)
    with col_ax1:
        st.markdown("###### Sisi Pendapatan Negara (Rp Triliun)")
        st.dataframe(pd.DataFrame({"Komponen Fiskal": ["PPh Migas", "PNBP SDA Migas", "Bea Keluar Komoditas", "Total Dampak Sisi Penerimaan"], "Delta Hasil Sim": [round(s_ext["ax"]["pph"], 2), round(s_ext["ax"]["sda"], 2), round(s_ext["ax"]["bea"], 2), round(s_ext["ax"]["rev"], 2)]}), hide_index=True, use_container_width=True)
    with col_ax2:
        st.markdown("###### Sisi Tekanan Belanja Negara (Rp Triliun)")
        st.dataframe(pd.DataFrame({"Komponen Fiskal": ["Subsidi Energi", "Bunga Utang Valas", "Total Dampak Sisi Pengeluaran"], "Delta Hasil Sim": [round(s_ext["ax"]["sube"], 2), round(s_ext["ax"]["bunga"], 2), round(s_ext["ax"]["bel"], 2)]}), hide_index=True, use_container_width=True)

# =========================================================================
# TAB 3: EKONOMI DAERAH (WIP)
# =========================================================================
with tab_daerah:
    st.markdown("### 📍 Modul 3: Command Center Ekonomi Kewilayahan & Provinsi")
    st.info("🚧 Status: Terbuka untuk Alokasi Pengkodingan MRIO / IRIO Wilayah. Menunggu sinkronisasi pemicu data sektoral BPS Regional.")

# =========================================================================
# 📊 TAB 4: NEW EXECUTIVE SUMMARY (PUSAT SINTESIS AI & DOWNOLAD PDF)
# =========================================================================
with tab_ai:
    st.markdown("### 🧠 Modul Laporan Eksekutif Utama & Sinkronisasi Multimodul")
    st.caption("Fungsi ini menyatukan indikator jangka pendek dari Modul 1 dan hasil simulasi sensitivitas guncangan global dari Modul 2 menjadi dokumen teknokratis untuk Bapak Menteri.")
    
    signature_super = make_signature(selected_view, current_avg, current_target, heatmap_summary_str, daily_summary_str, f"{nt_ext}_{oil_ext}_{s_ext['gdp']}")
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
- Pertumbuhan Proyeksi Jangka Pendek (DFM): {current_avg:.2f}% (Target APBN: {current_target}%)
- Sentimen Tracker Riil & Bulanan BPS: {heatmap_summary_str}

=======================================================
INPUT DATA MODUL 2 (RISIKO SENSITIVITAS & FISKAL)
=======================================================
- Skenario Guncangan: Skenario {scen_ext.upper()} pada Tahun {yr_ext}
- Parameter Parameter Guncangan: Nilai Tukar Rp{nt_ext}/USD dan ICP ${oil_ext}/bbl
- Dampak Terhadap PDB Sektor Eksternal: Pertumbuhan Ekonomi Terkoreksi menjadi {s_ext['gdp']:.2f}% (Delta vs Baseline: {s_ext['gdp']-b_ext['gdp']:.2f} pp)
- Transaksi Berjalan (CA): {s_ext['ca']:.2f} Md USD
- Defisit Fiskal APBN: {s_ext['defpdb']:.2f}% dari PDB

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
            
        # PROSES EXPORT PREMIUM KE PRESENTASI HTML / SAVE AS PDF
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
                file_name=f"Laporan_Strategis_Bappenas_{yr_ext}.html",
                mime="text/html",
                type="primary"
            )
            st.caption("💡 *Tips Pejabat:* Setelah mengunduh file .html di atas, buka file tersebut lalu tekan tombol **Ctrl + P** di keyboard komputer Anda, kemudian pilih opsi **'Save as PDF'** untuk menghasilkan file laporan PDF resmi berformat premium.")
        except Exception as e:
            st.warning(f"Gagal menyiapkan bundling dokumen: {e}")
