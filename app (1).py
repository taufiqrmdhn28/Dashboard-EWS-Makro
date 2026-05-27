"""
Dashboard Makroekonomi Indonesia 2026-2029
Streamlit + Plotly -- Multimodul (Nasional & Eksternal)

Jalankan: streamlit run dashboard_makro_id.py
"""

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

# ==========================================
# 0. KONFIGURASI API KEY (SECURE)
# ==========================================
try:
    USER_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    USER_API_KEY = ""

# ==========================================
# SETUP CACHE AI (Biar Abadi)
# ==========================================
CACHE_FILE = "policy_cache.pkl"

if 'policy_cache' not in st.session_state:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            st.session_state.policy_cache = pickle.load(f)
    else:
        st.session_state.policy_cache = {}

# FUNGSI SIGNATURE: Mengingat Data Makro (Lengkap) & Data Harian
def make_signature(view, avg, target, monthly_info, daily_info):
    raw_str = f"{view}_{avg:.2f}_{target}_{monthly_info}_{daily_info}"
    import hashlib
    return hashlib.md5(raw_str.encode()).hexdigest()

# ==========================================
# 1. SETUP & DESIGN (MAKRO NASIONAL)
# ==========================================
st.set_page_config(page_title="Macro AI Command Center", layout="wide", page_icon="🇮🇩", initial_sidebar_state="expanded")

# ===============================================================================
# 2. DATA BASELINE SEKTOR EKSTERNAL (DARI REVISI)
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
    "blue":    "#2563eb",
    "green":   "#16a34a",
    "red":     "#dc2626",
    "red2":    "#b91c1c",
    "amber":   "#d97706",
    "purple":  "#7c3aed",
    "orange":  "#ea580c",
    "orange2": "#c2410c",
    "teal":    "#0891b2",
    "gray":    "#6e7681",
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
    s["cons"] = b["cons"]
    s["gov"]  = b["gov"]
    s["inv"]  = b["inv"]
    s["gdp"]  = b["gdp"] + EL["w_exp"] * (dGexp_nt + dGexp_oil) - EL["w_imp"] * (dGimp_nt + dGimp_oil)

    s["tx"] = {
        "expNT":  dGexp_nt,
        "impNT":  dGimp_nt,
        "netNT":  EL["w_exp"] * dGexp_nt  - EL["w_imp"] * dGimp_nt,
        "expICP": dGexp_oil,
        "impICP": dGimp_oil,
        "netICP": EL["w_exp"] * dGexp_oil - EL["w_imp"] * dGimp_oil,
    }

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

    s["ax"] = {
        "pph":   dPPH,
        "sda":   dRevMigas,
        "bea":   dBea,
        "rev":   dRevTotal,
        "sube":  dSubsE,
        "bunga": dBunga,
        "bel":   dBelTotal,
    }

    return b, s

# --- Helper Chart Sektor Eksternal ---
_BASE_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#f8f9fa",
    font=dict(family="monospace", size=11, color="#4b5563"),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1, font_size=10,
    ),
    margin=dict(l=40, r=20, t=50, b=40),
    height=270,
    xaxis=dict(gridcolor="#e5e7eb", showgrid=True),
    yaxis=dict(gridcolor="#e5e7eb", showgrid=True),
)

def fig_layout(title: str, barmode: str = None, **kwargs) -> dict:
    layout = {**_BASE_LAYOUT, "title": dict(text=title, font=dict(size=13)), **kwargs}
    if barmode:
        layout["barmode"] = barmode
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
# 3. DATA LOADING & DFM ENGINE (MAKRO NASIONAL)
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

df_target, df_triwulan, df_makro, df_hist_gdp = load_data()

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
        st.warning(f"⚠️ Gagal sinkronisasi data Google Sheets. Info Error: {e}")
        return None, None

df_daily, date_col_daily = load_daily_data()

def apply_matlab_transformation(series, j1, j2, j3, freq='M'):
    out = series.copy().astype(float)
    if j1 == 1:
        out = out.mask(out <= 0, np.nan)
        out = 100 * np.log(out)
    if j2 == 1:
        out = out.diff(1)
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
        data_full.index = pd.to_datetime(data_full.index)
        data_full_resampled = data_full.resample('MS').first() 
        target_var = 'RGDP_growth'

        def get_actual_value(ref_period):
            target_date = ref_period.to_timestamp(how='end').replace(day=1).normalize()
            if target_date in data_full.index:
                val = data_full.loc[target_date, target_var]
                return val if pd.notna(val) else np.nan
            return np.nan

        jobs = []
        seen = set()
        hari_ini = pd.Timestamp.today().normalize()
        
        for vc in vintage_cols:
            col_name = vc.strftime('%Y-%m-%d 00:00:00') if vc.strftime('%Y-%m-%d 00:00:00') in df_cal.columns else df_cal.columns[2 + vintage_cols.index(vc)]
            release_dates = pd.to_datetime(df_cal[col_name], errors="coerce").dropna().unique()
            for rd in sorted(release_dates):
                if 2023 <= rd.year <= 2026 and rd <= hari_ini and (rd, vc) not in seen:
                    seen.add((rd, vc))
                    jobs.append((rd, vc)) 
                    
        jobs.sort(key=lambda x: x[0])

        results_table = []
        for actual_v_date, v_date_base in jobs:
            obs_cutoff = v_date_base.replace(day=1)
            ref_q = pd.Period(actual_v_date, freq='Q')
            v_data = build_ragged_vintage(data_full_resampled, df_cal, indicator_col, vintage_cols, actual_v_date, obs_cutoff).dropna(axis=1, how='all')
            end_m = v_data.drop(columns=[target_var], errors='ignore')
            q_freq = "QE" if pd.__version__ >= "2.2.0" else "Q"
            if target_var in v_data.columns:
                end_q = v_data[[target_var]].resample(q_freq).last()
            else:
                end_q = data_full_resampled.loc[data_full_resampled.index <= obs_cutoff, [target_var]].resample(q_freq).last()
            
            model = DynamicFactorMQ(endog=end_m, endog_quarterly=end_q, k_factors=1, factor_orders=1, idiosyncratic_ar=1, standardize=True)
            res = model.fit(method='em', maxiter=500, tolerance=1e-5, disp=False)
            means = res.get_prediction(end=res.model.nobs + 24).predicted_mean
            
            results_table.append({
                'Day Prediction': actual_v_date,
                'Reference Quarter': ref_q.strftime('%YQ%q'),
                'Actual': get_actual_value(ref_q), 
                'Backcast': get_prediction_value(means, target_var, ref_q - 1),
                'Nowcast': get_prediction_value(means, target_var, ref_q),
                'Forecast': get_prediction_value(means, target_var, ref_q + 1),
                '2-step': get_prediction_value(means, target_var, ref_q + 2),
                '3-step': get_prediction_value(means, target_var, ref_q + 3),
                'Annual Nowcast': calculate_annual_nowcast(means, target_var, actual_v_date)
            })
        return pd.DataFrame(results_table)
    except Exception as e:
        st.error(f"Error Replikasi DFM: {e}")
        return pd.DataFrame()


# ==========================================
# 4. SISTEM NAVIGASI DASHBOARD (SIDEBAR GLOBAL)
# ==========================================

st.markdown("""
<style>
[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e1e4e8; }
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
.live-badge {
    margin-left: auto; background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.5);
    padding: 5px 14px; border-radius: 20px;
    font-size: 12px; font-weight: 700; white-space: nowrap;
}
.disclaimer {
    background: #fef3c7; border: 1px solid #d97706; color: #92400e;
    padding: 10px 16px; border-radius: 7px;
    font-size: 13px; font-weight: 600; margin-bottom: 14px;
}
div[data-testid="metric-container"] > div { font-family: monospace; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div class="logo-box">ID</div>
    <div>
        <div class="hdr-title">Dashboard Makroekonomi & Pembangunan RI</div>
        <div class="hdr-sub">Multi-Modul: DFM Nasional &middot; Sektor Eksternal &middot; Kewilayahan</div>
    </div>
    <div class="live-badge">&#11044; SIM AKTIF</div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">&#9888;&#65039; <strong>Disclaimer:</strong>'
    ' Modul Sektor Eksternal dan Daerah masih dalam tahap pengembangan.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🎛️ MENU UTAMA")
    main_menu = st.radio(
        "Pilih Modul Analisis:",
        ["📊 Makro Nasional (DFM)", "🌍 Sektor Eksternal & Fiskal", "📍 Ekonomi Daerah (WIP)"],
        index=0
    )
    
    if main_menu == "🌍 Sektor Eksternal & Fiskal":
        st.divider()
        st.markdown("**BASELINE SKENARIO**")
        scen = st.radio("Skenario", ["med", "high"], horizontal=True, format_func=lambda x: "Med" if x == "med" else "High")
        st.markdown("**TAHUN PROYEKSI**")
        yr = st.select_slider("Tahun", options=YEARS, value=2026)

        D           = SCEN[scen]
        nt_default  = D["nt"][yr]
        icp_default = D["icp"][yr]

        st.divider()
        st.markdown("**PRESET SKENARIO CEPAT**")
        preset = st.selectbox("Pilih preset", [
            "-- pilih --", "📊 Base Med", "📈 Base High", "📉 Depresiasi (NT 19.500)",
            "🛢 Minyak Rendah ($40)", "🔥 Minyak Tinggi ($100)", "⚡ Twin Shock (NT 20.000, ICP $100)",
        ])

        if   "Depresiasi" in preset: nt_init, oil_init = 19_500,       icp_default
        elif "Rendah"     in preset: nt_init, oil_init = nt_default,  40
        elif "Tinggi"     in preset: nt_init, oil_init = nt_default,  100
        elif "Twin"       in preset: nt_init, oil_init = 20_000,      100
        else:                        nt_init, oil_init = nt_default,  icp_default

        st.divider()
        st.markdown("**NILAI TUKAR (Rp/USD)**")
        nt = st.number_input("NT", min_value=10_000, max_value=30_000, value=nt_init, step=50, label_visibility="collapsed")
        st.caption(f"Baseline {scen.upper()} {yr}: Rp{nt_default:,} — nilai naik = depresiasi Rupiah")

        st.markdown("**HARGA MINYAK ICP (USD/bbl)**")
        oil = st.number_input("ICP", min_value=20, max_value=150, value=oil_init, step=1, label_visibility="collapsed")
        st.caption(f"Baseline {scen.upper()} {yr}: ${icp_default} — kenaikan ICP meningkatkan penerimaan migas & subsidi")

        b_sim, s_sim = simulate_eksternal(nt, oil, yr, scen)

        st.divider()
        st.markdown(f"**DELTA VS BASELINE {yr}**")
        delta_rows = [
            ("Neraca Berjalan", s_sim["ca"]       - b_sim["ca"],       " Miliar USD"),
            ("Cadangan Devisa", s_sim["reserves"] - b_sim["reserves"], " Miliar USD"),
            ("PDB Growth",      s_sim["gdp"]      - b_sim["gdp"],      " pp"),
            ("Ekspor Riil",     s_sim["gexp"]     - b_sim["gexp"],     " pp"),
            ("Defisit APBN",    s_sim["def"]      - b_sim["def"],      " T"),
            ("Defisit/PDB",     s_sim["defpdb"]   - b_sim["defpdb"],   " pp"),
        ]
        for lbl, dv, sfx in delta_rows:
            sign  = "+" if dv >= 0 else ""
            color = delta_color(dv)
            st.markdown(f"**{lbl}**: :{color}[{sign}{dv:.2f}{sfx}]")
        st.caption(f"Skenario: {scen.upper()} | NT Rp{nt:,} | ICP ${oil}")


# =========================================================================
# MODUL 1: MAKRO NASIONAL (DFM)
# =========================================================================
if main_menu == "📊 Makro Nasional (DFM)":
    
    st.markdown("""
    <style>
    .glass-card {
        background: rgba(255, 255, 255, 0.65);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.7);
        padding: 24px; margin-bottom: 24px;
    }
    .card-title { font-size: 13px; color: #444; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .card-value { font-size: 26px; color: #111; font-weight: 800; margin: 4px 0; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 8px; font-size: 11px; font-weight: 700; margin-right: 6px; }
    .badge-green { background: rgba(212, 237, 218, 0.8); color: #155724; }
    .badge-red { background: rgba(248, 215, 218, 0.8); color: #721c24; }
    .badge-neutral { background: rgba(226, 227, 229, 0.8); color: #383d41; }
    </style>
    """, unsafe_allow_html=True)
    
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
                    if not fallback_df.empty:
                        preds_2026.append(float(fallback_df.sort_values('Day Prediction').iloc[-1]['Nowcast']))
                    else:
                        preds_2026.append(np.nan)
            
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
        
        if not df_full_results.empty:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_full_results.to_excel(writer, index=False, sheet_name='Nowcast Results')
            st.download_button(
                label="📥 Download Full Nowcast Results (Excel)",
                data=buffer.getvalue(),
                file_name="Replikasi_Final_MATLAB_Elaborated.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 📈 Monitoring Data Harian")
        selected_daily_view = st.radio("Pilih Mode Tampilan Pasar:", ["Data Berjalan", "Data Rata-Rata"], horizontal=True, key="daily_view_toggle")
        
        daily_summary_list, daily_berjalan_list, daily_rata_list = [], [], []
        daily_summary_str = daily_berjalan_str = daily_rata_str = "Data harian tidak tersedia."

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
                
            if daily_summary_list: daily_summary_str = " | ".join(daily_summary_list)
            if daily_berjalan_list: daily_berjalan_str = " | ".join(daily_berjalan_list)
            if daily_rata_list: daily_rata_str = " | ".join(daily_rata_list)
                
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
        monthly_summary_str = "Data bulanan tidak tersedia."
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
            
        if monthly_summary_list: monthly_summary_str = "\n".join(monthly_summary_list)

        st.markdown("### 🗺️ Heatmap Tracker (Tren YoY & Threshold Target)")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        df_hm = df_makro[df_makro['Tanggal'] >= '2025-01-01'].copy()
        heatmap_summary_list = [] 
        heatmap_summary_str = "Data Heatmap tidak tersedia."
        
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
                
            if heatmap_summary_list: heatmap_summary_str = " | ".join(heatmap_summary_list)
                
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
        
        st.markdown("### 🧠 AI Policy Generator")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        signature = make_signature(selected_view, current_avg, current_target, monthly_summary_str, daily_summary_str)
        editor_key = f"editor_{signature}"
        final_policy_text = ""

        if signature in st.session_state.policy_cache:
            if editor_key not in st.session_state:
                st.session_state[editor_key] = st.session_state.policy_cache[signature]
            st.success("✅ Draf tersedia. Silakan lakukan penyesuaian narasi pada kotak di bawah.")

        if signature not in st.session_state.policy_cache:
            if st.button("Generate Kebijakan Strategis (AI)"):
                genai.configure(api_key=USER_API_KEY)
                with st.spinner('AI sedang menganalisis fenomena spesifik dan merancang terobosan kebijakan...'):
                    try:
                        avail = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                        model_name = next((m for m in avail if 'flash' in m), avail[0] if avail else None)

                        if not model_name: st.error("Gagal mendeteksi model. Cek API Key.")
                        else:
                            generation_config = genai.types.GenerationConfig(temperature=0.7, top_p=0.9)
                            model = genai.GenerativeModel(model_name)
                            
                            prompt = f"""
Anda adalah Perencana Pembangunan Nasional Ahli Utama di Bappenas RI. 
Tugas Anda adalah menyusun Catatan Strategis (Executive Summary) mengenai prospek ekonomi makro dan arahan kebijakan spesifik ke depan.

=====================
KONDISI PDB, PERTUMBUHAN & SEKTOR RIIL
=====================
Fokus Indikator: {selected_view}
Target APBN: {current_target}% | Rata-rata Proyeksi DFM: {current_avg:.2f}%
Ringkasan Bulanan (Sektor Riil): {monthly_summary_str}
Status Momentum: {heatmap_summary_str}
Volatilitas Harian: {daily_summary_str}

=====================
TUGAS KHUSUS (WAJIB DIBACA SEBELUM MENULIS):
=====================
1. FOKUS PADA ANOMALI DATA TERBARU: Jangan berikan ringkasan umum. Cari indikator dari data di atas yang pergerakannya paling ekstrem/mengkhawatirkan, lalu jadikan itu sebagai FOKUS UTAMA perumusan masalah.
2. IDENTIFIKASI ROOT CAUSE BERBASIS BERITA NYATA: Kaitkan anomali data tersebut dengan pemicu faktual nyata di dunia saat ini (Misal: konflik Timur Tengah terbaru, *supply chain shock* komoditas tertentu, rilis suku bunga The Fed terbaru, atau kebijakan fiskal domestik terkini). 
3. ANTI-KLISE (DILARANG MENGGUNAKAN TEMPLATE): Saya MELARANG KERAS Anda menggunakan judul kebijakan yang terlalu umum seperti "Hilirisasi", "Transformasi Digital", atau "Ekonomi Hijau" KECUALI jika data di atas secara langsung dan eksplisit menunjukkan masalah di sektor tersebut. Buatlah penamaan kebijakan yang spesifik sesuai pemicu masalahnya.

=====================
STRUKTUR OUTPUT DOKUMEN:
=====================
Bagian Utama: ARAH KEBIJAKAN DAN STRATEGI MITIGASI (FOKUS PROBLEM-SOLVING)
Sajikan 5 Rekomendasi Kebijakan (2 Stabilisasi Pendek, 2 Solusi Struktural, 1 Inovatif Ekstrem) yang didesain HANYA untuk merespons pemicu spesifik (root cause) dari angka-angka di atas.

Format untuk masing-masing kebijakan:
- Arah Kebijakan: (Judul kebijakan harus spesifik, tajam, dan bukan slogan klise).
- Taktik Mitigasi: (Penjelasan teknokratis BAGAIMANA cara meredam root cause tersebut secara langsung agar *capital outflow*, inflasi, atau pelemahan daya beli segera berhenti).
- Referensi Dasar: [Nomor]. Dasar/Rilis Lembaga Resmi (Contoh: IMF/The Fed/World Bank/BI) terkait *root cause* tersebut.

---
Bagian Bawah: LAMPIRAN ANALISIS TEKNIS 
(Berikan 2 analisis teknis mendalam)
- 1. Diagnosa Fenomena Utama: (Jelaskan peristiwa global/domestik nyata apa yang sedang "mengendalikan" angka-angka di atas).
- 2. Titik Lemah Transmisi Sektor Riil: (Jelaskan dari mana dampak krisis akan merambat, misal: "Lonjakan Dolar merambat ke biaya impor bahan baku industri, berujung pada ancaman PHK sektor padat karya").
"""
                            res = model.generate_content(prompt, generation_config=generation_config)
                            st.session_state.policy_cache[signature] = res.text
                            with open(CACHE_FILE, "wb") as f: pickle.dump(st.session_state.policy_cache, f)
                            st.session_state[editor_key] = res.text
                            st.success(f"Analisis Selesai (Engine: {model_name} - Mode Dinamis)")
                            st.rerun()

                    except Exception as e: st.error(f"Error AI: {e}")

        if editor_key in st.session_state:
            st.markdown("---")
            st.session_state[editor_key] = st.text_area(
                "✍️ Ruang Editor Laporan:", value=st.session_state[editor_key], height=500,
                help="Anda bisa mengubah, menambah, atau menghapus narasi AI di sini sebelum laporan difinalisasi."
            )
            with st.expander("🔍 Pratinjau Hasil Akhir Laporan", expanded=True):
                st.markdown(st.session_state[editor_key])
            final_policy_text = st.session_state[editor_key]

        st.markdown('</div>', unsafe_allow_html=True)

        if final_policy_text:
            st.markdown("<br><hr style='border:1px dashed #ccc;'><br>", unsafe_allow_html=True)
            st.markdown("#### 📑 Export Executive Brief")
            st.caption("Download laporan berformat presentasi eksekutif (HTML Interaktif). Bisa di-Save as PDF saat dibuka.")
            try:
                import markdown
                import re
                import copy
                fig_export = copy.deepcopy(fig) if 'fig' in locals() else go.Figure()
                for trace in fig_export.data:
                    trace_type = getattr(trace, 'type', 'scatter')
                    if trace_type == 'scatter':
                        if trace.name and "Proyeksi" in trace.name:
                            jml_titik = len(trace.x) if trace.x is not None else len(trace.y)
                            pola_posisi = ['bottom center', 'top center'] * (jml_titik // 2 + 1)
                            trace.textposition = pola_posisi[:jml_titik]
                            trace.textfont = dict(size=9, color='#065f46', weight='bold')
                        elif trace.name and "Realisasi" in trace.name:
                            jml_titik = len(trace.x) if trace.x is not None else len(trace.y)
                            pola_posisi = ['top center'] * jml_titik
                            if jml_titik > 0: pola_posisi[-1] = 'top left'
                            trace.textposition = pola_posisi
                            trace.textfont = dict(size=9, color='#92400e', weight='bold')

                fig_export.update_layout(margin=dict(t=60, b=60, l=30, r=80))
                chart_html = fig_export.to_html(full_html=False, include_plotlyjs='cdn', default_height='450px')
                
                html_policy = markdown.markdown(final_policy_text)
                html_policy = html_policy.replace("<ul>", "<ul class='premium-list'>")
                html_policy = html_policy.replace("<li>", "<li>")
                html_policy = html_policy.replace("<h3>", "<h3 class='policy-title'>✨ ")
                html_policy = html_policy.replace("<strong>", "<strong class='highlight-text'>")
                
                def parse_to_html_list(data_str, is_market=False):
                    if not data_str: return "<p>Data tidak tersedia.</p>"
                    clean_data = data_str.replace('\n', ' | ')
                    html_list = "<ul class='data-list'>"
                    for item in clean_data.split(' | '):
                        item_clean = item.strip()
                        if item_clean and "tidak tersedia" not in item_clean.lower():
                            if is_market: html_list += f"<li><span class='bullet-blue'></span> {item_clean}</li>"
                            else:
                                if "-" in item_clean: html_list += f"<li><span class='badge-red'>▼</span> {item_clean}</li>"
                                else: html_list += f"<li><span class='badge-blue'>▲</span> {item_clean}</li>"
                    html_list += "</ul>"
                    return html_list if "<li" in html_list else "<p>Data tidak tersedia.</p>"

                html_monthly = parse_to_html_list(monthly_summary_str, False)
                db_str = locals().get('daily_berjalan_str', daily_summary_str)
                dr_str = locals().get('daily_rata_str', daily_summary_str)
                html_daily_berjalan = parse_to_html_list(db_str, True)
                html_daily_rata = parse_to_html_list(dr_str, True)

                html_template = f"""
                <!DOCTYPE html>
                <html lang="id">
                <head>
                    <meta charset="UTF-8">
                    <title>Brief: Macroeconomic Update RI</title>
                    <style>
                        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;700;800&display=swap');
                        body {{ font-family: 'Plus Jakarta Sans', sans-serif; background-color: #cbd5e1; color: #334155; padding: 50px 20px; line-height: 1.6; margin: 0; }}
                        .report-container {{ max-width: 1100px; margin: 0 auto; background: #ffffff; border-radius: 20px; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.3); overflow: hidden; }}
                        .header {{ background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); padding: 50px 70px; color: white; }}
                        .header h1 {{ font-size: 40px; font-weight: 800; margin: 0 0 10px 0; color: #ffffff; letter-spacing: -0.5px; line-height: 1.2; }}
                        .header p {{ font-size: 16px; color: #94a3b8; margin: 0; letter-spacing: 0.5px; }}
                        .content-body {{ padding: 40px 70px 60px 70px; }}
                        .section-header {{ display: flex; align-items: center; justify-content: space-between; margin: 40px 0 20px 0; }}
                        .section-label {{ font-size: 22px; font-weight: 800; color: #0f172a; display: flex; align-items: center; gap: 12px; margin: 0; }}
                        .section-label span {{ background: #eff6ff; border: 1px solid #bfdbfe; padding: 8px 12px; border-radius: 10px; font-size: 18px; }}
                        .tab-container {{ display: flex; gap: 10px; background: #f8fafc; padding: 6px; border-radius: 12px; border: 1px solid #e2e8f0; }}
                        .tab-btn {{ background: transparent; border: none; padding: 8px 16px; border-radius: 8px; cursor: pointer; font-weight: 700; font-family: inherit; color: #64748b; transition: 0.3s; font-size: 14px; }}
                        .tab-btn:hover {{ background: #e2e8f0; color: #0f172a; }}
                        .tab-btn.active {{ background: #3b82f6; color: white; box-shadow: 0 2px 4px rgba(59,130,246,0.3); }}
                        .chart-wrapper {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.03); margin-bottom: 50px; }}
                        .data-list {{ list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
                        .data-list li {{ background: #ffffff; padding: 14px 16px; border-radius: 10px; border: 1px solid #e2e8f0; font-size: 13.5px; color: #334155; display: flex; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02); font-weight: 600; margin: 0; }}
                        .bullet-blue {{ display: inline-block; width: 10px; height: 10px; background: #3b82f6; border-radius: 50%; flex-shrink: 0; margin-right: 10px; }}
                        .badge-blue {{ background: #dbeafe; color: #1e40af; padding: 4px 8px; border-radius: 6px; font-weight: 800; font-size: 11px; flex-shrink: 0; }}
                        .badge-red {{ background: #fee2e2; color: #991b1b; padding: 4px 8px; border-radius: 6px; font-weight: 800; font-size: 11px; flex-shrink: 0; }}
                        .ai-box {{ background: linear-gradient(145deg, #f8fafc, #eff6ff); border: 1px solid #bfdbfe; border-radius: 20px; padding: 40px; margin-top: 60px; position: relative; }}
                        .ai-box::before {{ content:''; position: absolute; top:0; left:0; width:100%; height:6px; background: linear-gradient(90deg, #2563eb, #9333ea); border-radius: 20px 20px 0 0; }}
                        .policy-title {{ color: #1e3a8a; font-size: 20px; font-weight: 800; border-bottom: 2px dashed #cbd5e1; padding-bottom: 12px; margin-top: 35px; margin-bottom: 20px; }}
                        .premium-list {{ list-style: none; padding: 0; margin: 0; }}
                        .premium-list li {{ background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #3b82f6; padding: 25px 30px; border-radius: 12px; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.03); font-size: 15.5px; color: #1e293b; line-height: 1.8; }}
                        .highlight-text {{ color: #2563eb; font-weight: 800; }}
                        .footer {{ text-align: center; padding: 30px; margin-top: 50px; color: #94a3b8; font-size: 13px; border-top: 1px solid #e2e8f0; }}
                    </style>
                    <script>
                        function openTabPH(evt, tabName) {{
                            var i, tabcontent, tablinks;
                            tabcontent = document.getElementsByClassName("tab-content-ph");
                            for (i = 0; i < tabcontent.length; i++) {{ tabcontent[i].style.display = "none"; }}
                            tablinks = document.getElementsByClassName("tab-btn-ph");
                            for (i = 0; i < tablinks.length; i++) {{ tablinks[i].className = tablinks[i].className.replace(" active", ""); }}
                            document.getElementById(tabName).style.display = "block";
                            evt.currentTarget.className += " active";
                        }}
                    </script>
                </head>
                <body>
                    <div class="report-container">
                        <div class="header">
                            <h1>Macroeconomic Brief</h1>
                            <p>Analisis Perkembangan Ekonomi Makro Bappenas RI</p>
                        </div>
                        <div class="content-body">
                            <div class="section-header"><div class="section-label"><span>📈</span> Proyeksi Pertumbuhan Ekonomi (DFM)</div></div>
                            <div class="chart-wrapper">{chart_html}</div>
                            <div class="section-header"><div class="section-label"><span>🏢</span> Kinerja Seluruh Sektor Riil</div></div>
                            {html_monthly}
                            <div class="section-header">
                                <div class="section-label"><span>⚡</span> Volatilitas Pasar Harian</div>
                                <div class="tab-container">
                                    <button class="tab-btn tab-btn-ph active" onclick="openTabPH(event, 'ph-berjalan')">Data Berjalan</button>
                                    <button class="tab-btn tab-btn-ph" onclick="openTabPH(event, 'ph-rata')">Rata-rata</button>
                                </div>
                            </div>
                            <div id="ph-berjalan" class="tab-content-ph" style="display: block;">{html_daily_berjalan}</div>
                            <div id="ph-rata" class="tab-content-ph" style="display: none;">{html_daily_rata}</div>
                            <div class="ai-box">
                                <div class="section-label" style="margin-top: 0; margin-bottom: 25px; border:none; padding:0;"><span>🧠</span> Rekomendasi Kebijakan</div>
                                {html_policy}
                            </div>
                            <div class="footer">Dokumen ini dihasilkan oleh model AI.<br>Dicetak pada: <strong>{pd.Timestamp.now().strftime('%d %B %Y %H:%M')} WIB</strong></div>
                        </div>
                    </div>
                </body>
                </html>
                """
                st.download_button(label="📥 Download Laporan Eksekutif (.html)", data=html_template, file_name="Laporan_Brief_Bappenas.html", mime="text/html", type="primary")
            except Exception as e:
                st.warning(f"Gagal menyiapkan dokumen HTML. Error detail: {e}")


# =========================================================================
# MODUL 2: SEKTOR EKSTERNAL & FISKAL
# =========================================================================
elif main_menu == "🌍 Sektor Eksternal & Fiskal":
    
    st.markdown("### 🌍 Dashboard Sensitivitas Eksternal & Fiskal RI")
    st.info(
        "📌 **Mekanisme Transmisi Utama:** "
        "Depresiasi Rupiah & Kenaikan Harga Minyak menekan daya beli (Konsumsi) dan biaya modal (Investasi). "
        "Pada APBN, hal tersebut memicu lonjakan Subsidi Energi dan beban Bunga Utang."
    )

    keys = ["ca", "exp", "imp", "reserves", "gdp", "gexp", "gimp", "def", "rev", "bel", "sube", "bunga", "pajak"]
    R = {k: {"b": [], "s": []} for k in keys}

    for y in YEARS:
        bb, ss = simulate_eksternal(nt, oil, y, scen)
        for k in keys:
            R[k]["b"].append(round(bb[k], 2))
            R[k]["s"].append(round(ss[k], 2))

    YL = [str(y) for y in YEARS]

    tab_bop, tab_gdp, tab_apbn, tab_table = st.tabs([
        "📊 Neraca Pembayaran (BOP)",
        "📈 Pertumbuhan Ekonomi (GDP)",
        "🏛 Defisit APBN",
        "📋 Tabel Lengkap",
    ])

    with tab_bop:
        bop_kpis = [
            ("🔵 Transaksi Berjalan",  f"{s_sim['ca']:.2f}",       "Miliar USD", s_sim["ca"]-b_sim["ca"],              " Miliar USD"),
            ("🟢 Ekspor Barang (fob)", f"{s_sim['exp']:.1f}",      "Miliar USD", s_sim["exp"]-b_sim["exp"],             " Miliar USD"),
            ("🔴 Impor Barang (fob)",  f"{s_sim['imp']:.1f}",      "Miliar USD", s_sim["imp"]-b_sim["imp"],             " Miliar USD"),
            ("🟠 Cadangan Devisa",     f"{s_sim['reserves']:.1f}", "Miliar USD", s_sim["reserves"]-b_sim["reserves"],  " Miliar USD"),
            ("🩵 Bulan Impor",         f"{s_sim['bulan_imp']:.1f}","Bulan",      s_sim["bulan_imp"]-b_sim["bulan_imp"]," bln"),
            ("🟡 CA / PDB",            f"{s_sim['capdb']:.2f}%",   "%",          s_sim["capdb"]-b_sim["capdb"],        " pp"),
        ]
        cols = st.columns(6)
        for col, (lbl, val, unit, dv, sfx) in zip(cols, bop_kpis):
            with col: st.metric(lbl, val, f"{'+' if dv>=0 else ''}{dv:.2f}{sfx}", delta_color=metric_delta_color(dv))

        c1, c2, c3 = st.columns(3)
        with c1:
            ca_bar_colors = ["rgba(22,163,74,0.7)" if v >= 0 else "rgba(220,38,38,0.7)" for v in R["ca"]["s"]]
            fig = go.Figure([bar_trace("Baseline", YL, R["ca"]["b"], C["blue"], opacity=0.35), go.Bar(name="Simulasi", x=YL, y=R["ca"]["s"], marker_color=ca_bar_colors, marker_line_width=0)])
            fig.update_layout(**fig_layout("Transaksi Berjalan (Miliar USD)", barmode="group"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure([bar_trace("Exp Baseline", YL, R["exp"]["b"], C["blue"], opacity=0.35), bar_trace("Exp Simulasi", YL, R["exp"]["s"], C["teal"], opacity=0.85), bar_trace("Imp Baseline", YL, R["imp"]["b"], C["red"],  opacity=0.25), bar_trace("Imp Simulasi", YL, R["imp"]["s"], C["red2"], opacity=0.75)])
            fig.update_layout(**fig_layout("Ekspor vs Impor Barang (Miliar USD)", barmode="group"))
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            fig = go.Figure([line_base_trace("Baseline", YL, R["reserves"]["b"]), line_trace("Simulasi", YL, R["reserves"]["s"], C["orange"], fill="tozeroy", fillcolor="rgba(234,88,12,0.08)")])
            fig.update_layout(**fig_layout("Cadangan Devisa (Miliar USD)"))
            st.plotly_chart(fig, use_container_width=True)

        c4, c5 = st.columns(2)
        with c4:
            tb_b   = [SCEN[scen]["tradebal"][y] for y in YEARS]
            svc_b  = [SCEN[scen]["svcbal"][y]   for y in YEARS]
            prim_b = [SCEN[scen]["primbal"][y]   for y in YEARS]
            sec_b  = [SCEN[scen]["secbal"][y]    for y in YEARS]
            fig = go.Figure([bar_trace("N. Barang", YL, tb_b, C["blue"], opacity=0.75), bar_trace("N. Jasa", YL, svc_b, C["amber"], opacity=0.75), bar_trace("Pend. Primer", YL, prim_b, C["red"], opacity=0.75), bar_trace("Pend. Sekunder", YL, sec_b, C["green"], opacity=0.75)])
            fig.update_layout(**fig_layout("Komponen Neraca Berjalan (Miliar USD)", barmode="group"))
            st.plotly_chart(fig, use_container_width=True)
        with c5:
            nt_range = list(range(12_000, 26_500, 500))
            ca_sens  = [round(simulate_eksternal(n, oil, yr, scen)[1]["ca"], 2) for n in nt_range]
            fig = go.Figure([go.Scatter(x=[n / 1000 for n in nt_range], y=ca_sens, mode="lines", name="CA (Miliar USD)", line=dict(color=C["blue"], width=2), fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"), dot_trace("Posisi kini", [nt / 1000], [round(s_sim["ca"], 2)])])
            fig.update_layout(**fig_layout("Sensitivitas CA vs Nilai Tukar", xaxis_title="NT (ribu Rp)", yaxis_title="CA (Miliar USD)"))
            st.plotly_chart(fig, use_container_width=True)


    with tab_gdp:
        gdp_kpis = [
            ("🟢 PDB Growth",     f"{s_sim['gdp']:.2f}%",  s_sim["gdp"]-b_sim["gdp"],  " pp"),
            ("🔵 Konsumsi RT",    f"{s_sim['cons']:.2f}%", 0.0,                ""),
            ("🩵 PMTB/Investasi", f"{s_sim['inv']:.2f}%",  0.0,                ""),
            ("🟢 Ekspor B&J",     f"{s_sim['gexp']:.2f}%", s_sim["gexp"]-b_sim["gexp"]," pp"),
            ("🔴 Impor B&J",      f"{s_sim['gimp']:.2f}%", s_sim["gimp"]-b_sim["gimp"]," pp"),
        ]
        cols = st.columns(5)
        for col, (lbl, val, dv, sfx) in zip(cols, gdp_kpis):
            with col:
                delta_str = f"{'+' if dv>=0 else ''}{dv:.2f}{sfx}" if sfx else None
                st.metric(lbl, val, delta_str, delta_color=metric_delta_color(dv))

        c1, c2, c3 = st.columns(3)
        with c1:
            fig = go.Figure([line_base_trace("Baseline", YL, R["gdp"]["b"]), line_trace("Simulasi", YL, R["gdp"]["s"], C["green"], fill="tozeroy", fillcolor="rgba(22,163,74,0.08)")])
            fig.update_layout(**fig_layout("Pertumbuhan PDB Riil (%)"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure([bar_trace("Exp B&J Baseline", YL, R["gexp"]["b"], C["blue"], opacity=0.35), bar_trace("Exp B&J Simulasi", YL, R["gexp"]["s"], C["teal"], opacity=0.85), bar_trace("Imp B&J Baseline", YL, R["gimp"]["b"], C["red"],  opacity=0.25), bar_trace("Imp B&J Simulasi", YL, R["gimp"]["s"], C["red2"], opacity=0.75)])
            fig.update_layout(**fig_layout("Ekspor & Impor Riil B&J (%)", barmode="group"))
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            gdp_sens  = [round(simulate_eksternal(n, oil, yr, scen)[1]["gdp"], 2) for n in nt_range]
            fig = go.Figure([go.Scatter(x=[n / 1000 for n in nt_range], y=gdp_sens, mode="lines", name="PDB Growth (%)", line=dict(color=C["green"], width=2), fill="tozeroy", fillcolor="rgba(22,163,74,0.08)"), dot_trace("Posisi kini", [nt / 1000], [round(s_sim["gdp"], 2)])])
            fig.update_layout(**fig_layout("Sensitivitas PDB vs Nilai Tukar", xaxis_title="NT (ribu Rp)", yaxis_title="PDB Growth (%)"))
            st.plotly_chart(fig, use_container_width=True)

        tx = s_sim["tx"]
        col_nt, col_icp = st.columns(2)
        with col_nt:
            st.markdown("##### 🔵 Jalur Nilai Tukar (NT) → Net Ekspor → PDB")
            st.dataframe(pd.DataFrame({"Komponen": ["Delta Ekspor Riil via NT", "Delta Impor Riil via NT", "Delta PDB via NT"], "Nilai (pp)": [round(tx["expNT"], 3), round(tx["impNT"], 3), round(tx["netNT"], 3)]}), hide_index=True, use_container_width=True)
        with col_icp:
            st.markdown("##### 🟠 Jalur Harga Minyak (ICP) → Net Ekspor → PDB")
            st.dataframe(pd.DataFrame({"Komponen": ["Delta Ekspor Riil via ICP", "Delta Impor Riil via ICP", "Delta PDB via ICP"], "Nilai (pp)": [round(tx["expICP"], 3), round(tx["impICP"], 3), round(tx["netICP"], 3)]}), hide_index=True, use_container_width=True)


    with tab_apbn:
        apbn_kpis = [
            ("🔵 Pendapatan Negara", f"Rp {s_sim['rev']:.0f} T",  s_sim["rev"]-b_sim["rev"],       " T"),
            ("🔴 Belanja Negara",    f"Rp {s_sim['bel']:.0f} T",  s_sim["bel"]-b_sim["bel"],       " T"),
            ("🟣 Defisit/Surplus",   f"Rp {s_sim['def']:.0f} T",  s_sim["def"]-b_sim["def"],       " T"),
            ("🟡 Defisit/PDB",       f"{s_sim['defpdb']:.2f}%",   s_sim["defpdb"]-b_sim["defpdb"], " pp"),
            ("🟠 Subsidi Energi",    f"Rp {s_sim['sube']:.0f} T", s_sim["sube"]-b_sim["sube"],     " T"),
        ]
        cols = st.columns(5)
        for col, (lbl, val, dv, sfx) in zip(cols, apbn_kpis):
            with col: st.metric(lbl, val, f"{'+' if dv>=0 else ''}{dv:.2f}{sfx}", delta_color=metric_delta_color(dv))

        st.markdown("##### 📊 Posisi Defisit APBN vs Batas 3% PDB")
        col_b, col_s = st.columns(2)
        with col_b:
            st.markdown(f"**Baseline {'🔴' if abs(b_sim['defpdb']) > 3.0 else '🟢'}:** `{b_sim['defpdb']:.2f}% PDB`")
            st.progress(min(abs(b_sim["defpdb"]) / 3.0, 1.0))
        with col_s:
            st.markdown(f"**Simulasi {'🔴' if abs(s_sim['defpdb']) > 3.0 else '🟢'}:** `{s_sim['defpdb']:.2f}% PDB`")
            st.progress(min(abs(s_sim["defpdb"]) / 3.0, 1.0))

        c1, c2, c3 = st.columns(3)
        with c1:
            def_colors = ["rgba(22,163,74,0.7)" if v >= 0 else "rgba(220,38,38,0.7)" for v in R["def"]["s"]]
            fig = go.Figure([bar_trace("Baseline", YL, R["def"]["b"], C["purple"], opacity=0.35), go.Bar(name="Simulasi", x=YL, y=R["def"]["s"], marker_color=def_colors, marker_line_width=0)])
            fig.update_layout(**fig_layout("Defisit APBN (Rp T)", barmode="group"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure([bar_trace("Pendapatan Baseline", YL, R["rev"]["b"], C["blue"], opacity=0.35), bar_trace("Pendapatan Simulasi", YL, R["rev"]["s"], C["teal"], opacity=0.85), bar_trace("Belanja Baseline", YL, R["bel"]["b"], C["red"],  opacity=0.25), bar_trace("Belanja Simulasi", YL, R["bel"]["s"], C["red2"], opacity=0.75)])
            fig.update_layout(**fig_layout("Pendapatan vs Belanja (Rp T)", barmode="group"))
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            icp_range = list(range(30, 135, 5))
            def_sens  = [round(simulate_eksternal(nt, ic, yr, scen)[1]["def"], 0) for ic in icp_range]
            fig = go.Figure([go.Scatter(x=[f"${ic}" for ic in icp_range], y=def_sens, mode="lines", name="Defisit (Rp T)", line=dict(color=C["purple"], width=2), fill="tozeroy", fillcolor="rgba(124,58,237,0.08)"), dot_trace("Posisi kini", [f"${oil}"], [round(s_sim["def"], 0)])])
            fig.update_layout(**fig_layout("Sensitivitas Defisit vs ICP", xaxis_title="ICP (USD/bbl)", yaxis_title="Defisit (Rp T)"))
            st.plotly_chart(fig, use_container_width=True)

        c4, c5 = st.columns(2)
        with c4:
            fig = go.Figure([bar_trace("Subsidi Energi Baseline", YL, R["sube"]["b"],  C["red"], opacity=0.35), bar_trace("Subsidi Energi Simulasi", YL, R["sube"]["s"],  C["red2"], opacity=0.85), bar_trace("Bunga Utang Baseline", YL, R["bunga"]["b"], C["orange"], opacity=0.35), bar_trace("Bunga Utang Simulasi", YL, R["bunga"]["s"], C["orange2"], opacity=0.85)])
            fig.update_layout(**fig_layout("Komponen Belanja Utama (Rp T)", barmode="group"))
            st.plotly_chart(fig, use_container_width=True)
        with c5:
            fig = go.Figure([bar_trace("Pajak Baseline", YL, R["pajak"]["b"], C["blue"], opacity=0.35), bar_trace("Pajak Simulasi", YL, R["pajak"]["s"], C["teal"], opacity=0.85)])
            fig.update_layout(**fig_layout("Komponen Penerimaan Pajak (Rp T)", barmode="group"))
            st.plotly_chart(fig, use_container_width=True)


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
            ("  Konsumsi RT (hardkey)",        "cons",      "cons",      2,   "%"),
            ("  PMTB/Investasi (hardkey)",     "inv",       "inv",       2,   "%"),
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
            bv   = b_sim.get(bk) or 0
            sv   = s_sim.get(sk) or 0
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


# =========================================================================
# MODUL 3: EKONOMI DAERAH (WIP)
# =========================================================================
elif main_menu == "📍 Ekonomi Daerah (WIP)":
    st.markdown("### 📍 Command Center: Ekonomi Kewilayahan")
    st.info("🚧 Modul analitik data daerah sedang dalam tahap pengkodingan dan pengembangan lanjutan oleh tim Data Science Bappenas. Silakan kembali lagi nanti.")
