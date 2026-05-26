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
# 1. SETUP & DESIGN
# ==========================================
st.set_page_config(page_title="Macro AI Command Center", layout="wide", page_icon="🇮🇩")

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
</style>
""", unsafe_allow_html=True)

st.title("🇮🇩 National Economic Command Center")
st.markdown("##### Engine: Holt-Winters (Econometric Forecasting) | Standard: Official Statistics")

# ==========================================
# 2. DATA LOADING
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

# ==========================================
# 2.5. DATA LOADING (GOOGLE SHEETS HARIAN)
# ==========================================
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

# ==========================================
# 2.8 ENGINE SEKTOR EKSTERNAL & FISKAL
# ==========================================
SCEN = {
    'med': {
        'nt': {2026:16700, 2027:16500, 2028:16300, 2029:16200},
        'icp': {2026:65, 2027:65, 2028:65, 2029:65},
        'ca': {2026:-4.512, 2027:-19.096, 2028:30.655, 2029:3.750},
        'tradebal': {2026:47.226, 2027:39.341, 2028:47.772, 2029:23.016},
        'exp': {2026:302.146, 2027:343.369, 2028:367.146, 2029:381.850},
        'imp': {2026:-254.920, 2027:-304.028, 2028:-319.374, 2029:-358.834},
        'svcbal': {2026:-20.581, 2027:-18.602, 2028:-8.188, 2029:-10.202},
        'primbal': {2026:-39.004, 2027:-49.990, 2028:-14.082, 2029:-14.907},
        'secbal': {2026:7.847, 2027:10.155, 2028:5.153, 2029:5.842},
        'capbal': {2026:0.352, 2027:0.352, 2028:0.352, 2029:0.352},
        'finbal': {2026:10.882, 2027:25.073, 2028:19.737, 2029:6.940},
        'total': {2026:6.721, 2027:6.329, 2028:50.744, 2029:11.042},
        'reserves': {2026:160.997, 2027:165.131, 2028:213.679, 2029:222.526},
        'bulan_imp': {2026:6.008, 2027:5.160, 2028:6.810, 2029:6.333},
        'capdb': {2026:-0.272, 2027:-1.059, 2028:1.499, 2029:0.163},
        'gdpnom_usd': {2026:1659.5, 2027:1803.9, 2028:2044.6, 2029:2299.6},
        'gdp': {2026:5.4, 2027:5.9, 2028:7.7, 2029:8.0},
        'cons': {2026:5.105, 2027:8.254, 2028:7.996, 2029:8.008},
        'gov': {2026:7.854, 2027:28.496, 2028:7.099, 2029:16.055},
        'inv': {2026:7.059, 2027:9.841, 2028:10.324, 2029:11.149},
        'gexp': {2026:17.310, 2027:5.132, 2028:-4.789, 2029:1.538},
        'gimp': {2026:8.461, 2027:13.907, 2028:-4.232, 2029:7.928},
        'rev': {2026:3223.6, 2027:3332.2, 2028:5015.6, 2029:6286.5},
        'bel': {2026:3936.8, 2027:3816.0, 2028:5526.9, 2029:6396.4},
        'def': {2026:-713.1, 2027:-483.8, 2028:-511.3, 2029:-109.9},
        'defpdb': {2026:-2.765, 2027:-1.715, 2028:-1.635, 2029:-0.315},
        'sube': {2026:213.0, 2027:186.4, 2028:197.8, 2029:234.9},
        'bunga': {2026:599.2, 2027:622.5, 2028:785.7, 2029:785.7},
        'pajak': {2026:2725.7, 2027:2825.1, 2028:4311.0, 2029:5466.6},
        'pnbp': {2026:497.3, 2027:506.6, 2028:699.0, 2029:813.6},
        'migas': {2026:114.0, 2027:131.7, 2028:167.8, 2029:194.8},
        'pdb': {2026:25788.9, 2027:28211.7, 2028:31283.1, 2029:34938.3},
    },
    'high': {
        'nt': {2026:16700, 2027:16500, 2028:16300, 2029:16200},
        'icp': {2026:65, 2027:75, 2028:75, 2029:75},
        'ca': {2026:-4.512, 2027:-6.802, 2028:18.853, 2029:10.329},
        'tradebal': {2026:47.226, 2027:46.285, 2028:63.627, 2029:51.137},
        'exp': {2026:302.146, 2027:341.304, 2028:397.482, 2029:427.061},
        'imp': {2026:-254.920, 2027:-295.019, 2028:-333.856, 2029:-375.924},
        'svcbal': {2026:-19.363, 2027:-19.060, 2028:-13.579, 2029:-13.500},
        'primbal': {2026:-39.719, 2027:-42.193, 2028:-37.201, 2029:-35.580},
        'secbal': {2026:7.266, 2027:8.421, 2028:7.725, 2029:8.239},
        'capbal': {2026:0.352, 2027:0.352, 2028:0.352, 2029:0.352},
        'finbal': {2026:8.358, 2027:13.112, 2028:16.630, 2029:14.697},
        'total': {2026:4.198, 2027:6.662, 2028:35.835, 2029:25.378},
        'reserves': {2026:172.208, 2027:186.145, 2028:219.162, 2029:208.666},
        'bulan_imp': {2026:6.4, 2027:5.8, 2028:7.0, 2029:6.6},
        'capdb': {2026:0.602, 2027:0.825, 2028:1.151, 2029:0.563},
        'gdpnom_usd': {2026:1532.5, 2027:1695.6, 2028:1982.1, 2029:2247.0},
        'gdp': {2026:5.4, 2027:7.5, 2028:7.7, 2029:8.0},
        'cons': {2026:5.105, 2027:5.727, 2028:8.512, 2029:8.525},
        'gov': {2026:7.796, 2027:7.807, 2028:9.557, 2029:10.901},
        'inv': {2026:5.340, 2027:8.845, 2028:11.601, 2029:12.410},
        'gexp': {2026:17.310, 2027:8.779, 2028:-0.811, 2029:3.454},
        'gimp': {2026:6.321, 2027:9.746, 2028:8.635, 2029:8.000},
        'rev': {2026:3225.4, 2027:3569.4, 2028:5047.0, 2029:6417.0},
        'bel': {2026:3938.4, 2027:4210.9, 2028:4977.3, 2029:6314.1},
        'def': {2026:-713.1, 2027:-641.5, 2028:69.7, 2029:102.9},
        'defpdb': {2026:-2.765, 2027:-2.226, 2028:0.219, 2029:0.290},
        'sube': {2026:284.5, 2027:192.8, 2028:195.4, 2029:194.1},
        'bunga': {2026:599.1, 2027:634.4, 2028:808.3, 2029:947.1},
        'pajak': {2026:2725.0, 2027:3025.6, 2028:4356.1, 2029:5558.6},
        'pnbp': {2026:499.7, 2027:542.9, 2028:685.2, 2029:852.0},
        'migas': {2026:150.5, 2027:136.0, 2028:146.1, 2029:222.7},
        'pdb': {2026:25788.9, 2027:28824.4, 2028:31810.0, 2029:35524.1},
    }
}

EL = {
    'bop_exp_nt': 0.15, 'bop_imp_nt': -0.25,
    'bop_exp_oil': 0.80, 'bop_imp_oil': 0.95,
    'bop_svc_nt': 0.05, 'bop_prim_nt': -0.03,
    'share_exp_migas': 0.06, 'share_imp_migas': 0.16,
    'gexp_nt': 0.08, 'gimp_nt': -0.06,
    'gexp_oil': 0.025, 'gimp_oil': 0.018,
    'w_exp': 0.27, 'w_imp': 0.22,
    'sube_oil': 0.95, 'sube_nt': -0.15,
    'bunga_nt': 0.006,
}

def simulate_eksternal(nt, oil, year, scen='med'):
    D = SCEN[scen]
    b = {k: D[k].get(year, 0) for k in D.keys()}
    
    dNT = (nt - b['nt']) / b['nt'] if b['nt'] else 0
    dOil = (oil - b['icp']) / b['icp'] if b['icp'] else 0
    dNT_pct, dOil_pct = dNT * 100, dOil * 100
    
    s = {}
    exp_migas = b['exp'] * EL['share_exp_migas']
    exp_non = b['exp'] * (1 - EL['share_exp_migas'])
    imp_migas = b['imp'] * EL['share_imp_migas']
    imp_non = b['imp'] * (1 - EL['share_imp_migas'])
    
    s['exp'] = exp_migas * (1 + EL['bop_exp_oil'] * dOil) + exp_non * (1 + EL['bop_exp_nt'] * dNT)
    s['imp'] = imp_migas * (1 + EL['bop_imp_oil'] * dOil) + imp_non * (1 + EL['bop_imp_nt'] * dNT)
    s['tradebal'] = s['exp'] + s['imp']
    s['svcbal'] = b['svcbal'] * (1 + EL['bop_svc_nt'] * dNT)
    s['primbal'] = b['primbal'] * (1 + EL['bop_prim_nt'] * dNT)
    s['secbal'] = b['secbal']
    s['ca'] = s['tradebal'] + s['svcbal'] + s['primbal'] + s['secbal']
    s['total'] = s['ca'] + b['capbal'] + b['finbal']
    s['reserves'] = b['reserves'] + (s['total'] - b['total'])
    
    base_import_monthly = abs(b['imp']) / 12
    sim_import_monthly = abs(s['imp']) / 12
    if b.get('bulan_imp'):
        adj_factor = b['bulan_imp'] / (b['reserves'] / base_import_monthly) if base_import_monthly else 1
        s['bulan_imp'] = (s['reserves'] / sim_import_monthly) * adj_factor if sim_import_monthly else 0
    else:
        s['bulan_imp'] = s['reserves'] / sim_import_monthly if sim_import_monthly else 0
        
    dGexp_nt = EL['gexp_nt'] * dNT_pct
    dGimp_nt = EL['gimp_nt'] * dNT_pct
    dGexp_oil = EL['gexp_oil'] * dOil_pct
    dGimp_oil = EL['gimp_oil'] * dOil_pct
    
    s['gexp'] = b['gexp'] + dGexp_nt + dGexp_oil
    s['gimp'] = b['gimp'] + dGimp_nt + dGimp_oil
    dGdp = EL['w_exp'] * (dGexp_nt + dGexp_oil) - EL['w_imp'] * (dGimp_nt + dGimp_oil)
    s['gdp'] = b['gdp'] + dGdp
    
    dRevMigas = b['migas'] * EL['bop_exp_oil'] * dOil
    dPPH = 55.6 * 0.7 * dOil
    dBea = b['pajak'] * 0.019 * (dOil + dNT * 0.3)
    dRevTotal = dRevMigas + dPPH + dBea
    dSubsE = b['sube'] * (EL['sube_oil'] * dOil + EL['sube_nt'] * dNT)
    dBunga = b['bunga'] * EL['bunga_nt'] * dNT
    dBelTotal = dSubsE + dBunga
    
    s['rev'] = b['rev'] + dRevTotal
    s['bel'] = b['bel'] + dBelTotal
    s['def'] = s['rev'] - s['bel']
    s['defpdb'] = (s['def'] / b['pdb']) * 100 if b['pdb'] else 0
    s['sube'] = b['sube'] + dSubsE
    
    return b, s

# ==========================================
# 3. ENGINE DFM NOWCASTING (TARGET 4 KUARTAL 2026)
# ==========================================
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
        # 1. Load Data
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

        # --- FUNGSI HELPER UNTUK MENGAMBIL NILAI ACTUAL (Excel Only) ---
        def get_actual_value(ref_period):
            # Berdasarkan file INO, data kuartalan ada di bln 3, 6, 9, 12 tanggal 1
            # Contoh: 2023Q1 -> target 2023-03-01
            target_date = ref_period.to_timestamp(how='end').replace(day=1).normalize()
            if target_date in data_full.index:
                val = data_full.loc[target_date, target_var]
                return val if pd.notna(val) else np.nan
            return np.nan

        # 2. Kumpulkan Jadwal Rilis (2023 - 2026)
        jobs = []
        seen = set()
        
        # a. Tentukan Lasteval: Tanggal persis saat dashboard dibuka
        hari_ini = pd.Timestamp.today().normalize()
        
        for vc in vintage_cols:
            col_name = vc.strftime('%Y-%m-%d 00:00:00') if vc.strftime('%Y-%m-%d 00:00:00') in df_cal.columns else df_cal.columns[2 + vintage_cols.index(vc)]
            
            # b. Ambil semua jadwal rilis dari kolom Kalender
            release_dates = pd.to_datetime(df_cal[col_name], errors="coerce").dropna().unique()
            
            for rd in sorted(release_dates):
                # c. LOGIKA PENGEREMAN (Sesuai Maksud Min):
                # Hanya simpan jadwal rilis (rd) yang TANGGALNYA KURANG DARI ATAU SAMA DENGAN HARI INI
                if 2023 <= rd.year <= 2026 and rd <= hari_ini and (rd, vc) not in seen:
                    seen.add((rd, vc))
                    jobs.append((rd, vc)) # Masukkan ke antrean proses MATLAB
                    
        jobs.sort(key=lambda x: x[0])

        # 3. Eksekusi Iterasi
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
                'Actual': get_actual_value(ref_q), # <--- KOLOM ACTUAL MASUK DISINI
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
# 3.5 SISTEM NAVIGASI DASHBOARD (SIDEBAR)
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ MENU UTAMA")
    main_menu = st.radio(
        "Pilih Modul Analisis:",
        ["📊 Makro Nasional (DFM)", "🌍 Sektor Eksternal & Fiskal", "📍 Ekonomi Daerah (WIP)"],
        index=0
    )
    
    # Jika Menu Eksternal dipilih, munculkan parameter simulasi di sidebar
    if main_menu == "🌍 Sektor Eksternal & Fiskal":
        st.markdown("---")
        st.markdown("**⚙️ Parameter Simulasi Eksternal**")
        scen_val = st.selectbox("Skenario Baseline:", ["med", "high"], format_func=lambda x: "Medium (Med)" if x == "med" else "High")
        year_val = st.selectbox("Tahun Proyeksi:", [2026, 2027, 2028, 2029])
        nt_val = st.number_input("Nilai Tukar (Rp/USD):", value=16700, step=50)
        oil_val = st.number_input("Harga Minyak ICP (USD/bbl):", value=65, step=1)

# ==========================================
# 4. EXECUTION DASHBOARD PDB
# ==========================================
if main_menu == "📊 Makro Nasional (DFM)":
    
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
        
        # EKSEKUSI DFM
        df_full_results = run_full_dfm_replication()
        
        if not df_full_results.empty:
            preds_2026 = []
            latest_row = df_full_results.sort_values('Day Prediction').iloc[-1]
            ref_q_str = latest_row['Reference Quarter'] 
            ref_year = int(ref_q_str[:4])
            ref_q_num = int(ref_q_str[-1])
            
            for target_q in [1, 2, 3, 4]:
                distance = (2026 - ref_year) * 4 + (target_q - ref_q_num)
                mapping_kolom = {
                    -1: 'Backcast',
                    0:  'Nowcast',
                    1:  'Forecast',
                    2:  '2-step',
                    3:  '3-step'
                }
                
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

        # =======================================================
        # BACA DATA REALISASI DARI EXCEL (ROBUST & KEBAL ERROR)
        # =======================================================
        real_2026 = [None, None, None, None]
        df_triwulan['Tahun_str'] = df_triwulan['Tahun'].astype(str).str.strip().str.replace('.0', '', regex=False)
        
        if '2026' in df_triwulan['Tahun_str'].values:
            row_2026 = df_triwulan[df_triwulan['Tahun_str'] == '2026'].iloc[0]
            for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
                col_name = f'Realisasi {q}'
                if col_name in row_2026.index:
                    r = row_2026[col_name]
                    try:
                        real_2026[i] = float(r) if pd.notna(r) and str(r).strip() != '' else None
                    except:
                        pass

        now_2026 = preds_2026

        # =======================================================
        # JURUS UI: MENYIAPKAN WADAH ATAS UNTUK JUDUL & METRIK
        # =======================================================
        header_ui = st.container()

        selected_view = st.radio(
            "Pilih Rentang Waktu Analisis:",
            ["2026", "2010 - 2026"],
            horizontal=True,
            index=0
        )

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

        # =======================================================
        # MENGISI WADAH ATAS DENGAN JUDUL DAN ANGKA
        # =======================================================
        title_text = f"Outlook Ekonomi: {selected_view}"
        if selected_view == "2026": title_text += " (Model: Dynamic Factor MQ)"
        else: title_text = "Historis & Proyeksi Ekonomi (DFM Model)"

        with header_ui:
            st.markdown(f"### {title_text}")
            
            # 🔥 INPUT MANUAL C-T-C (Sudah disesuaikan ke 5.61) 🔥
            realisasi_bps_ctc = 5.61 
            
            c1, c2, c3, c4 = st.columns(4)
            
            # 1. KOTAK TARGET
            c1.metric("Target Acuan", f"{current_target}%")
            
            # 2. KOTAK REALISASI BPS (MANUAL + KETERANGAN PERIODE)
            if realisasi_bps_ctc is not None:
                gap_realisasi = realisasi_bps_ctc - current_target
                c2.metric("Realisasi BPS (c-t-c)", f"{realisasi_bps_ctc:.2f}%", delta=f"{gap_realisasi:.2f}%")
                c2.caption("Capaian Triwulan I-2026") # 🔥 KETERANGAN TAMBAHAN DI BAWAH KOTAK
            else:
                c2.metric("Realisasi BPS (c-t-c)", "Belum Rilis", delta="-", delta_color="off")
                
            # 3. KOTAK PROYEKSI DFM (Avg)
            gap_proyeksi = current_avg - current_target
            c3.metric("Proyeksi DFM (Avg)", f"{current_avg:.2f}%", delta=f"{gap_proyeksi:.2f}%")
            
            # 4. KOTAK STATUS CAPAIAN
            angka_acuan_status = realisasi_bps_ctc if realisasi_bps_ctc is not None else current_avg
            gap_status = angka_acuan_status - current_target
            status = "✅ SESUAI TARGET" if gap_status >= -0.1 else "❌ BELOW TARGET"
            c4.metric("Status Capaian", status, delta_color="normal" if gap_status >= -0.1 else "inverse")

        # =======================================================
        # MEMBANGUN GRAFIK DI BAWAH TOMBOL
        # =======================================================
        fig = go.Figure()

        if selected_view == "2010 - 2026":
            latest_q_real = valid_x_2026_real[-1].split('-')[-1] if valid_x_2026_real else "Q4 2025"
            legend_realisasi = f"Realisasi (Q1 2010-{latest_q_real} 2026)"

            fig.add_trace(go.Scatter(x=full_x_real, y=full_y_real, name=legend_realisasi, mode='lines', line=dict(color='#f1c40f', width=2.5)))
            fig.add_trace(go.Scatter(x=full_x_proj, y=full_y_proj, name='Proyeksi DFM 2026', mode='lines', line=dict(color='#27ae60', width=2.5, dash='dot')))
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1), height=450)
        else:
            fig.add_trace(go.Bar(
                x=final_x, 
                y=final_real, 
                name='Realisasi (BPS)', 
                marker_color='#2980b9', 
                text=[f"{v:.2f}%" if v else "" for v in final_real], 
                textposition='inside', 
                insidetextanchor='middle', 
                textfont=dict(color='white', size=14)
            ))
            fig.add_trace(go.Scatter(x=final_x, y=final_now, name='DFM Nowcasting', mode='lines+markers', line=dict(color='#f39c12', width=4, shape='spline'), text=[f"{v:.2f}%" for v in final_now], textposition='top center'))
            fig.add_trace(go.Scatter(x=final_x, y=final_target, name='Target APBN', mode='lines', line=dict(color='#c0392b', width=3, dash='dash')))
            fig.update_layout(barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1), height=450)

        # =======================================================
        # SUNTIKAN MAGIC PLOTLY: MUNCULKAN ANGKA ATAS-BAWAH 
        # =======================================================
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

        # ==========================================
        # --- MONITORING DATA HARIAN (DTD & YTD) ---
        # ==========================================
        st.markdown("### 📈 Monitoring Data Harian")
        
        selected_daily_view = st.radio(
            "Pilih Mode Tampilan Pasar:",
            ["Data Berjalan", "Data Rata-Rata"],
            horizontal=True,
            key="daily_view_toggle"
        )
        
        daily_summary_list = []  
        daily_berjalan_list = [] 
        daily_rata_list = []     

        daily_summary_str = "Data harian tidak tersedia."
        daily_berjalan_str = "Data harian tidak tersedia."
        daily_rata_str = "Data harian tidak tersedia."

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
                else:
                    ytd = 0; ytd_str = "YTD: -"
                
                current_year_data = valid_series[valid_series[date_col_daily].dt.year == current_year]
                avg_current = current_year_data[col].mean() if not current_year_data.empty else val
                avg_prev = prev_year_data[col].mean() if not prev_year_data.empty else 0
                
                if avg_prev != 0:
                    avg_growth = ((avg_current - avg_prev) / avg_prev) * 100  
                else:
                    avg_growth = 0

                disp_val_b = f"{val:,.2f}" if val > 10 else f"{val:.2f}"
                daily_berjalan_list.append(f"{col}: {disp_val_b} (DTD: {dtd:+.2f}%, {ytd_str})")
                
                disp_val_r = f"{avg_current:,.2f}" if avg_current > 10 else f"{avg_current:.2f}"
                daily_rata_list.append(f"{col}: Avg {current_year} = {disp_val_r} (Perubahan vs Avg 2025: {avg_growth:+.2f}%)")

                if "Berjalan" in selected_daily_view:
                    disp_val = disp_val_b
                    color_1 = "badge-red" if dtd < 0 else "badge-green"
                    color_2 = "badge-red" if ytd < 0 else "badge-green"
                    badge_1_str = f"DTD: {dtd:+.2f}%"
                    badge_2_str = ytd_str
                    subtitle_str = f"Data Spot: {date_str}"
                    daily_summary_list.append(f"{col}: {disp_val_b} (DTD: {dtd:+.2f}%)")
                else:
                    disp_val = disp_val_r
                    color_1 = "badge-neutral" 
                    color_2 = "badge-red" if avg_growth < 0 else "badge-green"
                    avg_prev_disp = f"{avg_prev:,.2f}" if avg_prev > 10 else f"{avg_prev:.2f}"
                    badge_1_str = f"Avg '25: {avg_prev_disp}"
                    badge_2_str = f"Δ {avg_growth:+.2f}%"
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

        # ==========================================
        # --- ATURAN GLOBAL WARNA (WAJIB DI SINI) ---
        # ==========================================
        ATURAN_WARNA = {
            'PMI Manufaktur Negara Berkembang': True, 
            'Jumlah Uang Yang Beredar': True, 
            'Penjualan Mobil': True, 
            'Penjualan semen': True, 
            'Ekspor Barang': True, 
            'Impor Barang Modal': True, 
            'Impor Bahan Baku': True, 
            'Kredit Perbankan': True, 
            'Penjualan Motor': True, 
            'Indeks Keyakinan Konsumen': True, 
            'Impor Barang Konsumsi': True, 
            'Inflasi': False, 
            'Nilai Tukar terhadap Dolar AS': False, 
            'Suku Bunga': False
        }

        # ==========================================
        # --- DEEP DIVE (FIXED YoY & MtM FORMATTING) ---
        # ==========================================
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
            else:
                mtm_diff, mtm_pct = 0, 0

            target_date_yoy = date_obj - pd.DateOffset(years=1)
            row_yoy = df_makro[(df_makro['Tanggal'].dt.year == target_date_yoy.year) & (df_makro['Tanggal'].dt.month == target_date_yoy.month)]
            
            if not row_yoy.empty and pd.notna(row_yoy.iloc[0][col]):
                val_yoy = row_yoy.iloc[0][col]
                yoy_diff = val - val_yoy
                yoy_pct = (yoy_diff / abs(val_yoy)) * 100 if val_yoy != 0 else 0
                has_yoy = True
            else:
                yoy_diff, yoy_pct, has_yoy = 0, 0, False

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

            if "PMI" in col and val < 50: 
                is_bad_mtm, is_bad_yoy = True, True
                
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
            
        if monthly_summary_list:
            monthly_summary_str = "\n".join(monthly_summary_list)

        # ==========================================
        # --- HEATMAP BULANAN ---
        # ==========================================
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
                        col_z.append(0) 
                        col_text.append("-")
                    else:
                        if is_level_indicator:
                            txt = f"{val:,.2f}" if val > 1000 else f"{val:.2f}"
                            diff = val - val_prev
                        else:
                            yoy_curr = (val - val_prev) / abs(val_prev) * 100 if val_prev != 0 else 0
                            txt = f"{yoy_curr:+.2f}%"
                            if pd.isna(val_prev_prev):
                                diff = yoy_curr
                            else:
                                yoy_prev = (val_prev - val_prev_prev) / abs(val_prev_prev) * 100 if val_prev_prev != 0 else 0
                                diff = yoy_curr - yoy_prev
                            
                        is_green = False
                        is_special_indicator = False
                        
                        if "PMI" in col:
                            is_special_indicator = True
                            is_green = val >= 50.0
                        elif "Inflasi" in col:
                            is_special_indicator = True
                            is_green = 1.5 <= val <= 3.5
                        elif "Nilai Tukar" in col:
                            is_special_indicator = True
                            is_green = val <= 16900
                            
                        if is_special_indicator:
                            col_z.append(1 if is_green else -1)
                        else:
                            if diff == 0: 
                                col_z.append(0) 
                            elif rule_naik_bagus: 
                                is_green = diff > 0
                                col_z.append(1 if is_green else -1)
                            else: 
                                is_green = diff < 0
                                col_z.append(1 if is_green else -1)
                            
                        col_text.append(txt)
                        
                        if d == dates_hm[-1]:
                            sentimen = "Positif/Aman (Hijau)" if is_green else "Negatif/Waspada (Merah)"
                            heatmap_summary_list.append(f"{col}: Kondisi {sentimen} ({txt})")
                
                z_data.append(col_z)
                text_data.append(col_text)
                
            if heatmap_summary_list:
                heatmap_summary_str = " | ".join(heatmap_summary_list)
                
            fig_hm = go.Figure(data=go.Heatmap(
                z=z_data, x=x_labels, y=indicator_cols, text=text_data,
                texttemplate="<b>%{text}</b>", 
                textfont=dict(size=14, color='#111'),
                colorscale=[[0.0, '#e74c3c'], [0.5, '#ecf0f1'], [1.0, '#2ecc71']], 
                zmin=-1, zmax=1, showscale=False, xgap=3, ygap=3
            ))
            fig_hm.update_layout(
                height=150 + len(indicator_cols)*35,
                margin=dict(l=220, r=20, t=30, b=20),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(autorange="reversed", tickfont=dict(size=12, color='#333', weight='bold')) 
            )
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
        else:
            st.info("Belum ada data bulanan untuk ditampilkan.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ==========================================
        # --- AI ADVISOR & EDITOR ---
        # ==========================================
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

                        if not model_name: 
                            st.error("Gagal mendeteksi model. Cek API Key.")
                        else:
                            generation_config = genai.types.GenerationConfig(
                                temperature=0.7, 
                                top_p=0.9        
                            )
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
                            with open(CACHE_FILE, "wb") as f: 
                                pickle.dump(st.session_state.policy_cache, f)
                            
                            st.session_state[editor_key] = res.text
                            st.success(f"Analisis Selesai (Engine: {model_name} - Mode Dinamis)")
                            st.rerun()

                    except Exception as e: 
                        st.error(f"Error AI: {e}")

        if editor_key in st.session_state:
            st.markdown("---")
            st.session_state[editor_key] = st.text_area(
                "✍️ Ruang Editor Laporan:",
                value=st.session_state[editor_key],
                height=500,
                help="Anda bisa mengubah, menambah, atau menghapus narasi AI di sini sebelum laporan difinalisasi."
            )
            
            with st.expander("🔍 Pratinjau Hasil Akhir Laporan", expanded=True):
                st.markdown(st.session_state[editor_key])
            
            final_policy_text = st.session_state[editor_key]

        st.markdown('</div>', unsafe_allow_html=True)

        # =========================================================
        # FITUR MAGIC: EXPORT KE EXECUTIVE BRIEF
        # =========================================================
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
                            if jml_titik > 0:
                                pola_posisi[-1] = 'top left'
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
                            if is_market:
                                html_list += f"<li><span class='bullet-blue'></span> {item_clean}</li>"
                            else:
                                if "-" in item_clean:
                                    html_list += f"<li><span class='badge-red'>▼</span> {item_clean}</li>"
                                else:
                                    html_list += f"<li><span class='badge-blue'>▲</span> {item_clean}</li>"
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
                            
                            <div class="section-header">
                                <div class="section-label"><span>📈</span> Proyeksi Pertumbuhan Ekonomi (DFM)</div>
                            </div>
                            <div class="chart-wrapper">
                                {chart_html}
                            </div>

                            <div class="section-header">
                                <div class="section-label"><span>🏢</span> Kinerja Seluruh Sektor Riil</div>
                            </div>
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
                            
                            <div class="footer">
                                Dokumen ini dihasilkan oleh model AI.<br>
                                Dicetak pada: <strong>{pd.Timestamp.now().strftime('%d %B %Y %H:%M')} WIB</strong>
                            </div>
                        </div>

                    </div>
                </body>
                </html>
                """
                
                st.download_button(
                    label="📥 Download Laporan Eksekutif (.html)",
                    data=html_template,
                    file_name="Laporan_Brief_Bappenas.html",
                    mime="text/html",
                    type="primary"
                )
                
            except Exception as e:
                st.warning(f"Gagal menyiapkan dokumen HTML. Error detail: {e}")


# =========================================================================
# PENUTUP BLOK MAKRO NASIONAL
# =========================================================================

# ==========================================
# MODUL 2: SEKTOR EKSTERNAL & FISKAL (SUPER UI/UX DASH REPLICA)
# ==========================================
elif main_menu == "🌍 Sektor Eksternal & Fiskal":
    
    # --- 1. SETUP STATE UNTUK PRESET BUTTON ---
    if 'nt_val' not in st.session_state: st.session_state.nt_val = 16700
    if 'oil_val' not in st.session_state: st.session_state.oil_val = 65
    if 'year_val' not in st.session_state: st.session_state.year_val = 2026
    if 'scen_val' not in st.session_state: st.session_state.scen_val = 'med'

    # --- 2. ENGINE SIMULASI (LOGIKA EKONOMI SUDAH DIPERBAIKI) ---
    def simulate_eksternal_v2(nt, oil, year, scen):
        b = {k: SCEN[scen][k].get(year, 0) for k in SCEN[scen].keys()}
        dNT_pct = (nt - b['nt']) / b['nt'] * 100 if b['nt'] else 0
        dOil_pct = (oil - b['icp']) / b['icp'] * 100 if b['icp'] else 0

        s = b.copy() 
        
        # BOP: Depresiasi -> Ekspor naik dikit, Impor turun
        s['exp'] = b['exp'] * (1 + (0.015 * dOil_pct) + (0.005 * dNT_pct))
        s['imp'] = b['imp'] * (1 + (0.018 * dOil_pct) - (0.008 * dNT_pct)) 
        s['tradebal'] = s['exp'] + s['imp']
        s['ca'] = s['tradebal'] + b['svcbal'] + b['primbal'] + b['secbal']
        s['reserves'] = b['reserves'] + (s['ca'] - b['ca'])
        s['bulan_imp'] = s['reserves'] / (abs(s['imp']) / 12) if s['imp'] != 0 else 0

        # GDP LOGIC FIXED: NT Naik / Oil Naik -> PDB & Konsumsi Turun
        s['cons'] = b['cons'] - (0.10 * dNT_pct) - (0.05 * dOil_pct)
        s['inv'] = b['inv'] - (0.06 * dNT_pct) - (0.03 * dOil_pct)
        s['gexp'] = b['gexp'] + (0.05 * dNT_pct) + (0.02 * dOil_pct)
        s['gimp'] = b['gimp'] - (0.04 * dNT_pct) + (0.01 * dOil_pct)

        # Bobot PDB (Kira-kira): Cons 53%, Inv 30%, Exp 22%, Imp 20%
        d_gdp = (0.53 * (s['cons'] - b['cons'])) + (0.30 * (s['inv'] - b['inv'])) + (0.22 * (s['gexp'] - b['gexp'])) - (0.20 * (s['gimp'] - b['gimp']))
        s['gdp'] = b['gdp'] + d_gdp
        s['pdb'] = b['pdb'] * (1 + (d_gdp/100)) # PDB Nominal bergerak sejalan

        # APBN
        s['rev'] = b['rev'] + (15 * dOil_pct) + (5 * dNT_pct)
        s['bel'] = b['bel'] + (12 * dOil_pct) + (8 * dNT_pct)
        s['def'] = s['rev'] - s['bel']
        s['defpdb'] = (s['def'] / s['pdb']) * 100 if s['pdb'] else 0
        s['sube'] = b['sube'] + (2 * dOil_pct) + (0.5 * dNT_pct)

        d = {k: s[k] - b[k] for k in b.keys()}
        return b, s, d

    # --- 3. SIDEBAR KHUSUS EKSTERNAL ---
    with st.sidebar:
        st.markdown("<hr style='margin-top:0px; margin-bottom:15px;'>", unsafe_allow_html=True)
        st.markdown("<span style='font-size:12px; font-weight:bold; color:#64748b;'>SKENARIO BASELINE</span>", unsafe_allow_html=True)
        scen_col1, scen_col2 = st.columns(2)
        if scen_col1.button("Med", use_container_width=True): st.session_state.scen_val = 'med'
        if scen_col2.button("High", use_container_width=True): st.session_state.scen_val = 'high'
        
        st.info(f"Baseline {st.session_state.scen_val.capitalize()} {st.session_state.year_val}: NT Rp{SCEN[st.session_state.scen_val]['nt'][st.session_state.year_val]:,} / ICP USD {SCEN[st.session_state.scen_val]['icp'][st.session_state.year_val]}".replace(',','.'))

        st.markdown("<span style='font-size:12px; font-weight:bold; color:#64748b;'>INPUT NILAI TUKAR (RP/USD)</span>", unsafe_allow_html=True)
        st.session_state.nt_val = st.number_input("NT", value=st.session_state.nt_val, step=50, label_visibility="collapsed")
        
        st.markdown("<span style='font-size:12px; font-weight:bold; color:#64748b;'>INPUT HARGA MINYAK (USD/BBL)</span>", unsafe_allow_html=True)
        st.session_state.oil_val = st.number_input("OIL", value=st.session_state.oil_val, step=1, label_visibility="collapsed")
        
        st.markdown("<span style='font-size:12px; font-weight:bold; color:#64748b;'>TAHUN PROYEKSI</span>", unsafe_allow_html=True)
        y1, y2, y3, y4 = st.columns(4)
        if y1.button("2026", use_container_width=True): st.session_state.year_val = 2026
        if y2.button("2027", use_container_width=True): st.session_state.year_val = 2027
        if y3.button("2028", use_container_width=True): st.session_state.year_val = 2028
        if y4.button("2029", use_container_width=True): st.session_state.year_val = 2029

        st.markdown("<span style='font-size:12px; font-weight:bold; color:#64748b; margin-top:15px; display:block;'>PRESET SKENARIO</span>", unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        if p1.button("📊 Base Med", use_container_width=True): 
            st.session_state.nt_val = SCEN['med']['nt'][st.session_state.year_val]
            st.session_state.oil_val = SCEN['med']['icp'][st.session_state.year_val]
            st.session_state.scen_val = 'med'
        if p2.button("📈 Base High", use_container_width=True):
            st.session_state.nt_val = SCEN['high']['nt'][st.session_state.year_val]
            st.session_state.oil_val = SCEN['high']['icp'][st.session_state.year_val]
            st.session_state.scen_val = 'high'
        if p1.button("📉 Depresiasi", use_container_width=True): st.session_state.nt_val = 17500
        if p2.button("🛢️ Minyak Rndh", use_container_width=True): st.session_state.oil_val = 50
        if p1.button("🔥 Minyak Tgg", use_container_width=True): st.session_state.oil_val = 90
        if p2.button("⚡ Twin Shock", use_container_width=True): 
            st.session_state.nt_val = 17500
            st.session_state.oil_val = 90

        # Eksekusi State Saat Ini untuk Tampilan Delta di Sidebar
        b_curr, s_curr, delta_curr = simulate_eksternal_v2(st.session_state.nt_val, st.session_state.oil_val, st.session_state.year_val, st.session_state.scen_val)
        
        st.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)
        st.markdown("<span style='font-size:12px; font-weight:bold; color:#64748b;'>Δ VS BASELINE TAHUN TERPILIH</span>", unsafe_allow_html=True)
        col_sd1, col_sd2 = st.columns([6,4])
        col_sd1.markdown("PDB Growth")
        color_gdp = "red" if delta_curr['gdp'] < 0 else "green"
        col_sd2.markdown(f"<span style='color:{color_gdp}; font-weight:bold;'>{delta_curr['gdp']:+.2f}pp</span>", unsafe_allow_html=True)
        
        col_sd1, col_sd2 = st.columns([6,4])
        col_sd1.markdown("Trans. Berjalan")
        color_ca = "red" if delta_curr['ca'] < 0 else "green"
        col_sd2.markdown(f"<span style='color:{color_ca}; font-weight:bold;'>{delta_curr['ca']:+.2f} Md</span>", unsafe_allow_html=True)

    # --- 4. MAIN HEADER ---
    st.markdown("""
    <div style="display:flex; align-items:center; gap:15px; margin-bottom: 20px;">
        <div style="background:#2563eb; color:white; padding:8px 15px; border-radius:8px; font-weight:bold;">BOP</div>
        <div>
            <h2 style="margin:0; padding:0; font-size:22px; color:#1e293b;">Dashboard Neraca Pembayaran & Pertumbuhan Ekonomi</h2>
            <div style="color:#64748b; font-size:14px;">Analisis Sensitivitas: Nilai Tukar & Harga Minyak — Interaktif Live Sim</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Helper UI Cards
    def make_kpi_card(title, value, delta, is_up_good=True, top_color="#2563eb", subtitle="Miliar USD"):
        color = "#16a34a" if (delta >= 0 and is_up_good) or (delta < 0 and not is_up_good) else "#dc2626"
        arrow = "▲" if delta >= 0 else "▼"
        html = f"""
        <div style="background:white; padding:20px; border-radius:10px; border:1px solid #e2e8f0; border-top:4px solid {top_color}; box-shadow:0 1px 3px rgba(0,0,0,0.05); height:100%;">
            <div style="font-size:12px; font-weight:700; color:#64748b; text-transform:uppercase; margin-bottom:5px;">{title}</div>
            <div style="font-size:28px; font-weight:800; color:#0f172a; margin-bottom:0px;">{value}</div>
            <div style="font-size:12px; color:#94a3b8; margin-bottom:10px;">{subtitle}</div>
            <div style="font-size:13px; font-weight:700; color:{color};">{arrow} {abs(delta):.2f}</div>
        </div>
        """
        return html

    tab1, tab2, tab3 = st.tabs(['📊 Neraca Pembayaran (BOP)', '📈 Pertumbuhan Ekonomi (GDP)', '📋 Tabel Detail'])
    
    years_sim = [2026, 2027, 2028, 2029]
    def get_trend(key): return [simulate_eksternal_v2(st.session_state.nt_val, st.session_state.oil_val, y, st.session_state.scen_val)[1][key] for y in years_sim]
    def get_base_trend(key): return [simulate_eksternal_v2(st.session_state.nt_val, st.session_state.oil_val, y, st.session_state.scen_val)[0][key] for y in years_sim]

    # ==================== TAB 1: BOP ====================
    with tab1:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.markdown(make_kpi_card("Transaksi Berjalan", f"{s_curr['ca']:.2f}", delta_curr['ca'], True, "#2563eb"), unsafe_allow_html=True)
        with c2: st.markdown(make_kpi_card("Cadangan Devisa", f"{s_curr['reserves']:.1f}", delta_curr['reserves'], True, "#ea580c"), unsafe_allow_html=True)
        with c3: st.markdown(make_kpi_card("Ekspor Barang", f"{s_curr['exp']:.1f}", delta_curr['exp'], True, "#16a34a"), unsafe_allow_html=True)
        with c4: st.markdown(make_kpi_card("Impor Barang", f"{s_curr['imp']:.1f}", delta_curr['imp'], False, "#dc2626"), unsafe_allow_html=True)
        with c5: st.markdown(make_kpi_card("CA / PDB", f"{(s_curr['ca']/s_curr['gdpnom_usd']*100):.2f}%", (s_curr['ca']/s_curr['gdpnom_usd']*100) - (b_curr['ca']/b_curr['gdpnom_usd']*100), True, "#64748b", "% PDB"), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2, ch3 = st.columns([1,1,1])
        
        with ch1:
            st.markdown("<div style='background:white; border:1px solid #e2e8f0; border-radius:10px; padding:15px;'><h4 style='font-size:14px; color:#475569;'>🔵 TRANSAKSI BERJALAN (MD USD)</h4>", unsafe_allow_html=True)
            fig_ca = go.Figure(data=[
                go.Bar(name='Baseline', x=years_sim, y=get_base_trend('ca'), marker_color='#93c5fd'),
                go.Bar(name='Simulasi', x=years_sim, y=get_trend('ca'), marker_color='#2563eb')
            ])
            fig_ca.update_layout(template='plotly_white', margin=dict(l=0, r=0, t=30, b=0), height=250, legend=dict(orientation="h", y=1.2))
            st.plotly_chart(fig_ca, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with ch2:
            st.markdown("<div style='background:white; border:1px solid #e2e8f0; border-radius:10px; padding:15px;'><h4 style='font-size:14px; color:#475569;'>🟢 EKSPOR VS IMPOR BARANG</h4>", unsafe_allow_html=True)
            fig_ei = go.Figure(data=[
                go.Bar(name='Ekspor Base', x=years_sim, y=get_base_trend('exp'), marker_color='#bbf7d0'),
                go.Bar(name='Ekspor Sim', x=years_sim, y=get_trend('exp'), marker_color='#16a34a'),
                go.Bar(name='Impor Base', x=years_sim, y=get_base_trend('imp'), marker_color='#fecaca'),
                go.Bar(name='Impor Sim', x=years_sim, y=get_trend('imp'), marker_color='#dc2626')
            ])
            fig_ei.update_layout(template='plotly_white', margin=dict(l=0, r=0, t=30, b=0), height=250, barmode='group', legend=dict(orientation="h", y=1.2))
            st.plotly_chart(fig_ei, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with ch3:
            st.markdown("<div style='background:white; border:1px solid #e2e8f0; border-radius:10px; padding:15px;'><h4 style='font-size:14px; color:#475569;'>🟠 CADANGAN DEVISA (MD USD)</h4>", unsafe_allow_html=True)
            fig_res = go.Figure(data=[
                go.Scatter(name='Baseline', x=years_sim, y=get_base_trend('reserves'), mode='lines+markers', line=dict(dash='dash', color='#cbd5e1', width=3), marker=dict(size=8)),
                go.Scatter(name='Simulasi', x=years_sim, y=get_trend('reserves'), mode='lines+markers', line=dict(color='#ea580c', width=4), marker=dict(size=10))
            ])
            fig_res.update_layout(template='plotly_white', margin=dict(l=0, r=0, t=30, b=0), height=250, legend=dict(orientation="h", y=1.2))
            st.plotly_chart(fig_res, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Sensitivity Chart
        st.markdown("<div style='background:white; border:1px solid #e2e8f0; border-radius:10px; padding:20px; margin-top:15px;'><h4 style='font-size:14px; color:#475569;'>📉 KURVA SENSITIVITAS CA TERHADAP NILAI TUKAR</h4>", unsafe_allow_html=True)
        nt_range = np.linspace(12000, 24000, 20)
        ca_curve = [simulate_eksternal_v2(nt, st.session_state.oil_val, st.session_state.year_val, st.session_state.scen_val)[1]['ca'] for nt in nt_range]
        
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=nt_range, y=ca_curve, mode='lines', name='CA (Md USD)', line=dict(color='#3b82f6', width=4)))
        fig_sens.add_trace(go.Scatter(x=[st.session_state.nt_val], y=[s_curr['ca']], mode='markers', name='Posisi Saat Ini', marker=dict(color='#ea580c', size=14, symbol='circle')))
        fig_sens.update_layout(template='plotly_white', height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Nilai Tukar (Rp/USD)", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_sens, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # ==================== TAB 2: GDP ====================
    with tab2:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.markdown(make_kpi_card("PDB Growth", f"{s_curr['gdp']:.2f}%", delta_curr['gdp'], True, "#16a34a", "% YoY"), unsafe_allow_html=True)
        with c2: st.markdown(make_kpi_card("Konsumsi RT", f"{s_curr['cons']:.2f}%", delta_curr['cons'], True, "#3b82f6", "% YoY"), unsafe_allow_html=True)
        with c3: st.markdown(make_kpi_card("PMTB (Investasi)", f"{s_curr['inv']:.2f}%", delta_curr['inv'], True, "#ea580c", "% YoY"), unsafe_allow_html=True)
        with c4: st.markdown(make_kpi_card("Ekspor B&J", f"{s_curr['gexp']:.2f}%", delta_curr['gexp'], True, "#10b981", "% YoY"), unsafe_allow_html=True)
        with c5: st.markdown(make_kpi_card("Impor B&J", f"{s_curr['gimp']:.2f}%", delta_curr['gimp'], False, "#dc2626", "% YoY"), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2, ch3 = st.columns([1.2, 1.2, 1])
        
        with ch1:
            st.markdown("<div style='background:white; border:1px solid #e2e8f0; border-radius:10px; padding:15px;'><h4 style='font-size:14px; color:#475569;'>🟢 PERTUMBUHAN PDB (%)</h4>", unsafe_allow_html=True)
            fig_gdp = go.Figure(data=[
                go.Scatter(name='Baseline PDB%', x=years_sim, y=get_base_trend('gdp'), mode='lines+markers', line=dict(dash='dash', color='#cbd5e1', width=3), marker=dict(size=8)),
                go.Scatter(name='Simulasi PDB%', x=years_sim, y=get_trend('gdp'), mode='lines+markers', line=dict(color='#16a34a', width=4), marker=dict(size=10, symbol='circle-open', line_width=3))
            ])
            fig_gdp.update_layout(template='plotly_white', margin=dict(l=0, r=0, t=30, b=0), height=250, legend=dict(orientation="h", y=1.2))
            st.plotly_chart(fig_gdp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with ch2:
            st.markdown("<div style='background:white; border:1px solid #e2e8f0; border-radius:10px; padding:15px;'><h4 style='font-size:14px; color:#475569;'>🔵 KOMPONEN PDB: CONS, INV, EXP</h4>", unsafe_allow_html=True)
            fig_comp = go.Figure(data=[
                go.Bar(name='Konsumsi', x=years_sim, y=get_trend('cons'), marker_color='#93c5fd'),
                go.Bar(name='PMTB', x=years_sim, y=get_trend('inv'), marker_color='#fdba74'),
                go.Bar(name='Ekspor B&J', x=years_sim, y=get_trend('gexp'), marker_color='#86efac')
            ])
            fig_comp.update_layout(template='plotly_white', margin=dict(l=0, r=0, t=30, b=0), height=250, barmode='group', legend=dict(orientation="h", y=1.2))
            st.plotly_chart(fig_comp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with ch3:
            st.markdown("<div style='background:white; border:1px solid #e2e8f0; border-radius:10px; padding:15px;'><h4 style='font-size:14px; color:#475569;'>📉 SENSITIVITAS PDB VS NILAI TUKAR</h4>", unsafe_allow_html=True)
            gdp_curve = [simulate_eksternal_v2(nt, st.session_state.oil_val, st.session_state.year_val, st.session_state.scen_val)[1]['gdp'] for nt in nt_range]
            fig_gdp_sens = go.Figure()
            fig_gdp_sens.add_trace(go.Scatter(x=nt_range, y=gdp_curve, mode='lines', name='PDB Growth (%)', line=dict(color='#16a34a', width=4)))
            fig_gdp_sens.add_trace(go.Scatter(x=[st.session_state.nt_val], y=[s_curr['gdp']], mode='markers', name='Posisi Saat Ini', marker=dict(color='#ea580c', size=14)))
            fig_gdp_sens.update_layout(template='plotly_white', height=250, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=1.2))
            st.plotly_chart(fig_gdp_sens, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Tabel Mekanisme Transmisi
        st.markdown("<div style='background:white; border:1px solid #e2e8f0; border-radius:10px; padding:20px; margin-top:15px;'><h4 style='font-size:14px; color:#475569;'>🟠 MEKANISME TRANSMISI: NILAI TUKAR & HARGA MINYAK ➔ PDB</h4>", unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        
        # Simulasi Hitungan Transmisi (Visual Only)
        dNT_pct = (st.session_state.nt_val - b_curr['nt']) / b_curr['nt'] * 100 if b_curr['nt'] else 0
        dOil_pct = (st.session_state.oil_val - b_curr['icp']) / b_curr['icp'] * 100 if b_curr['icp'] else 0
        
        with t1:
            st.markdown("**Jalur Nilai Tukar ke PDB**")
            html_t1 = f"""
            <table style="width:100%; border-collapse: collapse; font-size:14px;">
                <tr style="border-bottom:1px solid #eee;"><td>Daya saing ekspor non-migas</td><td align="right" style="color:green; font-weight:bold;">+{max(0.05 * dNT_pct, 0):.2f}pp</td></tr>
                <tr style="border-bottom:1px solid #eee;"><td>Substitusi impor (volume turun)</td><td align="right" style="color:green; font-weight:bold;">+{max(0.04 * dNT_pct, 0):.2f}pp</td></tr>
                <tr style="border-bottom:1px solid #eee;"><td>Konsumsi (tekanan inflasi impor)</td><td align="right" style="color:red; font-weight:bold;">{-0.10 * dNT_pct:.2f}pp</td></tr>
                <tr style="border-bottom:1px solid #eee;"><td>Investasi (ketidakpastian/biaya modal)</td><td align="right" style="color:red; font-weight:bold;">{-0.06 * dNT_pct:.2f}pp</td></tr>
                <tr><td><b>Total Δ PDB via NT</b></td><td align="right"><b>{((-0.10)+(-0.06)+(0.05)+(0.04)) * dNT_pct:.2f}pp</b></td></tr>
            </table>
            """
            st.markdown(html_t1, unsafe_allow_html=True)
            
        with t2:
            st.markdown("**Jalur Harga Minyak ke PDB**")
            html_t2 = f"""
            <table style="width:100%; border-collapse: collapse; font-size:14px;">
                <tr style="border-bottom:1px solid #eee;"><td>Ekspor migas (pendapatan)</td><td align="right" style="color:green; font-weight:bold;">+{max(0.02 * dOil_pct, 0):.2f}pp</td></tr>
                <tr style="border-bottom:1px solid #eee;"><td>Tagihan impor BBM (beban)</td><td align="right" style="color:red; font-weight:bold;">{-0.01 * dOil_pct:.2f}pp</td></tr>
                <tr style="border-bottom:1px solid #eee;"><td>Biaya produksi / investasi</td><td align="right" style="color:red; font-weight:bold;">{-0.03 * dOil_pct:.2f}pp</td></tr>
                <tr style="border-bottom:1px solid #eee;"><td>Daya beli RT (harga BBM)</td><td align="right" style="color:red; font-weight:bold;">{-0.05 * dOil_pct:.2f}pp</td></tr>
                <tr><td><b>Total Δ PDB via Minyak</b></td><td align="right"><b>{((0.02)+(-0.01)+(-0.03)+(-0.05)) * dOil_pct:.2f}pp</b></td></tr>
            </table>
            """
            st.markdown(html_t2, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # ==================== TAB 3: TABEL DETAIL ====================
    with tab3:
        st.markdown("<div style='background:white; border:1px solid #e2e8f0; border-radius:10px; padding:20px;'><h4 style='font-size:16px; color:#1e293b; margin-bottom:20px;'>🔵 TABEL LENGKAP: BOP & GDP — BASELINE VS SIMULASI</h4>", unsafe_allow_html=True)
        
        def row_html(label, b_val, s_val, unit, is_bold=False, is_title=False):
            d_abs = s_val - b_val
            d_pct = (d_abs / abs(b_val) * 100) if b_val != 0 else 0
            c_abs = "green" if d_abs >= 0 else "red"
            c_pct = "green" if d_pct >= 0 else "red"
            fw = "bold" if is_bold else "normal"
            bg = "#f8fafc" if is_title else "white"
            color_title = "#2563eb" if is_title else "#334155"
            
            return f"""
            <tr style="border-bottom:1px solid #e2e8f0; background:{bg};">
                <td style="padding:12px; font-weight:{fw}; color:{color_title};">{label}</td>
                <td align="right" style="padding:12px;">{b_val:,.2f}</td>
                <td align="right" style="padding:12px; font-weight:bold; color:#0f172a;">{s_val:,.2f}</td>
                <td align="right" style="padding:12px; color:{c_abs}; font-weight:bold;">{d_abs:+.2f}</td>
                <td align="right" style="padding:12px; color:{c_pct}; font-weight:bold;">{d_pct:+.1f}%</td>
                <td align="right" style="padding:12px; color:#64748b; font-size:12px;">{unit}</td>
            </tr>
            """
            
        html_table = f"""
        <table style="width:100%; border-collapse: collapse; font-size:14px; font-family:sans-serif;">
            <tr style="border-bottom:2px solid #cbd5e1; color:#64748b; font-size:12px;">
                <th align="left" style="padding:12px;">INDIKATOR</th><th align="right">BASELINE</th><th align="right">SIMULASI</th><th align="right">Δ ABSOLUT</th><th align="right">Δ %</th><th align="right">SATUAN</th>
            </tr>
            {row_html("I. Transaksi Berjalan", b_curr['ca'], s_curr['ca'], "Md USD", True, True)}
            {row_html("Neraca Barang", b_curr['tradebal'], s_curr['tradebal'], "Md USD")}
            {row_html("Ekspor (fob)", b_curr['exp'], s_curr['exp'], "Md USD")}
            {row_html("Impor (fob)", b_curr['imp'], s_curr['imp'], "Md USD")}
            {row_html("Neraca Jasa", b_curr['svcbal'], s_curr['svcbal'], "Md USD")}
            {row_html("Pendapatan Primer", b_curr['primbal'], s_curr['primbal'], "Md USD")}
            {row_html("Pendapatan Sekunder", b_curr['secbal'], s_curr['secbal'], "Md USD")}
            {row_html("II. Transaksi Modal", b_curr['capbal'], s_curr['capbal'], "Md USD", True, True)}
            {row_html("III. Transaksi Finansial", b_curr['finbal'], s_curr['finbal'], "Md USD", True, True)}
            {row_html("IV. Total (I+II+III)", b_curr['total'], s_curr['total'], "Md USD", True, True)}
            {row_html("Cadangan Devisa", b_curr['reserves'], s_curr['reserves'], "Md USD", True, True)}
            {row_html("Bulan Impor", b_curr['bulan_imp'], s_curr['bulan_imp'], "Bulan", True, True)}
            {row_html("★ PDB Growth (GDP Riil)", b_curr['gdp'], s_curr['gdp'], "%", True, True)}
        </table>
        """
        st.markdown(html_table, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# MODUL 3: EKONOMI DAERAH (WIP)
# ==========================================
elif main_menu == "📍 Ekonomi Daerah (WIP)":
    st.markdown("### 📍 Command Center: Ekonomi Kewilayahan")
    st.info("🚧 Modul analitik data daerah sedang dalam tahap pengkodingan dan pengembangan lanjutan oleh tim Data Science Bappenas. Silakan kembali lagi nanti.")
