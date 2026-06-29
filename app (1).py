import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
import os
import pickle
import warnings
import hashlib
import sqlite3
from datetime import datetime
import traceback
import re
import gdown
import json

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

def make_signature(view, avg, target, monthly_info, daily_info, ext_nt, ext_oil):
    raw_str = f"{view}_{avg:.2f}_{target}_{monthly_info}_{daily_info}_{ext_nt}_{ext_oil}"
    return hashlib.md5(raw_str.encode()).hexdigest()

# ==========================================
# 1. DIREKTORI & AUTO-DOWNLOAD DATABASE BPS
# ==========================================
DATA_DIR     = os.environ.get("DATA_DIR", os.path.dirname(__file__) if '__file__' in globals() else os.getcwd())
BPS_DB_PATH  = os.path.join(DATA_DIR, os.environ.get("BPS_DB_FILE", "ekspor_impor_bps.db"))
BOP_DB_PATH  = os.path.join(DATA_DIR, os.environ.get("BOP_DB_FILE", "bop_indonesia.db"))
TM_XLSX      = os.path.join(DATA_DIR, os.environ.get("TM_XLSX_FILE", "data_trademap.xlsx"))

# Fungsi Krusial: Download DB BPS dari GDrive
GDRIVE_FILE_ID = "1IhKP7Jw7xhRPDzvw4CY7FVvK0lO_biy_"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

@st.cache_resource(show_spinner=False)
def download_bps_database():
    if not os.path.exists(BPS_DB_PATH):
        with st.spinner("Sedang mengunduh Database BPS (~600MB) dari Cloud. Mohon tunggu beberapa saat..."):
            try:
                gdown.download(GDRIVE_URL, BPS_DB_PATH, quiet=False)
            except Exception as e:
                st.error(f"Gagal mengunduh database: {e}")

download_bps_database()

BPS_TABLE    = "exim_data" 
HS_ALL       = [str(i).zfill(2) for i in range(1, 100)]
TAHUN_SAAT_INI = datetime.now().year
TAHUN_TERSEDIA_BPS = list(range(2015, TAHUN_SAAT_INI + 1))

PERIODE_OPSI = {
    "Tahunan": "tahunan", "Januari": "1", "Februari": "2", "Maret": "3", 
    "April": "4", "Mei": "5", "Juni": "6", "Juli": "7", "Agustus": "8", 
    "September": "9", "Oktober": "10", "November": "11", "Desember": "12"
}

PARTNER_LIST = [
    "China", "Amerika Serikat", "Jepang", "Singapura",
    "India", "Malaysia", "Korea Selatan", "Australia",
    "Jerman", "Belanda", "Thailand", "Vietnam",
]

HS_DESC = {
    "01":"Binatang Hidup","02":"Daging & Produk Daging","03":"Ikan & Produk Ikan",
    "04":"Produk Susu & Telur","05":"Produk Hewani Lainnya","06":"Tanaman Hidup & Bunga",
    "07":"Sayuran","08":"Buah-buahan","09":"Kopi, Teh & Rempah","10":"Serealia",
    "11":"Produk Penggilingan","12":"Biji Minyak","13":"Getah & Resin",
    "14":"Bahan Nabati Lainnya","15":"Lemak & Minyak Nabati/Hewani","16":"Olahan Daging & Ikan",
    "17":"Gula & Kembang Gula","18":"Kakao & Olahannya","19":"Olahan Sereal & Tepung",
    "20":"Olahan Sayur & Buah","21":"Aneka Olahan Pangan","22":"Minuman & Cuka",
    "23":"Ampas Industri Pangan","24":"Tembakau","25":"Garam, Belerang & Batu",
    "26":"Bijih, Terak & Abu","27":"Bahan Bakar Mineral & Minyak Bumi",
    "28":"Kimia Anorganik","29":"Kimia Organik","30":"Produk Farmasi",
    "31":"Pupuk","32":"Cat, Tinta & Vernis","33":"Minyak Atsiri & Kosmetik",
    "34":"Sabun & Deterjen","35":"Albuminoid & Pati","36":"Bahan Peledak",
    "37":"Produk Foto","38":"Kimia Lainnya","39":"Plastik & Barang Plastik",
    "40":"Karet & Barang Karet","41":"Kulit Mentah & Samak","42":"Barang Kulit & Tas",
    "43":"Bulu Binatang","44":"Kayu & Produk Kayu","45":"Gabus",
    "46":"Produk Anyaman","47":"Bubur Kayu (Pulp)","48":"Kertas & Karton",
    "49":"Buku & Produk Cetak","50":"Sutra","51":"Wol & Bulu Halus",
    "52":"Kapas","53":"Serat Tekstil Lainnya","54":"Filamen Buatan",
    "55":"Serat Staple Buatan","56":"Benang & Kain Khusus","57":"Karpet & Alas Lantai",
    "58":"Kain Tenunan Khusus","59":"Kain Teknik","60":"Kain Rajut",
    "61":"Pakaian Rajut","62":"Pakaian Tenun","63":"Tekstil Rumah Tangga",
    "64":"Alas Kaki","65":"Topi & Aksesori Kepala","66":"Payung & Tongkat",
    "67":"Bulu Olahan & Bunga Buatan","68":"Barang dari Batu & Semen",
    "69":"Produk Keramik","70":"Kaca & Produk Kaca",
    "71":"Batu Permata & Logam Mulia","72":"Besi & Baja",
    "73":"Barang dari Besi/Baja","74":"Tembaga & Produknya",
    "75":"Nikel & Produknya","76":"Aluminium & Produknya",
    "78":"Timbal & Produknya","79":"Seng & Produknya","80":"Timah & Produknya",
    "81":"Logam Dasar Lainnya","82":"Perkakas & Peralatan Logam",
    "83":"Barang Logam Lainnya","84":"Mesin & Perlengkapan Mekanik",
    "85":"Mesin & Perlengkapan Listrik","86":"Kereta Api & Komponen",
    "87":"Kendaraan Bermotor","88":"Pesawat Udara & Komponen",
    "89":"Kapal & Perahu","90":"Instrumen Optik & Medis",
    "91":"Jam & Arloji","92":"Instrumen Musik","93":"Senjata & Amunisi",
    "94":"Furnitur & Perlengkapan","95":"Mainan & Perlengkapan Olahraga",
    "96":"Produk Manufaktur Lainnya","97":"Karya Seni & Antik","99":"Barang Lainnya",
}

BOP_MAIN_ITEMS = {
    1:"Transaksi Berjalan", 2:"Barang", 17:"Jasa",
    20:"Pendapatan Primer", 23:"Pendapatan Sekunder",
    26:"Transaksi Modal", 29:"Transaksi Finansial",
    32:"Investasi Langsung", 35:"Investasi Portofolio",
    40:"Derivatif Finansial", 41:"Investasi Lainnya",
    46:"Total (I+II+III)", 47:"Selisih Perhitungan",
    48:"Neraca Keseluruhan", 54:"Cadangan Devisa",
    56:"CA % PDB",
}

_NEGARA_KW = {
    "China": ["china", "cina", "tiongkok", "zhongguo"],
    "Amerika Serikat": ["united states", "america", "u.s.a", "u.s.", "serikat", "usa"],
    "Jepang": ["japan", "jepang", "nippon"],
    "Singapura": ["singapore", "singapura"],
    "India": ["india"], "Malaysia": ["malaysia"], "Korea Selatan": ["korea"],
    "Australia": ["australia"], "Jerman": ["germany", "jerman"],
    "Belanda": ["netherlands", "belanda"], "Thailand": ["thailand"], "Vietnam": ["vietnam"],
}

ATURAN_WARNA = {
    'PMI Manufaktur Negara Berkembang': True, 'Jumlah Uang Yang Beredar': True, 
    'Penjualan Mobil': True, 'Penjualan semen': True, 'Ekspor Barang': True, 
    'Impor Barang Modal': True, 'Impor Bahan Baku': True, 'Kredit Perbankan': True, 
    'Penjualan Motor': True, 'Indeks Keyakinan Konsumen': True, 'Impor Barang Konsumsi': True, 
    'Inflasi': False, 'Nilai Tukar terhadap Dolar AS': False, 'Suku Bunga': False
}

# ==========================================
# 2. GLOBAL SETUP SENSITIVITAS
# ==========================================
st.set_page_config(
    page_title="Macro AI Command Center", 
    layout="wide", 
    page_icon="🇮🇩", 
    initial_sidebar_state="expanded"
)

SCEN = {
    "med": {
        "nt":         {2026: 16700, 2027: 17100, 2028: 16300, 2029: 16200},
        "icp":        {2026: 86,    2027: 70,    2028: 65,    2029: 65},
        "ca":         {2026: -3.568,   2027: -6.200,  2028: 30.655,   2029: 3.750},
        "tradebal":   {2026: 27.81,   2027: 24.73,   2028: 47.772,   2029: 23.016},
        "exp":        {2026: 346.64,  2027: 362.22,  2028: 367.146,  2029: 381.850},
        "imp":        {2026: -318.83, 2027: -337.49, 2028: -319.374, 2029: -358.834},
        "svcbal":     {2026: -19.42,  2027: -18.32,  2028: -8.188,   2029: -10.202},
        "primbal":    {2026: -38.40,  2027: -38.52,  2028: -14.082,  2029: -14.907},
        "secbal":     {2026: 7.01,    2027: 7.59,   2028: 5.153,    2029: 5.842},
        "capbal":     {2026: 0.35,    2027: 0.35,    2028: 0.352,    2029: 0.352},
        "finbal":     {2026: 6.65,   2027: 8.84,   2028: 19.737,   2029: 6.940},
        "total":      {2026: 3.42,    2027: 2.99,    2028: 50.744,   2029: 11.042},
        "reserves":   {2026: 157.70,  2027: 158.50,  2028: 213.679,  2029: 222.526},
        "bulan_imp":  {2026: 5.94,    2027: 5.64,    2028: 6.810,    2029: 6.333},
        "capdb":      {2026: -0.24,   2027: -0.38,   2028: 1.499,    2029: 0.163},
        "gdpnom_usd": {2026: 1504.52,   2027: 1633.79,   2028: 2044.6,   2029: 2299.6},
        "gdp":        {2026: 5.4,    2027: 5.8,    2028: 7.7,    2029: 8.0},
        "cons":       {2026: 5.22,  2027: 5.31,  2028: 7.996,  2029: 8.008},
        "gov":        {2026: 8.41,  2027: 4.34, 2028: 7.099,  2029: 16.055},
        "inv":        {2026: 5.41,  2027: 6.54,  2028: 10.324, 2029: 11.149},
        "gexp":       {2026: 3.92, 2027: 7.10,  2028: -4.789, 2029: 1.538},
        "gimp":       {2026: 5.51,  2027: 7.80, 2028: -4.232, 2029: 7.928},
        "rev":        {2026: 3196.9,  2027: 3302.9,  2028: 5015.6,  2029: 6286.5},
        "bel":        {2026: 3916.5,  2027: 3805.3,  2028: 5526.9,  2029: 6396.4},
        "def":        {2026: -719.63,  2027: -502.39,  2028: -511.3,  2029: -109.9},
        "defpdb":     {2026: -2.81,  2027: -1.80,  2028: -1.635,  2029: -0.315},
        "sube":       {2026: 286.6,   2027: 186.4,   2028: 197.8,   2029: 234.9},
        "subnon":     {2026: 106.3,   2027: 112.1,   2028: 164.5,   2029: 203.1},
        "bunga":      {2026: 600.6,   2027: 629.1,   2028: 785.7,   2029: 785.7},
        "pajak":      {2026: 2700.0,  2027: 2799.4,  2028: 4311.0,  2029: 5466.6},
        "pnbp":       {2026: 496.3,   2027: 503.0,   2028: 699.0,   2029: 813.6},
        "migas":      {2026: 150.3,   2027: 131.7,   2028: 167.8,   2029: 194.8},
        "pdb":        {2026: 25576.9, 2027: 27397.7, 2028: 31810.0, 2029: 34938.3},
    },
    "high": {
        "nt":         {2026: 16700, 2027: 16500, 2028: 16300, 2029: 16200},
        "icp":        {2026: 65,    2027: 75,    2028: 75,    2029: 75},
        "ca":         {2026: -3.568,   2027: -3.340,   2028: 18.853,   2029: 10.329},
        "tradebal":   {2026: 27.81,   2027: 27.23,   2028: 63.627,   2029: 51.137},
        "exp":        {2026: 346.64,  2027: 385.66,  2028: 397.482,  2029: 427.061},
        "imp":        {2026: -318.83, 2027: -358.43, 2028: -333.856, 2029: -375.924},
        "svcbal":     {2026: -19.42,  2027: -18.08,  2028: -13.579,  2029: -13.500},
        "primbal":    {2026: -38.40,  2027: -38.93,  2028: -37.201,  2029: -35.580},
        "secbal":     {2026: 7.01,    2027: 8.35,    2028: 7.725,    2029: 8.239},
        "capbal":     {2026: 0.35,    2027: 0.35,    2028: 0.352,    2029: 0.352},
        "finbal":     {2026: 6.65,    2027: 9.55,   2028: 16.630,   2029: 14.697},
        "total":      {2026: 3.42,    2027: 6.56,    2028: 35.835,   2029: 25.378},
        "reserves":   {2026: 157.70,  2027: 162.07,  2028: 219.162,  2029: 208.666},
        "bulan_imp":  {2026: 5.94,    2027: 5.43,    2028: 6.810,    2029: 6.333},
        "capdb":      {2026: -0.24,    2027: -0.20,    2028: 1.151,    2029: 0.563},
        "gdpnom_usd": {2026: 1504.52,   2027: 1644.20,   2028: 1982.1,   2029: 2247.0},
        "gdp":        {2026: 5.4,    2027: 6.5,    2028: 7.7,    2029: 8.0},
        "cons":       {2026: 5.22,  2027: 5.61,  2028: 8.512,  2029: 8.525},
        "gov":        {2026: 8.41,  2027: 7.72,  2028: 9.557,  2029: 10.901},
        "inv":        {2026: 5.41,  2027: 7.47,  2028: 11.601, 2029: 12.410},
        "gexp":       {2026: 3.92, 2027: 8.80,  2028: -0.811, 2029: 3.454},
        "gimp":       {2026: 5.51,  2027: 9.89,  2028: 8.635,  2029: 8.000},
        "rev":        {2026: 3196.9,  2027: 3486.0,  2028: 5047.0,  2029: 6417.0},
        "bel":        {2026: 3916.5,  2027: 4162.3,  2028: 4977.3,  2029: 6314.1},
        "def":        {2026: -719.63,  2027: -676.34,  2028: 69.7,    2029: 102.9},
        "defpdb":     {2026: -2.81,  2027: -2.40,  2028: 0.219,   2029: 0.290},
        "sube":       {2026: 286.6,   2027: 192.8,   2028: 195.4,   2029: 194.1},
        "subnon":     {2026: 107.1,   2027: 114.3,   2028: 142.3,   2029: 293.9},
        "bunga":      {2026: 600.6,   2027: 636.5,   2028: 808.3,   2029: 947.1},
        "pajak":      {2026: 2700.0,  2027: 2952.4,  2028: 4356.1,  2029: 5558.6},
        "pnbp":       {2026: 496.3,   2027: 532.7,   2028: 685.2,   2029: 852.0},
        "migas":      {2026: 150.3,   2027: 136.0,   2028: 146.1,   2029: 222.7},
        "pdb":        {2026: 25576.9, 2027: 28115.9, 2028: 31810.0, 2029: 35524.1},
    },
}

# --- Koefisien Elastisitas ---
EL = {
    "bop_exp_nt":      0.15,   "bop_imp_nt":      -0.25,
    "bop_exp_oil":     0.80,   "bop_imp_oil":      0.95,
    "bop_svc_nt":      0.05,   "bop_prim_nt":     -0.03,
    "share_exp_migas": 0.06,   "share_imp_migas":  0.16,
    
    # Transmisi GDP
    "gexp_nt":         0.15,   "gimp_nt":         -0.10, 
    "cons_nt":        -0.10,   "inv_nt":          -0.06, 
    "gexp_oil":        0.025,  "gimp_oil":         0.018,
    "cons_oil":       -0.05,   "inv_oil":         -0.03, 

    # Bobot PDB terhadap pertumbuhan
    "w_cons":          0.547,  "w_inv":            0.317,
    "w_exp":           0.22,   "w_imp":            0.27,
    
    "sube_oil":        0.95,   "sube_nt":         -0.15,
    "bunga_nt":        0.006,
}

YEARS = [2026, 2027, 2028, 2029]

C = {
    "blue":    "#2563eb", "green":   "#16a34a", "red":     "#dc2626",
    "red2":    "#b91c1c", "amber":   "#d97706", "purple":  "#7c3aed",
    "orange":  "#ea580c", "orange2": "#c2410c", "teal":    "#0891b2",
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
    
    # Kalkulasi Delta untuk BOP
    delta_trade = s["tradebal"] - b["tradebal"]
    delta_svc = s["svcbal"] - b["svcbal"]
    delta_prim = s["primbal"] - b["primbal"]
    s["ca"]         = b["ca"] + delta_trade + delta_svc + delta_prim
    
    delta_ca = s["ca"] - b["ca"]
    s["total"]      = b["total"] + delta_ca
    s["reserves"]   = b["reserves"] + delta_ca
    
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
    dCons_nt  = EL["cons_nt"]  * dNT_pct
    dInv_nt   = EL["inv_nt"]   * dNT_pct

    dGexp_oil = EL["gexp_oil"] * dOil_pct
    dGimp_oil = EL["gimp_oil"] * dOil_pct
    dCons_oil = EL["cons_oil"] * dOil_pct
    dInv_oil  = EL["inv_oil"]  * dOil_pct

    s["gexp"] = b["gexp"] + dGexp_nt + dGexp_oil
    s["gimp"] = b["gimp"] + dGimp_nt + dGimp_oil
    s["cons"] = b["cons"] + dCons_nt + dCons_oil
    s["inv"]  = b["inv"] + dInv_nt + dInv_oil

    impact_netexp_nt = EL["w_exp"] * dGexp_nt - EL["w_imp"] * dGimp_nt
    impact_cons_nt   = EL["w_cons"] * dCons_nt
    impact_inv_nt    = EL["w_inv"] * dInv_nt
    total_gdp_nt     = impact_netexp_nt + impact_cons_nt + impact_inv_nt

    impact_netexp_oil = EL["w_exp"] * dGexp_oil - EL["w_imp"] * dGimp_oil
    impact_cons_oil   = EL["w_cons"] * dCons_oil
    impact_inv_oil    = EL["w_inv"] * dInv_oil
    total_gdp_oil     = impact_netexp_oil + impact_cons_oil + impact_inv_oil

    s["gov"]  = b["gov"]
    s["gdp"]  = b["gdp"] + total_gdp_nt + total_gdp_oil

    s["tx"] = {
        "expNT": dGexp_nt, "impNT": dGimp_nt, "netNT": impact_netexp_nt,
        "consNT": impact_cons_nt, "invNT": impact_inv_nt, "totalNT": total_gdp_nt,
        "expICP": dGexp_oil, "impICP": dGimp_oil, "netICP": impact_netexp_oil,
        "consICP": impact_cons_oil, "invICP": impact_inv_oil, "totalICP": total_gdp_oil,
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
    s["def"]    = b["def"]   + dRevTotal - dBelTotal 
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

def kpi_card(title, value, color, sub_text=""):
    st.markdown(f"""
        <div style="border: 1px solid rgba(128,128,128,0.2); border-top: 3px solid {color}; 
                    padding: 15px; border-radius: 6px; margin-bottom: 15px; background-color: transparent;">
            <div style="font-size: 11px; color: gray; letter-spacing: 1px;">{title}</div>
            <div style="font-size: 22px; font-weight: bold; color: {color}; margin-top: 5px;">{value}</div>
            <div style="font-size: 11px; color: gray; margin-top: 5px;">{sub_text}</div>
        </div>
    """, unsafe_allow_html=True)

def section_title(title):
    st.markdown(f"<p style='font-size:11px; font-weight:bold; letter-spacing:1px; margin-bottom:0px; text-transform:uppercase;'>{title}</p>", unsafe_allow_html=True)

def normalize_negara(nama: str):
    if not nama: return nama
    low = str(nama).strip().lower()
    for display, kws in _NEGARA_KW.items():
        if low in kws or any(kw in low for kw in kws): return display
    return str(nama).strip()

def clean_hs(raw):
    if pd.isna(raw) or str(raw).strip() == "": return ""
    s = str(raw).strip()
    m = re.search(r'\d+', s)
    if m: return m.group(0)[:2].zfill(2)
    return s[:2].zfill(2)

def get_periode_params(pilihan):
    return ("2", "") if pilihan == "tahunan" else ("1", pilihan)

def check_bps_db():
    return os.path.exists(BPS_DB_PATH)

_BASE_LAYOUT = dict(
    paper_bgcolor="white", plot_bgcolor="#f8f9fa",
    font=dict(family="monospace", size=11, color="#4b5563"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=10),
    margin=dict(l=40, r=20, t=50, b=40), height=270,
    xaxis=dict(gridcolor="#e5e7eb", showgrid=True), yaxis=dict(gridcolor="#e5e7eb", showgrid=True),
)

def fig_layout(title: str, barmode: str = None, **kwargs):
    layout = {**_BASE_LAYOUT, "title": dict(text=title, font=dict(size=13)), **kwargs}
    if barmode: layout["barmode"] = barmode
    return layout

def bar_trace(name, x, y, color, opacity=0.75): return go.Bar(name=name, x=x, y=y, marker_color=color, marker_line_width=0, opacity=opacity)
def line_trace(name, x, y, color, fill=None, fillcolor=None, width=2, size=6): return go.Scatter(name=name, x=x, y=y, mode="lines+markers", line=dict(color=color, width=width), marker=dict(size=size, color=color), fill=fill, fillcolor=fillcolor)
def line_base_trace(name, x, y): return go.Scatter(name=name, x=x, y=y, mode="lines+markers", line=dict(color=C["gray"], dash="dash", width=1.5), marker=dict(size=4, color=C["gray"]))
def dot_trace(name, x, y): return go.Scatter(name=name, x=x, y=y, mode="markers", marker=dict(size=11, color=C["amber"], line=dict(color="white", width=2)))
def delta_color(val: float): return "green" if val > 0.005 else "red" if val < -0.005 else "gray"
def metric_delta_color(val: float): return "normal" if delta_color(val) == "green" else "inverse" if delta_color(val) == "red" else "off"

@st.cache_data
def load_data():
    try:
        df_target = pd.read_excel(file_makro, sheet_name=0)
        df_triwulan = pd.read_excel(file_makro, sheet_name=1)
        df_makro = pd.read_excel(file_makro, sheet_name=2)
        df_hist_gdp = pd.read_excel(file_adb, sheet_name=2)
        return df_target, df_triwulan, df_makro, df_hist_gdp
    except Exception as e:
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
        return None, None

df_daily, date_col_daily = load_daily_data()

@st.cache_data(ttl=600)
def fetch_bps_db(sumber, tahun, tipe, bulan=""):
    if not check_bps_db(): return pd.DataFrame()
    try:
        conn = sqlite3.connect(BPS_DB_PATH)
        jenis = "Ekspor" if str(sumber) == "1" else "Impor"
        query = f"SELECT kode_hs as kodehs, ctr as negara, value, netweight as berat FROM {BPS_TABLE} WHERE jenis_transaksi = ? AND tahun = ?"
        params = [jenis, str(tahun)]
        if tipe == "1" and bulan:
            query += " AND bulan_kode = ?"
            params.append(str(bulan).zfill(2)) 
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        if not df.empty:
            df["kodehs"] = df["kodehs"].apply(clean_hs)
            df["negara"] = df["negara"].apply(normalize_negara)
            df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
            df["berat"] = pd.to_numeric(df["berat"], errors="coerce").fillna(0)
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_hist_bps_db(sumber, hs, tipe, bulan=""):
    if not check_bps_db(): return pd.DataFrame()
    try:
        conn = sqlite3.connect(BPS_DB_PATH)
        jenis = "Ekspor" if str(sumber) == "1" else "Impor"
        query = f"SELECT tahun, ctr as negara, value FROM {BPS_TABLE} WHERE jenis_transaksi = ? AND kode_hs = ?"
        params = [jenis, str(hs).zfill(2)]
        if tipe == "1" and bulan:
            query += " AND bulan_kode = ?"
            params.append(str(bulan).zfill(2))
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        if not df.empty:
            df["negara"] = df["negara"].apply(normalize_negara)
            df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_trademap(mitra, tahun, sumber):
    if not os.path.exists(TM_XLSX): return pd.DataFrame(), "FILE_NOT_FOUND"
    try:
        df = pd.read_excel(TM_XLSX)
        need = ["Tahun", "Mitra", "HS", "Impor_Mitra", "Ekspor_Mitra"]
        if not all(c in df.columns for c in need): return pd.DataFrame(), "INVALID_COLUMNS"
        
        df["HS"] = df["HS"].apply(clean_hs)
        df["Tahun"] = df["Tahun"].astype(str).str.strip()
        df["_mitra_n"] = df["Mitra"].apply(normalize_negara)
        mitra_n = normalize_negara(mitra)
        
        df_f = df[(df["_mitra_n"] == mitra_n) & (df["Tahun"] == str(tahun))].copy()
        if df_f.empty:
            tahun_ada = df[df["_mitra_n"] == mitra_n]["Tahun"].unique().tolist()
            if tahun_ada: return pd.DataFrame(), f"DATA_EMPTY_TAHUN|{','.join(sorted(tahun_ada))}"
            return pd.DataFrame(), "DATA_EMPTY"
            
        col = "Impor_Mitra" if str(sumber) == "1" else "Ekspor_Mitra"
        df_f[col] = pd.to_numeric(df_f[col], errors="coerce").fillna(0)
        df_out = df_f.groupby("HS", as_index=False)[col].sum().rename(columns={col: "Trademap_Value"})
        return df_out, "SUCCESS"
    except Exception as e:
        return pd.DataFrame(), str(e)

@st.cache_data(ttl=3600)
def bop_query(sql, params=()):
    if not os.path.exists(BOP_DB_PATH): return pd.DataFrame()
    try:
        with sqlite3.connect(BOP_DB_PATH) as conn:
            return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()

def bop_latest():
    df = bop_query("""SELECT period FROM bop_quarterly WHERE value_mn_usd IS NOT NULL ORDER BY year DESC, quarter DESC LIMIT 1""")
    return df["period"].iloc[0] if not df.empty else "-"

def bop_latest_val(item_id):
    df = bop_query("""SELECT value_mn_usd FROM bop_quarterly WHERE item_id=? AND value_mn_usd IS NOT NULL ORDER BY year DESC, quarter DESC LIMIT 1""", (item_id,))
    return float(df["value_mn_usd"].iloc[0]) if not df.empty else None

def bop_series(item_ids, y1, y2, freq):
    ph = ",".join("?" * len(item_ids))
    sql = f"""SELECT item_id, keterangan, items_en, year, quarter, period, value_mn_usd
              FROM bop_quarterly WHERE item_id IN ({ph}) AND year >= ? AND year <= ? ORDER BY item_id, year, quarter"""
    df = bop_query(sql, tuple(item_ids) + (y1, y2))
    if df.empty or freq == "quarterly": return df
    
    parts = []
    ratio = {54, 55, 56, 57, 58}
    for iid, grp in df.groupby("item_id"):
        if iid in ratio: r = grp[grp["quarter"] == "Q4"].copy()
        else: r = grp.groupby(["item_id","keterangan","items_en","year"], as_index=False)["value_mn_usd"].sum()
        parts.append(r)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def calculate_ews(df):
    if df.empty: return df
    df["harga"] = df.apply(lambda row: row["value"] / row["berat"] if row.get("berat", 0) > 0 else 0, axis=1)
    m_val, s_val = df["value"].mean(), df["value"].std()
    df["z_score"] = 0 if pd.isna(s_val) or s_val == 0 else (df["value"] - m_val) / s_val
    df_vh = df[df["harga"] > 0]
    m_h, s_h = (df_vh["harga"].mean(), df_vh["harga"].std()) if not df_vh.empty else (0, 0)
    df["z_score_harga"] = df.apply(lambda row: 0 if pd.isna(s_h) or s_h == 0 or row["harga"] == 0 else (row["harga"] - m_h) / s_h, axis=1)
    
    df["status_ews"] = "Normal"
    df.loc[df["z_score"] >  1.5, "status_ews"] = "🔴 Batas Atas Nilai"
    df.loc[df["z_score"] < -0.5, "status_ews"] = "🟡 Batas Bawah Nilai"
    df.loc[df["z_score_harga"] > 2.0, "status_ews"] = "🟣 Anomali Harga"
    mask = (df["z_score"] > 1.5) & (df["z_score_harga"] > 2.0)
    df.loc[mask, "status_ews"] = "🚨 KRITIS: Spike"
    return df

def apply_matlab_transformation(series, j1, j2, j3, freq='M'):
    out = series.copy().astype(float)
    if j1 == 1:
        out = out.mask(out <= 0, np.nan)
        out = 100 * np.log(out)
    if j2 == 1: out = out.diff(1)
    elif j3 == 1: out = out.diff(12 if freq == 'M' else 4)
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
            
            model = DynamicFactorMQ(endog=end_m, endog_quarterly=end_q, factors=2, factor_orders=2, idiosyncratic_ar=2, standardize=True)
            res = model.fit(method='em', maxiter=1500, tolerance=1e-6, disp=False)
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
        return pd.DataFrame()


# ==========================================
# 4. UI HEADER & MENU
# ==========================================
st.markdown("""
<style>
[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e1e4e8; }
.main-header { background: linear-gradient(135deg, #1d4ed8, #2563eb); color: white; padding: 14px 20px; border-radius: 10px; margin-bottom: 16px; display: flex; align-items: center; gap: 14px; }
.logo-box { background: white; color: #1d4ed8; width: 38px; height: 38px; border-radius: 7px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 14px; flex-shrink: 0; }
.hdr-title { font-size: 17px; font-weight: 700; }
.hdr-sub   { font-size: 12px; opacity: 0.8; margin-top: 2px; }
.live-badge { margin-left: auto; background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.5); padding: 5px 14px; border-radius: 20px; font-size: 12px; font-weight: 700; white-space: nowrap; }
div.row-widget.stRadio > div { flex-direction: row; align-items: center; justify-content: center; background: #f8f9fa; padding: 10px; border-radius: 10px; border: 1px solid #e5e7eb;}
</style>
<div class="main-header">
    <div class="logo-box">ID</div>
    <div>
        <div class="hdr-title">Dashboard Makroekonomi & Pembangunan RI</div>
        <div class="hdr-sub">Multi-Modul: DFM Nasional &middot; Sektor Eksternal &middot; Kewilayahan</div>
    </div>
    <div class="live-badge">&#11044; SIM AKTIF</div>
</div>
""", unsafe_allow_html=True)

main_menu = st.radio(
    "Pilih Modul Analisis:",
    [
        "📊 Makro Nasional (DFM)", 
        "🚢 Analisis Komoditas & Eksternal", 
        "📍 Ekonomi Daerah", 
        "🌍 Analisis Sensitivitas", 
        "🧠 AI Executive Brief (Synthesis)"
    ],
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)
st.divider()

# =========================================================================
# CONTROLLER TABS
# =========================================================================

if main_menu == "📊 Makro Nasional (DFM)":
    
    st.markdown("""
    <style>
    .glass-card { background: rgba(255, 255, 255, 0.65); box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.7); padding: 24px; margin-bottom: 24px; }
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
        
        st.session_state['mac_avg'] = current_avg
        st.session_state['mac_target'] = current_target
        st.session_state['mac_view'] = selected_view

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
            # 1. Trace Bar Realisasi BPS
            fig.add_trace(go.Bar(
                x=final_x, y=final_real, name='Realisasi (BPS)', marker_color='#2980b9', 
                text=[f"{v:.2f}%" if v else "" for v in final_real], 
                textposition='inside', insidetextanchor='middle', textfont=dict(color='white', size=14)
            ))
            
            # --- PENAMBAHAN GARIS BATAS ATAS & BAWAH ---
            std_dev = [0.15, 0.25, 0.35, 0.45] 
            upper_bound = [val + std if pd.notna(val) else None for val, std in zip(final_now, std_dev)]
            lower_bound = [val - std if pd.notna(val) else None for val, std in zip(final_now, std_dev)]
            
            # Trace Batas Atas (Garis Hijau)
            fig.add_trace(go.Scatter(
                x=final_x, y=upper_bound, name='Batas Atas', mode='lines+markers', 
                line=dict(color='#27ae60', width=3, shape='spline'), 
                text=[f"{v:.2f}%" if v is not None else "" for v in upper_bound], textposition='top center'
            ))
            
            # Trace Batas Bawah (Garis Oranye)
            fig.add_trace(go.Scatter(
                x=final_x, y=lower_bound, name='Batas Bawah', mode='lines+markers', 
                line=dict(color='#e67e22', width=3, shape='spline'), 
                text=[f"{v:.2f}%" if v is not None else "" for v in lower_bound], textposition='bottom center'
            ))
            # ----------------------------------------
            
            # Trace Garis Target APBN
            fig.add_trace(go.Scatter(
                x=final_x, y=final_target, name='Target APBN', mode='lines', 
                line=dict(color='#c0392b', width=3, dash='dash')
            ))
            
            fig.update_layout(barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1), height=450)

        # --- RE-FORMATTING LABEL & TRACES ---
        for trace in fig.data:
            trace_name = str(getattr(trace, 'name', ''))
            if "Realisasi" in trace_name and "2010-" in trace_name:
                text_labels, marker_sizes, text_pos = [], [], []
                if trace.x is not None and trace.y is not None:
                    for i, y_val in enumerate(trace.y):
                        if i == len(trace.x) - 1 and pd.notna(y_val): 
                            text_labels.append(f"<b>{float(y_val):.2f}%</b>"); marker_sizes.append(10); text_pos.append("top center") 
                        else:
                            text_labels.append(""); marker_sizes.append(0); text_pos.append("top center")
                trace.update(mode="lines+markers+text", text=text_labels, textposition=text_pos, textfont=dict(size=13, color="#0f172a"))
                if not trace.marker: trace.marker = dict()
                trace.marker.update(size=marker_sizes, symbol="circle", color="#f1c40f", line=dict(width=2, color="white"))
                    
            elif trace_name == 'Proyeksi DFM 2026':
                text_labels, marker_sizes, text_pos = [], [], []
                pos_toggle = True
                if trace.x is not None and trace.y is not None:
                    for i, (x_val, y_val) in enumerate(zip(trace.x, trace.y)):
                        if i > 0 and '2026' in str(x_val) and pd.notna(y_val):
                            text_labels.append(f"<b>{float(y_val):.2f}%</b>"); marker_sizes.append(10); text_pos.append("bottom center" if pos_toggle else "top center")
                            pos_toggle = not pos_toggle
                        else:
                            text_labels.append(""); marker_sizes.append(0); text_pos.append("top center")
                trace.update(mode="lines+markers+text", text=text_labels, textposition=text_pos, textfont=dict(size=13, color="#0f172a"))
                if not trace.marker: trace.marker = dict()
                trace.marker.update(size=marker_sizes, symbol="circle", color="#27ae60", line=dict(width=2, color="white"))
                    
            # Styling untuk Batas Atas dan Batas Bawah
            elif trace_name in ['Batas Atas', 'Batas Bawah']:
                text_labels, marker_sizes = [], []
                if trace.x is not None and trace.y is not None:
                    for y_val in trace.y:
                        if pd.notna(y_val):
                            text_labels.append(f"<b>{float(y_val):.2f}%</b>"); marker_sizes.append(10)
                        else:
                            text_labels.append(""); marker_sizes.append(0)
                
                # Dinamis: Atas posisinya 'top center', Bawah posisinya 'bottom center'
                pos = "top center" if trace_name == 'Batas Atas' else "bottom center"
                color_line = "#27ae60" if trace_name == 'Batas Atas' else "#e67e22"
                
                trace.update(mode="lines+markers+text", text=text_labels, textposition=pos, textfont=dict(size=14, color="#0f172a"))
                if not trace.marker: trace.marker = dict()
                trace.marker.update(size=marker_sizes, symbol="circle", color=color_line, line=dict(width=2, color="white"))

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        
        if not df_full_results.empty:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_full_results.to_excel(writer, index=False, sheet_name='Nowcast Results')
            st.download_button("📥 Download Full Nowcast Results (Excel)", data=buffer.getvalue(), file_name="Replikasi_Final_MATLAB_Elaborated.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 📈 Monitoring Data Harian")
        selected_daily_view = st.radio("Pilih Mode Tampilan Pasar:", ["Data Berjalan", "Data Rata-Rata"], horizontal=True, key="daily_view_toggle")
        
        daily_summary_list, daily_berjalan_list, daily_rata_list = [], [], []
        daily_summary_str = "Data harian tidak tersedia."

        if 'df_daily' in locals() and df_daily is not None:
            group_keuangan = ['IHSG', 'Saham Daily', 'Obligasi Daily']
            group_komoditas = ['Brent', 'WTI', 'CPO', 'Emas', 'Batubara', 'Natural Gas', 'Nikel']
            
            def render_cards(indicators_list):
                cols = st.columns(4)
                idx = 0
                for col in indicators_list:
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
                    with cols[idx % 4]: st.markdown(html, unsafe_allow_html=True)
                    idx += 1

            st.markdown("##### 🏦 Pasar Keuangan")
            render_cards(group_keuangan)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("##### 🛢️ Komoditas")
            render_cards(group_komoditas)
                
            if daily_summary_list: daily_summary_str = " | ".join(daily_summary_list)
            
        st.session_state['mac_daily'] = daily_summary_str
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 🔍 Deep Dive: Indikator Makro Bulanan")
        df_makro['Tanggal'] = pd.to_datetime(df_makro['Tanggal'])
        df_makro = df_makro.sort_values(by='Tanggal')
        
        indicator_cols = [c for c in df_makro.columns if c != 'Tanggal']
        
        monthly_summary_list = [] 
        monthly_summary_str = "Data bulanan tidak tersedia."
        
        group_riil = ['PMI Manufaktur Negara Berkembang', 'Penjualan Mobil', 'Penjualan Motor', 'Penjualan semen', 'Indeks Keyakinan Konsumen', 'Inflasi']
        group_eksternal = ['Ekspor Barang', 'Impor Barang Konsumsi', 'Impor Bahan Baku', 'Impor Barang Modal']
        group_moneter = ['Jumlah Uang Yang Beredar', 'Kredit Perbankan', 'Suku Bunga', 'Nilai Tukar terhadap Dolar AS']
        
        def render_monthly_cards(indicators_list):
            cols = st.columns(4)
            idx = 0
            for col in indicators_list:
                if col not in df_makro.columns: continue
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
                with cols[idx%4]: st.markdown(html, unsafe_allow_html=True)
                idx += 1

        st.markdown("##### 🏭 Sektor Riil & Daya Beli")
        render_monthly_cards(group_riil)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("##### 🚢 Sektor Eksternal (Ekspor & Impor)")
        render_monthly_cards(group_eksternal)

        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("##### 🏦 Sektor Moneter & Keuangan")
        render_monthly_cards(group_moneter)
        
        all_grouped = set(group_riil + group_eksternal + group_moneter)
        remaining_cols = [c for c in df_makro.columns if c != 'Tanggal' and c not in all_grouped]
        if remaining_cols:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📌 Indikator Lainnya")
            render_monthly_cards(remaining_cols)
            
        if monthly_summary_list: monthly_summary_str = "\n".join(monthly_summary_list)
        st.session_state['mac_monthly'] = monthly_summary_str

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
            st.session_state['mac_heat'] = heatmap_summary_str
                
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
# MODUL 2: ANALISIS KOMODITAS & EKSTERNAL
# =========================================================================
elif main_menu == "🚢 Analisis Komoditas & Eksternal":
    with st.sidebar:
        st.markdown("### 🔍 PARAMETER INTELIJEN")
        f_tahun_b = st.selectbox("Tahun Data", reversed(TAHUN_TERSEDIA_BPS), index=1, key="exim_yr")
        f_per_label = st.selectbox("Periode Waktu", list(PERIODE_OPSI.keys()), key="exim_per")
        f_per = PERIODE_OPSI[f_per_label]
        f_smb_label = st.radio("Arah Dagang", ["Ekspor", "Impor"], key="exim_dir")
        f_smb = "1" if f_smb_label == "Ekspor" else "2"
        f_unt = st.radio("Skala Unit", ["USD", "Miliar USD"], key="exim_unit")
        
        tipe_b, bulan_b = get_periode_params(f_per)
        df_exim_raw = fetch_bps_db(f_smb, str(f_tahun_b), tipe_b, bulan_b)
        if not df_exim_raw.empty:
            st.session_state['exim_data_state'] = df_exim_raw
            st.session_state['exim_meta_state'] = {"tahun": f_tahun_b, "sumber": f_smb_label, "sumber_kode": f_smb, "unit": f_unt, "tipe": tipe_b, "bulan": bulan_b}
        else:
            st.session_state['exim_data_state'] = pd.DataFrame()
            st.session_state['exim_meta_state'] = {}

    sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
        "📊 Ringkasan BPS", "🗄️ Data Lengkap", "⚠️ Early Warning System", "🪞 Mirroring", "🏦 Neraca Pembayaran"
    ])
    
    df_m3 = st.session_state.get('exim_data_state', pd.DataFrame())
    m3_meta = st.session_state.get('exim_meta_state', {})
    
    if not df_m3.empty:
        div_m3 = 1e9 if m3_meta['unit'] == "Miliar USD" else 1
        df_m3_c = df_m3.copy()
        df_m3_c["value"] = df_m3_c["value"] / div_m3
        
        kmd_m3 = df_m3_c.groupby("kodehs", as_index=False)[["value","berat"]].sum().sort_values("value", ascending=False)
        neg_m3 = df_m3_c.groupby("negara", as_index=False)["value"].sum().sort_values("value", ascending=False)
        kmd_m3["deskripsi"] = kmd_m3["kodehs"].map(HS_DESC).fillna("Lainnya")
        kmd_m3["label"] = kmd_m3["kodehs"].astype(str) + " - " + kmd_m3["deskripsi"].str[:15]

    with sub_tab1:
        if df_m3.empty:
            st.info("👈 Silakan konfigurasikan parameter Intelijen Perdagangan di sidebar kiri.")
        else:
            sk1, sk2, sk3 = st.columns(3)
            c_g = "#3fb950" if m3_meta['sumber'] == "Ekspor" else "#f78166"
            with sk1: kpi_card(f"TOTAL {m3_meta['sumber'].upper()}", f"{df_m3_c['value'].sum():,.2f} {m3_meta['unit']}", c_g)
            with sk2: kpi_card("KOMODITAS UTAMA (HS)", kmd_m3.iloc[0]["kodehs"] if not kmd_m3.empty else "-", "#58a6ff")
            with sk3: kpi_card("MITRA STRATEGIS UTAMA", neg_m3.iloc[0]["negara"] if not neg_m3.empty else "-", "#e3b341")
            
            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
            
            ml_l, ml_r = st.columns(2)
            with ml_l:
                section_title("ANALISIS TREN HISTORIS")
                h_c1, h_c2, h_c3, h_c4 = st.columns([2,2,2,2])
                hs_h = h_c1.selectbox("HS Code", options=HS_ALL, index=26, key="m3_hs_sel", label_visibility="collapsed")
                neg_h = h_c2.selectbox("Mitra", options=["Semua Negara"] + PARTNER_LIST, key="m3_neg_sel", label_visibility="collapsed")
                met_h = h_c3.radio("Metrik", ["Nilai", "YoY %"], horizontal=True, key="m3_met_sel", label_visibility="collapsed")
                btn_h = h_c4.button("Tampilkan Histori", use_container_width=True, key="m3_btn_hist")
                
                if btn_h:
                    with st.spinner("Menarik data historis..."):
                        df_h_raw = fetch_hist_bps_db(m3_meta['sumber_kode'], hs_h, m3_meta['tipe'], m3_meta['bulan'])
                        if df_h_raw.empty:
                            st.warning("Tidak ada data historis.")
                        else:
                            if neg_h != "Semua Negara": df_h_raw = df_h_raw[df_h_raw["negara"] == normalize_negara(neg_h)]
                            df_h_g = df_h_raw.groupby("tahun", as_index=False)["value"].sum().sort_values("tahun")
                            df_h_g["Tahun"], df_h_g["Value"] = df_h_g["tahun"].astype(str), df_h_g["value"] / div_m3
                            if met_h == "YoY %":
                                df_h_g["Value"] = df_h_g["Value"].pct_change() * 100
                                fig_m3_h = px.line(df_h_g, x="Tahun", y="Value", markers=True, title=f"YoY (%) - HS {hs_h}")
                            else:
                                fig_m3_h = px.line(df_h_g, x="Tahun", y="Value", markers=True, title=f"Tren Nilai ({m3_meta['unit']}) - HS {hs_h}")
                            fig_m3_h.update_layout(xaxis=dict(type='category'), margin=dict(l=0,r=0,t=30,b=0))
                            st.plotly_chart(fig_m3_h, use_container_width=True)
            with ml_r:
                section_title("STRUKTUR KOMODITAS UNGGULAN (TOP 15)")
                fig_k3 = px.bar(kmd_m3.head(15), y="label", x="value", orientation='h')
                fig_k3.update_yaxes(categoryorder='total ascending', type='category')
                fig_k3.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=350)
                fig_k3.update_traces(marker_color=c_g)
                st.plotly_chart(fig_k3, use_container_width=True)

            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
            
            # Row 3: Top Negara & Share Pie
            r2_left, r2_right = st.columns(2)
            with r2_left:
                section_title("TOP NEGARA MITRA")
                fig_neg = px.bar(neg_m3.head(15), y="negara", x="value", orientation='h')
                fig_neg.update_yaxes(categoryorder='total ascending', type='category')
                fig_neg.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=350)
                fig_neg.update_traces(marker_color="#58a6ff")
                st.plotly_chart(fig_neg, use_container_width=True)
                
            with r2_right:
                section_title("SHARE KOMODITAS (TOP 8)")
                fig_pie = px.pie(kmd_m3.head(8), values='value', names='label', hole=0.45)
                fig_pie.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=350, legend=dict(orientation="v"))
                st.plotly_chart(fig_pie, use_container_width=True)

    with sub_tab2:
        if not df_m3.empty:
            section_title("TABEL PENUH DATA PERDAGANGAN BPS")
            full_m3 = df_m3_c.groupby(["negara","kodehs"], as_index=False)[["value","berat"]].sum()
            full_m3["deskripsi"] = full_m3["kodehs"].map(HS_DESC).fillna("Lainnya")
            st.download_button(label="Download CSV Table", data=full_m3.to_csv(index=False).encode('utf-8'), file_name=f"BPS_Intel_{m3_meta['sumber']}.csv", mime='text/csv')
            st.dataframe(full_m3, use_container_width=True, hide_index=True)

    with sub_tab3:
        if not df_m3.empty:
            section_title("DETEKSI STRUKTUR ANOMALI & EW_METRICS")
            ews_m3 = calculate_ews(kmd_m3.copy())
            def style_ews_m3(val):
                c = '#ffcccb' if 'Atas' in str(val) else '#ffe8b5' if 'Bawah' in str(val) else '#dcbdfb' if 'Harga' in str(val) else '#ff7b72' if 'KRITIS' in str(val) else ''
                return f'background-color: {c}; color: black' if c else ''
            st.dataframe(ews_m3[[col for col in ews_m3.columns if col != "label"]].style.map(style_ews_m3, subset=['status_ews']), use_container_width=True, hide_index=True)

    with sub_tab4:
        section_title("KOMPARASI ASIMETRI REKONSILIASI KEBOCORAN / MIRRORING")
        if not df_m3.empty:
            cm_c1, cm_c2, cm_c3 = st.columns([2,2,1])
            m_mitra = cm_c1.selectbox("Mitra Rekonsiliasi", PARTNER_LIST, key="m3_mirror_mitra")
            m_unit = cm_c2.radio("Satuan Skala", ["USD", "Juta USD"], horizontal=True, key="m3_mirror_unit")
            btn_m = cm_c3.button("JALANKAN AUDIT ASIMETRI", type="primary", use_container_width=True, key="m3_run_mirror")
            
            if btn_m:
                div_m = 1e6 if m_unit == "Juta USD" else 1
                df_tm, status = load_trademap(m_mitra, m3_meta['tahun'], m3_meta['sumber_kode'])
                if status == "SUCCESS":
                    df_bps_m = df_m3[df_m3["negara"] == normalize_negara(m_mitra)].copy()
                    df_bps_m = df_bps_m.groupby("kodehs", as_index=False)["value"].sum().rename(columns={"kodehs":"HS","value":"BPS_Value"})
                    df_merge = pd.merge(df_bps_m, df_tm, on="HS", how="outer").fillna(0)
                    df_merge[["BPS_Value", "Trademap_Value"]] /= div_m
                    df_merge["Selisih"] = df_merge["Trademap_Value"] - df_merge["BPS_Value"]
                    df_merge["Deskripsi"] = df_merge["HS"].map(HS_DESC).fillna("Lainnya")
                    
                    st.success("✅ Audit Asimetri Pencatatan Trade Map Selesai.")
                    lbl_b = "EKSPOR IDN KE" if str(m3_meta['sumber_kode']) == "1" else "IMPOR IDN DARI"
                    lbl_t = "IMPOR MITRA (TRADE MAP)" if str(m3_meta['sumber_kode']) == "1" else "EKSPOR MITRA (TRADE MAP)"
                    
                    kpx1, kpx2, kpx3 = st.columns(3)
                    with kpx1: kpi_card(f"{lbl_b} {m_mitra.upper()} (BPS)", f"{df_merge['BPS_Value'].sum():,.1f} {m_unit}", "#3fb950")
                    with kpx2: kpi_card(f"{lbl_t}", f"{df_merge['Trademap_Value'].sum():,.1f} {m_unit}", "#58a6ff")
                    with kpx3: kpi_card("GAP KEBOCORAN / ASIMETRI PENCATATAN", f"{df_merge['Selisih'].sum():,.1f} {m_unit}", "#e3b341")
                    
                    st.dataframe(df_merge[["HS", "Deskripsi", "BPS_Value", "Trademap_Value", "Selisih"]], use_container_width=True, hide_index=True)

    with sub_tab5:
        if not os.path.exists(BOP_DB_PATH): 
            st.error("Database bop_indonesia.db tidak tersedia.")
        else:
            section_title("SEKI BANK INDONESIA (BALANCE OF PAYMENTS HISTORICAL)")
            with st.form("m3_seki_form"):
                cx1, cx2, cx3, cx4, cx5 = st.columns([1,1,1,1,1])
                s_y1 = cx1.number_input("Tahun Awal", min_value=2004, max_value=2025, value=2015, key="m3_bop_y1")
                s_y2 = cx2.number_input("Tahun Akhir", min_value=2004, max_value=2025, value=2024, key="m3_bop_y2")
                s_frq = cx3.selectbox("Frekuensi", ["Kuartalan", "Tahunan"], key="m3_bop_frq")
                s_uni = cx4.selectbox("Satuan", ["Juta USD", "Miliar USD"], key="m3_bop_uni")
                cx5.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                btn_s = cx5.form_submit_button("▶ TAMPILKAN NERACA", use_container_width=True)
                
            if btn_s:
                f_val = "quarterly" if s_frq == "Kuartalan" else "annual"
                div_s = 1000 if s_uni == "Miliar USD" else 1
                needed_ids = [1,2,17,20,23,26,29,32,35,40,41,46,47,48,54,55,56]
                df_seki = bop_series(needed_ids, s_y1, s_y2, f_val)
                
                if not df_seki.empty:
                    df_seki["nilai"] = df_seki["value_mn_usd"] / div_s
                    
                    # Row 1: KPI Cards
                    ck1, ck2, ck3 = st.columns(3)
                    ca_v, cad_v, ner_v, _BOP_LATEST = bop_latest_val(1), bop_latest_val(54), bop_latest_val(48), bop_latest()
                    with ck1: kpi_card("TRANSAKSI BERJALAN", f"{(ca_v/div_s if ca_v else 0):,.1f} {s_uni}", "#3fb950" if (ca_v or 0) >= 0 else "#f78166", f"Periode: {_BOP_LATEST}")
                    with ck2: kpi_card("CADANGAN DEVISA", f"{(cad_v/div_s if cad_v else 0):,.1f} {s_uni}", "#58a6ff", f"Periode: {_BOP_LATEST}")
                    with ck3: kpi_card("NERACA KESELURUHAN", f"{(ner_v/div_s if ner_v else 0):,.1f} {s_uni}", "#bc8cff", f"Periode: {_BOP_LATEST}")

                    def gs(iid):
                        s = df_seki[df_seki["item_id"] == iid].copy()
                        s = s.sort_values("year" if f_val == "annual" else ["year","quarter"])
                        s["v"] = s["nilai"]
                        return s
                    xcol = "period" if f_val == "quarterly" else "year"

                    st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

                    # Row 2: CA Trends & Waterfall
                    cc1, cc2 = st.columns([3, 2])
                    with cc1:
                        section_title("TREN TRANSAKSI BERJALAN & KOMPONEN")
                        ca_df = df_seki[df_seki['item_id'].isin([2, 17, 20, 23, 1])] 
                        fig_ca = px.bar(ca_df[ca_df['item_id']!=1], x=xcol, y="nilai", color="keterangan", barmode="relative")
                        fig_ca.add_scatter(x=ca_df[ca_df['item_id']==1][xcol], y=ca_df[ca_df['item_id']==1]["nilai"], name="Total CA", line=dict(color="#000000", width=2))
                        fig_ca.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=1.1, title=""), xaxis=dict(type='category'))
                        st.plotly_chart(fig_ca, use_container_width=True)
                    with cc2:
                        section_title("DEKOMPOSISI NERACA (TOTAL PERIODE)")
                        wf_ids = [1, 26, 29, 47, 48]
                        wf_lbl = ["Transaksi<br>Berjalan","Transaksi<br>Modal","Transaksi<br>Finansial","Selisih<br>Perhitungan","Neraca<br>Keseluruhan"]
                        wf_v = [float(gs(iid)["v"].sum()) if not gs(iid).empty else 0 for iid in wf_ids]
                        fig_wf = go.Figure(go.Waterfall(x=wf_lbl, measure=["relative","relative","relative","relative","total"], y=wf_v, textposition="outside", decreasing=dict(marker_color="#f78166"), increasing=dict(marker_color="#3fb950"), totals=dict(marker_color="#bc8cff")))
                        fig_wf.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                        st.plotly_chart(fig_wf, use_container_width=True)

                    # Row 3: Financial & Reserves
                    cd1, cd2 = st.columns(2)
                    with cd1:
                        section_title("TRANSAKSI FINANSIAL: KOMPONEN INVESTASI")
                        fig_inv = go.Figure()
                        for iid, lbl_i, c in [(32,"FDI","#3fb950"),(35,"Portofolio","#58a6ff"),(41,"Lainnya","#ffa657"),(40,"Derivatif","#e3b341")]:
                            s = gs(iid)
                            if not s.empty: fig_inv.add_trace(go.Bar(x=s[xcol], y=s["v"], name=lbl_i, marker_color=c))
                        s_fin = gs(29)
                        if not s_fin.empty: fig_inv.add_trace(go.Scatter(x=s_fin[xcol], y=s_fin["v"], name="Total Fin.", line=dict(color="#bc8cff", width=2, dash="dot")))
                        fig_inv.update_layout(barmode="relative", height=320, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=1.1), xaxis=dict(type='category'))
                        st.plotly_chart(fig_inv, use_container_width=True)
                    with cd2:
                        section_title("CADANGAN DEVISA")
                        cad_df = df_seki[df_seki['item_id'] == 54]
                        fig_cad = px.area(cad_df, x=xcol, y="nilai")
                        fig_cad.update_traces(line_color="#39d0d8", fillcolor="rgba(57,208,216,0.1)")
                        fig_cad.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), xaxis=dict(type='category'))
                        st.plotly_chart(fig_cad, use_container_width=True)

                    # Row 4: Custom Compare & CA%
                    ce1, ce2 = st.columns(2)
                    with ce1:
                        section_title("KOMPARASI INDIKATOR")
                        seki_cmp = st.multiselect("Pilih Indikator:", options=list(BOP_MAIN_ITEMS.values()), default=["Transaksi Finansial", "Neraca Keseluruhan"], label_visibility="collapsed", key="bop_cmp")
                        inv_map = {v: k for k, v in BOP_MAIN_ITEMS.items()}
                        fig_cmp = go.Figure()
                        pal = ["#58a6ff", "#3fb950", "#f78166", "#e3b341", "#bc8cff"]
                        for i, ind in enumerate(seki_cmp):
                            s = gs(inv_map[ind])
                            if not s.empty: fig_cmp.add_trace(go.Scatter(x=s[xcol], y=s["v"], name=ind, mode="lines+markers", line=dict(color=pal[i%len(pal)], width=2)))
                        fig_cmp.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=1.1), xaxis=dict(type='category'))
                        st.plotly_chart(fig_cmp, use_container_width=True)
                    with ce2:
                        section_title("CURRENT ACCOUNT % PDB")
                        s_pct = df_seki[df_seki["item_id"] == 56].copy().sort_values("year" if f_val == "annual" else ["year","quarter"])
                        fig_pct = go.Figure()
                        if not s_pct.empty:
                            fig_pct.add_trace(go.Bar(x=s_pct[xcol], y=s_pct["value_mn_usd"], marker_color=["#3fb950" if v >= 0 else "#f78166" for v in s_pct["value_mn_usd"]]))
                        fig_pct.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), xaxis=dict(type='category'))
                        st.plotly_chart(fig_pct, use_container_width=True)

                    st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
                    section_title("DATA LENGKAP NERACA PEMBAYARAN")
                    
                    seki_tbl_filter = st.multiselect("Filter Indikator:", options=list(BOP_MAIN_ITEMS.values()), default=None, label_visibility="collapsed", key="bop_tbl_flt")
                    df_seki_table = df_seki.copy()
                    if seki_tbl_filter:
                        df_seki_table = df_seki_table[df_seki_table["keterangan"].isin(seki_tbl_filter)]
                    
                    col_dl, col_spc = st.columns([1, 4])
                    with col_dl:
                        csv_seki = df_seki_table[["year", "period", "keterangan", "items_en", "nilai"]].to_csv(index=False).encode('utf-8')
                        st.download_button("⬇ Download CSV", data=csv_seki, file_name=f"SEKI_{s_y1}_{s_y2}.csv", mime='text/csv', use_container_width=True)
                    
                    st.dataframe(df_seki_table[["year", "period", "keterangan", "items_en", "nilai"]], use_container_width=True, hide_index=True)
                else:
                    st.warning("Tidak ada data SEKI untuk rentang waktu tersebut.")

# =========================================================================
# MODUL 3: EKONOMI DAERAH (WIP)
# =========================================================================
elif "Ekonomi Daerah" in main_menu:

    # ==============================================================================
    # FUNGSI LOKAL DAERAH
    # ==============================================================================
    def smart_load_daerah(filename_base):
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        formats = ['.xlsx', '.csv'] 
        folders = [base_dir, os.path.join(base_dir, 'data')]
        
        for fldr in folders:
            for fmt in formats:
                path = os.path.join(fldr, f"{filename_base}{fmt}")
                if os.path.exists(path):
                    try:
                        if fmt == '.xlsx':
                            df = pd.read_excel(path, engine='openpyxl')
                        else:
                            try:
                                df = pd.read_csv(path, sep=";", encoding='cp1252', engine='python')
                                if len(df.columns) < 2:
                                    df = pd.read_csv(path, sep=",", encoding='cp1252', engine='python')
                            except:
                                df = pd.read_csv(path, sep=",", encoding='cp1252', engine='python')
                        
                        df.columns = df.columns.astype(str).str.strip().str.lower()
                        return df
                    except Exception:
                        continue
        return pd.DataFrame()

    def load_data_aman_daerah(provinsi, tahun):
        df_all = smart_load_daerah("data_ekonomi")
        if df_all is None or df_all.empty:
            return pd.DataFrame(columns=['provinsi', 'tahun', 'klasifikasi', 'lpe_tw1', 'lpe_tw2', 'lpe_tw3', 'lpe_tw4', 'lpe_ctc'])
        try:
            df_all['provinsi'] = df_all['provinsi'].astype(str).str.strip()
            df_all['tahun'] = pd.to_numeric(df_all['tahun'], errors='coerce').fillna(0).astype(int)
            kolom_angka = ['lpe_tw1', 'lpe_tw2', 'lpe_tw3', 'lpe_tw4', 'lpe_ctc', 'kontribusi', 'pdrb_perkapita', 'inflasi', 'pma', 'pmdn', 'ipm', 'kemiskinan', 'tpt', 'gini']
            for kol in kolom_angka:
                if kol in df_all.columns:
                    df_all[kol] = df_all[kol].astype(str).str.strip().replace(['-', '', 'nan', 'none', 'None'], np.nan)
                    df_all[kol] = df_all[kol].str.replace(',', '.', regex=False)
                    df_all[kol] = pd.to_numeric(df_all[kol], errors='coerce')
            df_filtered = df_all[df_all['tahun'] == int(tahun)]
            return df_filtered.reset_index(drop=True) if not df_filtered.empty else pd.DataFrame(columns=df_all.columns)
        except Exception:
            return pd.DataFrame(columns=['provinsi', 'tahun', 'klasifikasi', 'lpe_tw1', 'lpe_tw2', 'lpe_tw3', 'lpe_tw4', 'lpe_ctc'])

    def load_data_sektoral_aman_daerah(provinsi):
        df_sektoral = smart_load_daerah("data_sektoral")
        if df_sektoral is None or df_sektoral.empty: return pd.DataFrame()
        try:
            df_sektoral['provinsi'] = df_sektoral['provinsi'].astype(str).str.strip()
            df_filtered = df_sektoral[df_sektoral['provinsi'] == str(provinsi).strip()]
            return df_filtered.reset_index(drop=True) if not df_filtered.empty else pd.DataFrame(columns=df_sektoral.columns)
        except:
            return pd.DataFrame()

    def load_data_struktur_aman_daerah(provinsi):
        df_all = smart_load_daerah("data_struktur")
        if df_all is None or df_all.empty: return pd.DataFrame()
        try:
            df_all['provinsi'] = df_all['provinsi'].astype(str).str.strip()
            df_filtered = df_all[df_all['provinsi'] == str(provinsi).strip()]
            return df_filtered.reset_index(drop=True) if not df_filtered.empty else pd.DataFrame(columns=df_all.columns)
        except:
            return pd.DataFrame()

    WARNA_SEKTOR_GLOBAL = {
        "pertanian": "#22C55E", "pertambangan": "#D97706", "industri": "#6B21A8",
        "pengadaan listrik": "#EA580C", "pengadaan air": "#1E3A1E", "konstruksi": "#8B5CF6",
        "perdagangan": "#1D4ED8", "transportasi": "#FBBF24", "akmamin": "#F472B6",
        "informasi dan komunikasi": "#3B82F6", "jasa keuangan": "#EC4899", "real estat": "#6B7280",
        "jasa perusahaan": "#9CA3AF", "adm. pemerintahan": "#DC2626", "jasa pendidikan": "#0D9488",
        "jasa kesehatan": "#78350F", "jasa lainnya": "#D97706"
    }

    def get_warna_sektor_map(df_column):
        return {sektor: WARNA_SEKTOR_GLOBAL.get(str(sektor).lower(), "#6B7280") for sektor in df_column.unique()}

    def buat_bar_chart_makro(df_aktif, tipe_chart, provinsi_aktif=None):
        if df_aktif is None or df_aktif.empty:
            st.warning("Data makro untuk grafik batang kosong.")
            return

        if tipe_chart == "Pertumbuhan Ekonomi":
            if "lpe_ctc" not in df_aktif.columns: return st.warning("Kolom lpe_ctc tidak ditemukan.")
            kolom_nilai = "lpe_ctc"
            label_x = "LPE c-to-c (%)"
            skala_warna = "Viridis"
        else:
            if "kontribusi" not in df_aktif.columns: return st.warning("Kolom kontribusi tidak ditemukan.")
            kolom_nilai = "kontribusi"
            label_x = "Kontribusi PDRB (%)"
            skala_warna = "Cividis"

        df_sorted = df_aktif.dropna(subset=[kolom_nilai]).sort_values(by=kolom_nilai, ascending=True).copy()
        
        df_sorted['label_provinsi'] = df_sorted['provinsi'].apply(
            lambda x: f"<b>📍 {x}</b>" if str(x).strip().lower() == str(provinsi_aktif).strip().lower() else x
        )
        
        warna_kustom = []
        for idx, row in df_sorted.iterrows():
            if str(row['provinsi']).strip().lower() == str(provinsi_aktif).strip().lower():
                warna_kustom.append("#EF4444") 
            else:
                warna_kustom.append(row[kolom_nilai])

        ada_terpilih = df_sorted['provinsi'].str.strip().str.lower().eq(str(provinsi_aktif).strip().lower()).any()

        if ada_terpilih:
            fig = go.Figure(go.Bar(
                x=df_sorted[kolom_nilai],
                y=df_sorted['label_provinsi'],
                orientation='h',
                marker=dict(
                    color=warna_kustom, colorscale=skala_warna, showscale=True,
                    colorbar=dict(title=label_x, thickness=15, len=0.4, yanchor="middle", y=0.5, outlinewidth=0, ticks="", tickfont=dict(size=12)),
                    line=dict(width=0, color='rgba(0,0,0,0)')
                ),
                text=df_sorted[kolom_nilai], 
                texttemplate='%{text:.1f}', 
                textposition='outside',
                textfont=dict(size=14, color='#1E293B') # <-- PERBESAR UKURAN ANGKA DI LUAR BAR
            ))
            fig.update_layout(
                xaxis_title=label_x, 
                yaxis_title="Provinsi",
                yaxis=dict(tickfont=dict(size=13)), # <-- PERBESAR NAMA PROVINSI
                xaxis=dict(tickfont=dict(size=13))  # <-- PERBESAR ANGKA SUMBU BAWAH
            )
        else:
            fig = px.bar(df_sorted, x=kolom_nilai, y="label_provinsi", orientation='h', labels={kolom_nilai: label_x, "label_provinsi": "Provinsi"}, color=kolom_nilai, color_continuous_scale=skala_warna)
            fig.update_traces(texttemplate='%{x:.1f}', textposition='outside', textfont_size=14) # <-- PERBESAR ANGKA
            fig.update_layout(
                yaxis=dict(tickfont=dict(size=13)), 
                xaxis=dict(tickfont=dict(size=13))
            )
            
        fig.update_layout(height=800, margin={"r":40,"t":10,"l":10,"b":10}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    def buat_peta_klasifikasi(df_aktif, provinsi_aktif=None):
        if df_aktif is None or df_aktif.empty or "klasifikasi" not in df_aktif.columns:
            st.warning("Data kosong atau kolom klasifikasi tidak ditemukan, peta tidak dapat dimuat.")
            return
            
        try:
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            geojson_path1 = os.path.join(base_dir, "indonesia_provinces.geojson")
            geojson_path2 = os.path.join(base_dir, "data", "indonesia_provinces.geojson")
            
            geojson_path = geojson_path1 if os.path.exists(geojson_path1) else geojson_path2
            
            if not os.path.exists(geojson_path):
                st.info("🗺️ *[File GeoJSON batas wilayah tidak ditemukan]*")
                return
                
            with open(geojson_path, "r") as f:
                geojson_indonesia = json.load(f)
                
            df_peta = df_aktif.copy()
            df_peta['provinsi'] = df_peta['provinsi'].replace({"DI Yogyakarta": "Daerah Istimewa Yogyakarta", "D.I. Yogyakarta": "Daerah Istimewa Yogyakarta"})
                
            fig = px.choropleth_mapbox(
                df_peta, geojson=geojson_indonesia, locations="provinsi", featureidkey="properties.PROVINSI", color="klasifikasi",                 
                color_discrete_map={"Daerah Maju dan Cepat Tumbuh": "#031926", "Daerah Berkembang Cepat": "#468189", "Daerah Maju tapi Tertekan": "#9DBEBB", "Daerah Relatif Tertinggal": "#F4E9CD"},
                mapbox_style="carto-positron", center={"lat": -2.5, "lon": 118.0}, zoom=3.5, opacity=0.8, labels={"klasifikasi": "<b>Status Klasifikasi</b>"}
            )
            
            if provinsi_aktif:
                nama_cari = "Daerah Istimewa Yogyakarta" if provinsi_aktif in ["DI Yogyakarta", "D.I. Yogyakarta"] else provinsi_aktif
                lat_center, lon_center = None, None
                for feature in geojson_indonesia['features']:
                    if feature['properties']['PROVINSI'].strip().lower() == nama_cari.strip().lower():
                        coords = feature['geometry']['coordinates']
                        if feature['geometry']['type'] == 'MultiPolygon': sample_point = coords[0][0][0]
                        else: sample_point = coords[0][0]
                        lon_center, lat_center = sample_point[0], sample_point[1]
                        break
                
                if lat_center is not None and lon_center is not None:
                    fig.add_trace(go.Scattermapbox(
                        lat=[lat_center], lon=[lon_center], mode='markers+text',
                        marker=dict(size=35, color='rgba(0,0,0,0)'), text=["📍"], textposition="top center",
                        textfont=dict(size=24), showlegend=False, hoverinfo='none'
                    ))
            
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=450)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"🗺️ *[Gagal memuat peta GeoJSON. Error: {e}]*")

    def buat_line_growth(provinsi):
        df_raw = smart_load_daerah("data_ekonomi")
        if df_raw is None or df_raw.empty: return
        try:
            df_raw['provinsi'] = df_raw['provinsi'].astype(str).str.strip()
            df_prov = df_raw[df_raw['provinsi'] == provinsi].sort_values(by="tahun")
            if df_prov.empty:
                st.warning(f"Data tren historis untuk {provinsi} tidak ditemukan.")
                return
            df_prov['lpe_ctc'] = pd.to_numeric(df_prov['lpe_ctc'].astype(str).str.replace(',', '.', regex=False).str.strip().replace('-', np.nan), errors='coerce')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_prov['tahun'], y=df_prov['lpe_ctc'], name=f"{provinsi}", mode='lines+markers+text',
                text=df_prov['lpe_ctc'].round(1), textposition='top center', 
                textfont=dict(color='#1E3A8A', size=14), # <-- PERBESAR ANGKA LABEL BIRU
                texttemplate='<span style="background-color: #E0F2FE; padding: 2px 4px; border-radius: 3px;">%{text:.1f}</span>',
                line=dict(width=3, color='#1D4ED8', shape='spline'), marker=dict(size=8)
            ))
            
            if 'lpe_nasional' in df_prov.columns:
                df_prov['lpe_nasional'] = pd.to_numeric(df_prov['lpe_nasional'].astype(str).str.replace(',', '.', regex=False).str.strip().replace('-', np.nan), errors='coerce')
                fig.add_trace(go.Scatter(
                    x=df_prov['tahun'], y=df_prov['lpe_nasional'], name='Nasional          Catatan: Data tahun 2026 bersifat sementara (c-to-c)', 
                    mode='lines+markers+text', text=df_prov['lpe_nasional'].round(1), textposition='top center',
                    textfont=dict(color='#7F1D1D', size=14), # <-- PERBESAR ANGKA LABEL MERAH
                    texttemplate='<span style="background-color: #FFE4E6; padding: 2px 4px; border-radius: 3px;">%{text:.1f}</span>',
                    line=dict(dash='dash', width=2.5, color='#DC2626', shape='spline'), marker=dict(size=8)
                ))
            
            fig.update_layout(
                xaxis=dict(dtick=1, type='category', tickfont=dict(size=13)), # <-- PERBESAR ANGKA TAHUN
                yaxis=dict(tickformat='.1f', tickfont=dict(size=13)),       # <-- PERBESAR ANGKA PERSEN Y
                xaxis_title="Tahun", 
                yaxis_title="Pertumbuhan Ekonomi (%)", 
                margin={"r":10,"t":30,"l":10,"b":10}, 
                legend=dict(orientation="h", font=dict(size=13))            # <-- PERBESAR TEKS LEGENDA
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal memuat tren pertumbuhan makro historis: {e}")

    def buat_area_struktur(df_aktif):
        if df_aktif is not None and not df_aktif.empty and 'sektor' in df_aktif.columns and 'kontribusi_sektor' in df_aktif.columns:
            df_display = df_aktif.sort_values(by="tahun")
            warna_map = get_warna_sektor_map(df_display['sektor'])
            fig = px.area(df_display, x="tahun", y="kontribusi_sektor", color="sektor", line_group="sektor", color_discrete_map=warna_map, labels={"tahun": "Tahun Analisis", "kontribusi_sektor": "Kontribusi Sektor PDRB (%)"})
            fig.update_layout(showlegend=True, xaxis=dict(dtick=1, type='category'), margin={"r": 10, "t": 10, "l": 10, "b": 10})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 *[Grafik Tren Area Struktur Ekonomi belum dapat dimuat karena data kosong]*")

    def buat_scatter_sektoral(df_aktif, jenis_analisis):
        if df_aktif is None or df_aktif.empty:
            st.info(f"🎯 *[Grafik Scatter Plot {jenis_analisis} akan muncul otomatis setelah data sektoral provinsi termuat]*")
            return

        if jenis_analisis == "Overlay":
            judul_full = 'Scatter Plot "Overlay (MRP - LQ) 2025"'
            help_teks = "Metode Overlay merupakan teknik yang menggabungkan hasil analisis Location Quotient (LQ) dan Model Rasio Pertumbuhan (MRP) untuk mengidentifikasi sektor yang memiliki keunggulan sekaligus pertumbuhan yang kuat. Dengan mengombinasikan kedua pendekatan tersebut, metode ini menghasilkan penentuan sektor prioritas yang lebih robust dibandingkan penggunaan satu metode secara terpisah."
            kriteria_teks = (
                "- **Kriteria I (Rasio Pertumbuhan > 1 dan LQ > 1):** Sektor Unggulan dan Dominan $\\rightarrow$ sektor dengan pertumbuhan tinggi dan kontribusi besar yang menjadi motor utama perekonomian daerah.\n\n"
                "- **Kriteria II (Rasio Pertumbuhan > 1 dan LQ < 1):** Sektor Berkembang $\\rightarrow$ sektor dengan pertumbuhan tinggi namun kontribusinya masih kecil, sehingga berpotensi menjadi sumber pertumbuhan baru.\n\n"
                "- **Kriteria III (Rasio Pertumbuhan < 1 dan LQ > 1):** Sektor Potensial $\\rightarrow$ sektor dengan kontribusi besar tetapi pertumbuhannya mulai melambat, sehingga perlu dijaga keberlanjutannya.\n\n"
                "- **Kriteria IV (Rasio Pertumbuhan < 1 dan LQ < 1):** Sektor Tertinggal $\\rightarrow$ sektor dengan pertumbuhan dan kontribusi yang rendah, sehingga belum memiliki peran signifikan dalam perekonomian."
            )
            col_x, col_y = "lq_2025", "rps_2025"
            garis_x, garis_y = 1.0, 1.0  
            labels_x, labels_y = "Location Quotient (LQ)", "Rasio Pertumbuhan Sektoral (RPS)"

        elif jenis_analisis == "Shift Share":
            judul_full = 'Scatter Plot "Shift Share 2015/2025"'
            help_teks = "Metode Shift Share digunakan untuk menguraikan pertumbuhan suatu sektor ke dalam komponen pengaruh pertumbuhan nasional, struktur ekonomi, dan daya saing daerah. Melalui metode ini, dapat diketahui apakah kinerja suatu sektor didorong oleh dinamika nasional atau oleh keunggulan kompetitif yang dimiliki daerah."
            kriteria_teks = (
                "- **Kriteria I (RS + IM +):** Sektor Tumbuh Pesat $\\rightarrow$ sektor yang memiliki daya saing tinggi di tingkat lokal dan didukung oleh tren pertumbuhan nasional.\n\n"
                "- **Kriteria II (RS + IM -):** Sektor Berpotensi $\\rightarrow$ sektor yang kuat secara lokal meskipun secara nasional cenderung melambat, sehingga berpotensi menjadi keunggulan spesifik daerah.\n\n"
                "- **Kriteria III (RS - IM +):** Sektor Berkembang $\\rightarrow$ sektor yang tumbuh secara nasional namun belum diikuti oleh daya saing daerah, sehingga memerlukan penguatan kapasitas lokal.\n\n"
                "- **Kriteria IV (RS - IM -):** Sektor Tertinggal $\\rightarrow$ sektor dengan daya saing dan pertumbuhan yang rendah baik di tingkat lokal maupun nasional."
            )
            col_x, col_y = "im_2025", "rs_2025"
            garis_x, garis_y = 0.0, 0.0  
            labels_x, labels_y = "Regional Share (RS)", "Industrial Mix (IM)"

        else:  
            judul_full = 'Scatter Plot "Tipologi Klassen Rata-Rata 2022-2025"'
            help_teks = "Tipologi Klassen merupakan metode klasifikasi sektor berdasarkan tingkat pertumbuhan dan kontribusinya terhadap perekonomian daerah. Hasil analisisnya memberikan gambaran yang jelas mengenai posisi relatif setiap sektor, mulai dari sektor unggulan hingga sektor yang masih tertinggal, sehingga mendukung perumusan arah pembangunan ekonomi daerah."
            kriteria_teks = (
                "- **Kriteria I (Pertumbuhan > Nas & Kontribusi > Nas):** Sektor Andalan $\\rightarrow$ sektor dengan pertumbuhan dan kontribusi tinggi yang menjadi prioritas utama pembangunan ekonomi.\n\n"
                "- **Kriteria II (Pertumbuhan > Nas & Kontribusi < Nas):** Sektor Berkembang $\\rightarrow$ sektor dengan pertumbuhan tinggi namun kontribusi masih kecil, sehingga berpotensi menjadi andalan baru.\n\n"
                "- **Kriteria III (Pertumbuhan < Nas & Kontribusi > Nas):** Sektor Potensial $\\rightarrow$ sektor dengan kontribusi besar tetapi pertumbuhan melambat, sehingga perlu dijaga agar tidak menurun.\n\n"
                "- **Kriteria IV (Pertumbuhan < Nas & Kontribusi < Nas):** Sektor Tertinggal $\\rightarrow$ sektor dengan pertumbuhan dan kontribusi rendah yang memerlukan perhatian dan intervensi khusus."
            )
            col_x, col_y = "kontribusi_2025", "pertumbuhan_2025"
            garis_x, garis_y = 5.6, 5.1  
            labels_x, labels_y = "Rata-Rata Kontribusi (%)", "Rata-Rata Pertumbuhan (%)"

        if col_x not in df_aktif.columns or col_y not in df_aktif.columns:
            return st.warning(f"Kolom {col_x} atau {col_y} tidak ditemukan pada data sektoral.")

        st.markdown(f"##### {judul_full}", help=help_teks)
        
        # PERBAIKAN: Melebarkan sedikit porsi kolom narasi agar teks panjang tidak terlalu tergencet
        col_grafik, col_narasi = st.columns([1.8, 1.2]) 
        
        with col_grafik:
            warna_map = get_warna_sektor_map(df_aktif.get('sektor', pd.Series(dtype=str)))
            fig = px.scatter(
                df_aktif, x=col_x, y=col_y, text="sektor" if "sektor" in df_aktif.columns else None,        
                color="sektor" if "sektor" in df_aktif.columns else None,        
                labels={col_x: labels_x, col_y: labels_y}, color_discrete_map=warna_map  
            )
            fig.add_hline(y=garis_y, line_dash="dash", line_color="#475569", line_width=1.5)
            fig.add_vline(x=garis_x, line_dash="dash", line_color="#475569", line_width=1.5)
            fig.update_traces(textposition='top center', marker=dict(size=14))
            
            # Membuat grafik lebih bersih & modern
            fig.update_layout(
                showlegend=False, 
                margin={"r": 20, "t": 30, "l": 20, "b": 20},
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
                yaxis=dict(showgrid=True, gridcolor='#F1F5F9')
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col_narasi:
            # Sentuhan UI: Kotak header deskripsi yang rapi dan elegan
            st.markdown('''
            <div style="background: rgba(241, 245, 249, 0.8); border-left: 4px solid #3B82F6; padding: 12px 15px; border-radius: 6px; margin-bottom: 15px;">
                <h6 style="margin: 0; color: #1E293B; font-weight: 700; font-size: 14px;">📖 Deskripsi Kuadran Sektor:</h6>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown(kriteria_teks)

    def format_val_daerah(val, unit=""):
        if pd.isna(val) or val == "" or str(val).lower() == 'nan': return "-"
        try: return f"{float(val):,.1f}{unit}"
        except ValueError: return f"{val}{unit}"

    def format_gini_daerah(val):
        if pd.isna(val) or val == "" or str(val).lower() == 'nan': return "-"
        try: return f"{float(val):.3f}"
        except ValueError: return f"{val}"

    # ==============================================================================
    # UI DASHBOARD DAERAH
    # ==============================================================================
    st.markdown("### 📍 Command Center: Ekonomi Kewilayahan")
    st.markdown("---")

    col_provinsi, col_tahun = st.columns(2)
    with col_provinsi:
        list_provinsi = [
            "Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Jambi", "Sumatera Selatan", "Bengkulu", "Lampung", "Kepulauan Bangka Belitung", "Kepulauan Riau", "DKI Jakarta", "Jawa Barat", "Jawa Tengah", "DI Yogyakarta", "Jawa Timur", "Banten", "Bali", "Nusa Tenggara Barat", "Nusa Tenggara Timur", "Kalimantan Barat", "Kalimantan Tengah", "Kalimantan Selatan", "Kalimantan Timur", "Kalimantan Utara", "Sulawesi Utara", "Sulawesi Tengah", "Sulawesi Selatan", "Sulawesi Tenggara", "Gorontalo", "Sulawesi Barat", "Maluku", "Maluku Utara", "Papua Barat", "Papua Barat Daya", "Papua", "Papua Selatan", "Papua Tengah", "Papua Pegunungan"
        ]
        provinsi_terpilih = st.selectbox("Pilih Wilayah Analisis:", list_provinsi, key="daerah_prov")

    with col_tahun:
        tahun_terpilih = st.selectbox("Tahun Analisis:", list(range(2011, 2027)), index=14, key="daerah_tahun")

    st.markdown("---")

    df_all_prov = load_data_aman_daerah(provinsi_terpilih, tahun_terpilih) 
    df_sektoral_aktif = load_data_sektoral_aman_daerah(provinsi_terpilih)
    df_struktur_aktif = load_data_struktur_aman_daerah(provinsi_terpilih)


    df_row = df_all_prov[(df_all_prov['provinsi'] == provinsi_terpilih) & (df_all_prov['tahun'] == int(tahun_terpilih))]
    df_active_dict = df_row.iloc[0].to_dict() if not df_row.empty else {}

    st.header("1. KONDISI EKONOMI MAKRO DAERAH 38 PROVINSI")
    col_Grafik1, col_Grafik2 = st.columns(2)
    with col_Grafik1:
        st.subheader(f"Laju Pertumbuhan Ekonomi ({tahun_terpilih})")
        buat_bar_chart_makro(df_all_prov, "Pertumbuhan Ekonomi", provinsi_terpilih)
    with col_Grafik2:
        st.subheader(f"Kontribusi PDRB terhadap Nasional ({tahun_terpilih})")
        buat_bar_chart_makro(df_all_prov, "Kontribusi PDRB", provinsi_terpilih)

    st.subheader(f"🗺️ Sebaran Klasifikasi Wilayah")
    buat_peta_klasifikasi(df_all_prov, provinsi_terpilih)
    st.markdown("***Catatan:*** *Klasifikasi menggunakan metode Tipologi Klassen dengan mempertimbangkan rata-rata pertumbuhan ekonomi dan PDRB per Kapita Tahun 2022-2025*")

    st.markdown("---")
    st.header(f"2. KINERJA INDIKATOR EKONOMI DAN SOSIAL {provinsi_terpilih.upper()}")

    st.markdown("#### Pertumbuhan Ekonomi (YoY)")
    buat_line_growth(provinsi_terpilih)

    st.write(f"**Capaian Laju Pertumbuhan Ekonomi Makro Daerah**")
    q1, q2, q3, q4, q5 = st.columns(5)
    
    # PERBAIKAN DI SINI: Semua pemanggilan format menggunakan fungsi berakhiran _daerah
    q1.metric("TW I YoY (%)", format_val_daerah(df_active_dict.get("lpe_tw1")))
    q2.metric("TW II YoY (%)", format_val_daerah(df_active_dict.get("lpe_tw2")))
    q3.metric("TW III YoY (%)", format_val_daerah(df_active_dict.get("lpe_tw3")))
    q4.metric("TW IV YoY (%)", format_val_daerah(df_active_dict.get("lpe_tw4")))

    capaian_ctc = df_active_dict.get("lpe_ctc", np.nan)
    capaian_ctc_str = format_val_daerah(capaian_ctc)

    with q5:
        st.markdown(f'<div style="background-color:#0A192F; color:white; padding:10px; border-radius:5px; text-align:center;"><p style="margin:0; font-size:20px;">Capaian c-to-c (%)</p><h3 style="margin:0; color:#00CC96;">{capaian_ctc_str}</h3></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div style="background-color:#1E293B; padding:5px; border-radius:10px;"><h4 style="color:#F8FAFC; margin-top:0; padding-left:10px;">Simulasi Pencapaian Target Pertumbuhan Ekonomi Tahun 2026</h4></div>', unsafe_allow_html=True)
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            target_2026 = st.number_input("**Target Pertumbuhan Ekonomi (%):**", value=5.0, step=0.1, key="daerah_target")
            try: capaian_realitas = float(capaian_ctc) if pd.notna(capaian_ctc) else 0.0
            except ValueError: capaian_realitas = 0.0
            status_track = "On Track / Realistis untuk Dicapai" if capaian_realitas >= target_2026 else "Memerlukan Dukungan Percepatan / Upaya Ekstra"
            st.write(f"**Status Capaian:** {status_track}")
            pembagi = max(target_2026, 0.1)
            st.progress(min(max(float(capaian_realitas / pembagi), 0.0), 1.0))
        with col_sim2:
            sisa_target = max((target_2026 * 4 - capaian_realitas) / 3, 0.0)
            st.write(f"**Interpretasi Singkat:** Untuk mencapai target pertumbuhan sebesar {target_2026} %, laju pertumbuhan rata-rata pada Triwulan selanjutnya minimal harus didorong sebesar {sisa_target:.2f} %.")

    st.markdown("#### Struktur Ekonomi Daerah")
    buat_area_struktur(df_struktur_aktif)

    st.markdown("#### Indikator Ekonomi dan Sosial Lainnya")
    col_ek1, col_ek2, col_ek3, col_ek4 = st.columns(4)
    with col_ek1: st.metric(label="PDRB Perkapita (Juta Rupiah)", value=format_val_daerah(df_active_dict.get('pdrb_perkapita')))
    with col_ek2: st.metric(label="Inflasi Tahunan (%)", value=format_val_daerah(df_active_dict.get('inflasi')))
    with col_ek3: st.metric(label="Nilai PMA (Juta USD)", value=format_val_daerah(df_active_dict.get('pma')))
    with col_ek4: st.metric(label="Nilai PMDN (Miliar Rupiah)", value=format_val_daerah(df_active_dict.get('pmdn')))

    col_sos1, col_sos2, col_sos3, col_sos4 = st.columns(4)
    with col_sos1: st.metric(label="IPM", value=format_val_daerah(df_active_dict.get('ipm')))
    with col_sos2: st.metric(label="Kemiskinan (%)", value=format_val_daerah(df_active_dict.get('kemiskinan')))
    with col_sos3: st.metric(label="TPT (%)", value=format_val_daerah(df_active_dict.get('tpt')))
    with col_sos4: st.metric(label="Rasio Gini", value=format_gini_daerah(df_active_dict.get('gini')))

    st.markdown("<br>", unsafe_allow_html=True)
    col_bawah1, col_bawah2 = st.columns(2)
    with col_bawah1:
        nilai_ekspor = format_val_daerah(df_active_dict.get('ekspor_top3'))
        st.markdown(f'<div style="line-height: 1.3;"><p style="margin:0; font-size:14px; font-weight:500; opacity: 0.8;">Ekspor Terbesar</p><h3 style="margin:0; font-size:16px; font-weight:600; color: var(--text-color, inherit); white-space: normal; word-wrap: break-word;">{nilai_ekspor}</h3></div>', unsafe_allow_html=True)
    with col_bawah2:
        nilai_naker = format_val_daerah(df_active_dict.get('naker_top'))
        st.markdown(f'<div style="line-height: 1.3;"><p style="margin:0; font-size:14px; font-weight:500; opacity: 0.8;">Tenaga Kerja Terbesar</p><h3 style="margin:0; font-size:16px; font-weight:600; color: var(--text-color, inherit); white-space: normal; word-wrap: break-word;">{nilai_naker}</h3></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.header(f"3. ANALISIS SEKTOR UNGGULAN DAERAH {provinsi_terpilih.upper()}")
    buat_scatter_sektoral(df_sektoral_aktif, "Overlay")
    buat_scatter_sektoral(df_sektoral_aktif, "Shift Share")
    buat_scatter_sektoral(df_sektoral_aktif, "Tipologi Klassen")

    st.markdown("---")
    st.header("4. INTERPRETASI DAN REKOMENDASI")
    st.markdown("### Interpretasi")
    st.info(format_val_daerah(df_active_dict.get("interpretasi_ekonomi_riil")))
    st.markdown("### Rekomendasi")
    st.success(format_val_daerah(df_active_dict.get("rekomendasi_ekonomi_riil")))
# =========================================================================
# MODUL 4: ANALISIS SENSITIVITAS (SEKTOR EKSTERNAL & FISKAL)
# =========================================================================
elif main_menu == "🌍 Analisis Sensitivitas":
    with st.sidebar:
        st.markdown("### 🇮🇩 BI-BAPPENAS")
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
            "-- pilih --", "📊 Base Med", "📈 Base High", "📉 Depresiasi",
            "🛢 Minyak Rendah", "🔥 Minyak Tinggi", "⚡ Twin Shock",
        ])

        # Penentuan nilai absolut awal berdasarkan preset yang dipilih
        if   "Depresiasi" in preset: nt_init, oil_init = 19_500,       icp_default
        elif "Rendah"     in preset: nt_init, oil_init = nt_default,  40
        elif "Tinggi"     in preset: nt_init, oil_init = nt_default,  100
        elif "Twin"       in preset: nt_init, oil_init = 20_000,      100
        else:                        nt_init, oil_init = nt_default,  icp_default

        # KUNCI PERUBAHAN: Konversi nilai absolut menjadi Delta (Selisih) untuk UI
        delta_nt_init = int(nt_init - nt_default)
        delta_oil_init = int(oil_init - icp_default)

        st.divider()
        st.markdown(f"**PERUBAHAN NILAI TUKAR (Δ Rp/USD)**")
        delta_nt = st.number_input("Delta NT", min_value=-10000, max_value=15000, value=delta_nt_init, step=50, label_visibility="collapsed")

        st.markdown(f"**PERUBAHAN HARGA MINYAK ICP (Δ USD/bbl)**")
        delta_oil = st.number_input("Delta ICP", min_value=-100, max_value=100, value=delta_oil_init, step=1, label_visibility="collapsed")

        # KUNCI PERUBAHAN: Konversi kembali Delta menjadi Absolut untuk dihitung oleh Engine
        nt = nt_default + delta_nt
        oil = icp_default + delta_oil

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
        
    st.markdown("### 🌍 Dashboard Sensitivitas Eksternal & Fiskal RI")
    
    st.session_state['ext_nt'] = nt
    st.session_state['ext_oil'] = oil
    st.session_state['ext_gdp_drop'] = s_sim["gdp"] - b_sim["gdp"]
    st.session_state['ext_def'] = s_sim["defpdb"] - b_sim["defpdb"]
    
    keys = ["ca", "exp", "imp", "reserves", "gdp", "gexp", "gimp",
            "def", "rev", "bel", "sube", "bunga", "pajak"]
    R = {k: {"b": [], "s": []} for k in keys}

    for y in YEARS:
        bb, ss = simulate_eksternal(nt, oil, y, scen)
        if y == yr: 
            b_sim, s_sim = bb, ss
            
        for k, bval, sval, dec in [
            ("ca",       bb["ca"],       ss["ca"],       2),
            ("exp",      bb["exp"],      ss["exp"],      1),
            ("imp",      bb["imp"],      ss["imp"],      1),
            ("reserves", bb["reserves"], ss["reserves"], 1),
            ("gdp",      bb["gdp"],      ss["gdp"],      2),
            ("gexp",     bb["gexp"],     ss["gexp"],     2),
            ("gimp",     bb["gimp"],     ss["gimp"],     2),
            ("def",      bb["def"],      ss["def"],      0),
            ("rev",      bb["rev"],      ss["rev"],      0),
            ("bel",      bb["bel"],      ss["bel"],      0),
            ("sube",     bb["sube"],     ss["sube"],     0),
            ("bunga",    bb["bunga"],    ss["bunga"],    0),
            ("pajak",    bb["pajak"],    ss["pajak"],    0),
        ]:
            R[k]["b"].append(round(bval, dec))
            R[k]["s"].append(round(sval, dec))

    YL = [str(y) for y in YEARS]

    tab_bop, tab_gdp, tab_apbn, tab_table = st.tabs([
        "📊 Neraca Pembayaran (BOP)",
        "📈 Pertumbuhan Ekonomi (GDP)",
        "🏛 Defisit APBN",
        "📋 Tabel Lengkap",
    ])

    with tab_bop:
        bop_kpis = [
            ("🔵 Transaksi Berjalan",  f"{s_sim['ca']:.2f}",       "Miliar USD", s_sim["ca"]-b_sim["ca"],             " Miliar USD"),
            ("🟢 Ekspor Barang (fob)", f"{s_sim['exp']:.1f}",      "Miliar USD", s_sim["exp"]-b_sim["exp"],             " Miliar USD"),
            ("🔴 Impor Barang (fob)",  f"{s_sim['imp']:.1f}",      "Miliar USD", s_sim["imp"]-b_sim["imp"],             " Miliar USD"),
            ("🟠 Cadangan Devisa",     f"{s_sim['reserves']:.1f}", "Miliar USD", s_sim["reserves"]-b_sim["reserves"],  " Miliar USD"),
            ("🩵 Bulan Impor",         f"{s_sim['bulan_imp']:.1f}","Bulan",      s_sim["bulan_imp"]-b_sim["bulan_imp"]," bln"),
            ("🟡 CA / PDB",            f"{s_sim['capdb']:.2f}%",   "%",          s_sim["capdb"]-b_sim["capdb"],        " pp"),
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

        c4, c5 = st.columns(2)
        with c4:
            tb_b   = [SCEN[scen]["tradebal"][y] for y in YEARS]
            svc_b  = [SCEN[scen]["svcbal"][y]   for y in YEARS]
            prim_b = [SCEN[scen]["primbal"][y]   for y in YEARS]
            sec_b  = [SCEN[scen]["secbal"][y]    for y in YEARS]
            fig = go.Figure([
                bar_trace("N. Barang",      YL, tb_b,   C["blue"],   opacity=0.75),
                bar_trace("N. Jasa",        YL, svc_b,  C["amber"],  opacity=0.75),
                bar_trace("Pend. Primer",   YL, prim_b, C["red"],    opacity=0.75),
                bar_trace("Pend. Sekunder", YL, sec_b,  C["green"],  opacity=0.75),
            ])
            fig.update_layout(**fig_layout("Komponen Neraca Berjalan (Miliar USD)", barmode="group"))
            st.plotly_chart(fig, use_container_width=True)
        with c5:
            nt_range = list(range(12_000, 26_500, 500))
            ca_sens  = [round(simulate_eksternal(n, oil, yr, scen)[1]["ca"], 2) for n in nt_range]
            fig = go.Figure([
                go.Scatter(
                    x=[n / 1000 for n in nt_range], y=ca_sens,
                    mode="lines", name="CA (Miliar USD)",
                    line=dict(color=C["blue"], width=2), fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"
                ),
                dot_trace("Posisi kini", [nt / 1000], [round(s_sim["ca"], 2)])
            ])
            fig.update_layout(**fig_layout("Sensitivitas CA vs Nilai Tukar", xaxis_title="NT (ribu Rp)", yaxis_title="CA (Miliar USD)"))
            st.plotly_chart(fig, use_container_width=True)

    with tab_gdp:
        gdp_kpis = [
            ("🟢 PDB Growth",     f"{s_sim['gdp']:.2f}%",  s_sim["gdp"]-b_sim["gdp"],  " pp"),
            ("🔵 Konsumsi RT",    f"{s_sim['cons']:.2f}%", s_sim["cons"]-b_sim["cons"]," pp"),
            ("🩵 PMTB/Investasi", f"{s_sim['inv']:.2f}%",  s_sim["inv"]-b_sim["inv"],  " pp"),
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
            gdp_sens  = [round(simulate_eksternal(n, oil, yr, scen)[1]["gdp"], 2) for n in nt_range2]
            fig = go.Figure([
                go.Scatter(
                    x=[n / 1000 for n in nt_range2], y=gdp_sens,
                    mode="lines", name="PDB Growth (%)",
                    line=dict(color=C["green"], width=2), fill="tozeroy", fillcolor="rgba(22,163,74,0.08)"
                ),
                dot_trace("Posisi kini", [nt / 1000], [round(s_sim["gdp"], 2)])
            ])
            fig.update_layout(**fig_layout("Sensitivitas PDB vs Nilai Tukar", xaxis_title="NT (ribu Rp)", yaxis_title="PDB Growth (%)"))
            st.plotly_chart(fig, use_container_width=True)

        tx = s_sim["tx"]
        st.markdown("#### 🔴 MEKANISME TRANSMISI: NILAI TUKAR & HARGA MINYAK → PDB")
        col_nt, col_icp = st.columns(2)

        def color_val(val, inverse=False):
            if abs(val) < 0.005: return f"<span style='color:#6b7280; font-weight:bold;'>{val:+.2f}pp</span>"
            is_green = (val > 0) if not inverse else (val < 0)
            color = "#16a34a" if is_green else "#dc2626"
            return f"<span style='color:{color}; font-weight:bold;'>{val:+.2f}pp</span>"

        with col_nt:
            html_nt = f"""
            <table style="width:100%; text-align:left; border-collapse: collapse; font-family: sans-serif; font-size:14px; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden;">
                <tr style="background-color: #f9fafb; border-bottom: 2px solid #e5e7eb;">
                    <th style="padding:12px 16px; color:#374151;">Jalur Nilai Tukar ke PDB</th><th style="text-align:right; padding:12px 16px;"></th>
                </tr>
                <tr style="border-bottom: 1px solid #f3f4f6; background-color: #ffffff;">
                    <td style="padding:12px 16px; color:#4b5563;">Daya saing ekspor non-migas</td><td style="text-align:right; padding:12px 16px;">{color_val(tx['expNT'])}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f3f4f6; background-color: #ffffff;">
                    <td style="padding:12px 16px; color:#4b5563;">Substitusi impor (volume turun)</td><td style="text-align:right; padding:12px 16px;">{color_val(tx['impNT'])}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f3f4f6; background-color: #f0fdf4;">
                    <td style="padding:12px 16px; color:#166534; font-weight:600;">Net ekspor Δ ke PDB</td><td style="text-align:right; padding:12px 16px;">{color_val(tx['netNT'])}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f3f4f6; background-color: #ffffff;">
                    <td style="padding:12px 16px; color:#4b5563;">Konsumsi (tekanan inflasi impor)</td><td style="text-align:right; padding:12px 16px;">{color_val(tx['consNT'])}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f3f4f6; background-color: #ffffff;">
                    <td style="padding:12px 16px; color:#4b5563;">Investasi (ketidakpastian)</td><td style="text-align:right; padding:12px 16px;">{color_val(tx['invNT'])}</td>
                </tr>
                <tr style="background-color: #f8fafc; font-weight:bold; border-top: 2px solid #e5e7eb;">
                    <td style="padding:12px 16px; color:#1e293b;">Total Δ PDB via NT</td><td style="text-align:right; padding:12px 16px;">{color_val(tx['totalNT'])}</td>
                </tr>
            </table>
            """
            st.markdown(html_nt, unsafe_allow_html=True)

        with col_icp:
            html_icp = f"""
            <table style="width:100%; text-align:left; border-collapse: collapse; font-family: sans-serif; font-size:14px; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden;">
                <tr style="background-color: #f9fafb; border-bottom: 2px solid #e5e7eb;">
                    <th style="padding:12px 16px; color:#374151;">Jalur Harga Minyak ke PDB</th><th style="text-align:right; padding:12px 16px;"></th>
                </tr>
                <tr style="border-bottom: 1px solid #f3f4f6; background-color: #ffffff;">
                    <td style="padding:12px 16px; color:#4b5563;">Ekspor migas (pendapatan)</td><td style="text-align:right; padding:12px 16px;">{color_val(tx['expICP'])}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f3f4f6; background-color: #ffffff;">
                    <td style="padding:12px 16px; color:#4b5563;">Tagihan impor BBM (beban)</td><td style="text-align:right; padding:12px 16px;">{color_val(tx['impICP'])}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f3f4f6; background-color: #f0fdf4;">
                    <td style="padding:12px 16px; color:#166534; font-weight:600;">Net ekspor Δ ke PDB</td><td style="text-align:right; padding:12px 16px;">{color_val(tx['netICP'])}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f3f4f6; background-color: #ffffff;">
                    <td style="padding:12px 16px; color:#4b5563;">Konsumsi (tekanan daya beli)</td><td style="text-align:right; padding:12px 16px;">{color_val(tx['consICP'])}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f3f4f6; background-color: #ffffff;">
                    <td style="padding:12px 16px; color:#4b5563;">Biaya produksi / investasi</td><td style="text-align:right; padding:12px 16px;">{color_val(tx['invICP'])}</td>
                </tr>
                <tr style="background-color: #f8fafc; font-weight:bold; border-top: 2px solid #e5e7eb;">
                    <td style="padding:12px 16px; color:#1e293b;">Total Δ PDB via Minyak</td><td style="text-align:right; padding:12px 16px;">{color_val(tx['totalICP'])}</td>
                </tr>
            </table>
            """
            st.markdown(html_icp, unsafe_allow_html=True)

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

        limit = 3.0
        col_b, col_s = st.columns(2)
        with col_b:
            alert_b = "🔴" if abs(b_sim["defpdb"]) > limit else "🟢"
            st.markdown(f"**Baseline {alert_b}:** `{b_sim['defpdb']:.2f}% PDB`")
            st.progress(min(abs(b_sim["defpdb"]) / limit, 1.0))
        with col_s:
            alert_s = "🔴" if abs(s_sim["defpdb"]) > limit else "🟢"
            st.markdown(f"**Simulasi {alert_s}:** `{s_sim['defpdb']:.2f}% PDB`")
            st.progress(min(abs(s_sim["defpdb"]) / limit, 1.0))

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
                bar_trace("Belanja Simulasi",    YL, R["bel"]["s"], C["red2"], opacity=0.75)
            ])
            fig.update_layout(**fig_layout("Pendapatan vs Belanja (Rp T)", barmode="group"))
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            icp_range = list(range(30, 135, 5))
            def_sens  = [round(simulate_eksternal(nt, ic, yr, scen)[1]["def"], 0) for ic in icp_range]
            fig = go.Figure([
                go.Scatter(
                    x=[f"${ic}" for ic in icp_range], y=def_sens,
                    mode="lines", name="Defisit (Rp T)",
                    line=dict(color=C["purple"], width=2), fill="tozeroy", fillcolor="rgba(124,58,237,0.08)"
                ),
                dot_trace("Posisi kini", [f"${oil}"], [round(s_sim["def"], 0)])
            ])
            fig.update_layout(**fig_layout("Sensitivitas Defisit vs ICP", xaxis_title="ICP (USD/bbl)", yaxis_title="Defisit (Rp T)"))
            st.plotly_chart(fig, use_container_width=True)

        ax = s_sim["ax"]
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
        st.markdown(f"#### Tabel Lengkap BOP, GDP & APBN — Baseline vs Simulasi (Tahun {yr})")
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
            ("  Konsumsi RT (aktif)",          "cons",      "cons",      2,   "%"),
            ("  PMTB/Investasi (aktif)",       "inv",       "inv",       2,   "%"),
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
                "Indikator": lbl, "Baseline": round(bv, dec), "Simulasi": round(sv, dec),
                "Delta Abs": f"{sign}{round(diff, dec)}", "Delta %": f"{sign}{pct:.1f}%", "Satuan": unit,
            })
        st.dataframe(pd.DataFrame(rows_tbl), hide_index=True, use_container_width=True, height=720)


# =========================================================================
# MODUL 5: AI EXECUTIVE BRIEF (SYNTHESIS K/L)
# =========================================================================
elif main_menu == "🧠 AI Executive Brief (Synthesis)":
    st.markdown("""
    <style>
    .glass-card { background: rgba(255, 255, 255, 0.65); box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.7); padding: 24px; margin-bottom: 24px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧠 AI Policy Synthesis & Executive Brief")
    st.markdown("Modul ini mensintesis data dari seluruh *dashboard* untuk memproduksi rumusan kebijakan lintas sektoral secara makro dan mikro komprehensif.")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    mac_view = st.session_state.get('mac_view', '2026')
    mac_avg = st.session_state.get('mac_avg', 5.4)
    mac_target = st.session_state.get('mac_target', 5.4)
    mac_month = st.session_state.get('mac_monthly', 'Indikator riil berjalan stabil, perlu atensi khusus pada volatilitas komponen daya beli.')
    mac_heat = st.session_state.get('mac_heat', 'Mayoritas indikator dalam batas aman.')
    mac_day = st.session_state.get('mac_daily', 'Pasar saham dan komoditas bergerak normal.')
    
    ext_nt = st.session_state.get('ext_nt', 16700)
    ext_oil = st.session_state.get('ext_oil', 65)
    ext_gdp_drop = st.session_state.get('ext_gdp_drop', 0.0)
    ext_def = st.session_state.get('ext_def', 0.0)

    signature = make_signature(mac_view, mac_avg, mac_target, mac_month, mac_day, ext_nt, ext_oil)
    editor_key = f"editor_synthesis_{signature}"
    final_policy_text = ""

    if signature in st.session_state.policy_cache:
        if editor_key not in st.session_state:
            st.session_state[editor_key] = st.session_state.policy_cache[signature]
        st.success("✅ Draf Sintesis Lintas Sektor tersedia. Silakan tinjau dan edit di bawah.")

    if signature not in st.session_state.policy_cache:
        if st.button("Generate Sintesis Kebijakan (AI Bappenas)", type="primary"):
            genai.configure(api_key=USER_API_KEY)
            with st.spinner('AI sedang merumuskan arahan strategis The Bappenas Way (Menggunakan mode Streaming agar tidak terpotong)...'):
                try:
                    avail = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    model_name = next((m for m in avail if 'flash' in m), avail[0] if avail else None)

                    if not model_name: 
                        st.error("Gagal mendeteksi model. Cek API Key.")
                    else:
                        generation_config = genai.types.GenerationConfig(
                            temperature=0.7, top_p=0.9, max_output_tokens=8192
                        )
                        safety_settings = [
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                        ]
                        model = genai.GenerativeModel(model_name)
                        
                        # --- TANGKAP DATA INTELIJEN & KEBOCORAN UNTUK AI ---
                        df_exim_ai = st.session_state.get('exim_data_state', pd.DataFrame())
                        meta_exim_ai = st.session_state.get('exim_meta_state', {})
                        
                        exim_info = "Menunggu pembaruan data real-time."
                        mirroring_info = "Waspadai indikasi under-invoicing dan asimetri pencatatan (kebocoran devisa) dengan negara mitra dagang utama berbasis data rekonsiliasi Trade Map."
                        
                        if not df_exim_ai.empty:
                            total_val = df_exim_ai['value'].sum() / (1e9 if meta_exim_ai.get('unit') == 'Miliar USD' else 1)
                            arah = meta_exim_ai.get('sumber', 'Perdagangan')
                            exim_info = f"Total {arah} mencapai {total_val:,.2f} {meta_exim_ai.get('unit', 'USD')}. Perlu atensi pada stabilitas volume komoditas unggulan dan diversifikasi pasar."
                        
                        # --- TANGKAP DATA KEWILAYAHAN UNTUK AI ---
                        regional_info = "Data kewilayahan belum termuat secara penuh."
                        try:
                            import os
                            import pandas as pd
                            base_dir = os.path.dirname(os.path.abspath(__file__))
                            path_daerah = os.path.join(base_dir, 'data_ekonomi.xlsx')
                            if not os.path.exists(path_daerah):
                                path_daerah = os.path.join(base_dir, 'data', 'data_ekonomi.xlsx')
                            
                            if os.path.exists(path_daerah):
                                df_d = pd.read_excel(path_daerah, engine='openpyxl')
                                df_d.columns = df_d.columns.astype(str).str.strip().str.lower()
                                df_d['tahun'] = pd.to_numeric(df_d['tahun'], errors='coerce')
                                df_d['lpe_ctc'] = pd.to_numeric(df_d['lpe_ctc'].astype(str).str.replace(',', '.'), errors='coerce')
                                df_d['kontribusi'] = pd.to_numeric(df_d['kontribusi'].astype(str).str.replace(',', '.'), errors='coerce')
                                
                                t_max = df_d['tahun'].max()
                                df_latest = df_d[df_d['tahun'] == t_max]
                                
                                top_g = ", ".join([f"{r['provinsi']} ({r['lpe_ctc']}%)" for _, r in df_latest.nlargest(3, 'lpe_ctc').iterrows()])
                                bot_g = ", ".join([f"{r['provinsi']} ({r['lpe_ctc']}%)" for _, r in df_latest.nsmallest(3, 'lpe_ctc').iterrows()])
                                top_k = ", ".join([f"{r['provinsi']} ({r['kontribusi']}%)" for _, r in df_latest.nlargest(3, 'kontribusi').iterrows()])
                                
                                if 'klasifikasi' in df_latest.columns:
                                    klas = df_latest['klasifikasi'].value_counts()
                                    klas_str = ", ".join([f"{k} ({v} Prov)" for k, v in klas.items()])
                                else:
                                    klas_str = "-"
                                    
                                regional_info = f"Tahun Analisis: {int(t_max)}. LPE Tertinggi: {top_g}. LPE Terendah: {bot_g}. Top Kontributor PDRB thd Nasional: {top_k}. Sebaran Klasifikasi Wilayah (Tipologi Klassen): {klas_str}."
                        except Exception:
                            pass
                        # ---------------------------------------------------

                        prompt = f"""
Anda adalah Perencana Pembangunan Nasional Ahli Utama di Direktorat Perencanaan Ekonomi Makro dan Pengembangan Model Pembangunan, Kementerian PPN/Bappenas. 
Tugas Anda adalah menulis Executive Brief dari "Dashboard Macro Early Warning System" untuk dilaporkan kepada Menteri PPN/Kepala Bappenas.

ATURAN MUTLAK DAN SANGAT PENTING:
1. JANGAN PERNAH memberikan kalimat pembuka atau basa-basi (seperti "Baik, berikut adalah draf...", "Sebagai Ahli Utama...", atau "Ini adalah hasilnya").
2. Tuliskan jawaban ANDA LANGSUNG dimulai dengan "### [JUDUL EXECUTIVE BRIEF]".
3. Gunakan gaya bahasa perencanaan strategis khas Bappenas (The Bappenas Way) yang mengedepankan "Evidence-Based Planning", visioner, teknokratis, dan tegas.
4. Fokus pada penyelesaian masalah (problem-solving), optimalisasi tata niaga, dan antisipasi risiko ke depan.

=====================
DATA & EVIDENCE MAKRO, EKSTERNAL & KEWILAYAHAN
=====================
Fokus Indikator Makro: {mac_view}
Target PDB Nasional: {mac_target}% | Proyeksi DFM: {mac_avg:.2f}%
Status Real Sector: {mac_month}
Momentum Indikator: {mac_heat}
Pasar Harian: {mac_day}
Guncangan Eksternal: NT Rp {ext_nt}/USD, ICP $ {ext_oil}/bbl
Dampak Skenario Eksternal: PDB {ext_gdp_drop:+.2f} pp, Defisit APBN {ext_def:+.2f} pp thd PDB.
Kondisi Ekspor-Impor: {exim_info}
Isu Kebocoran Perdagangan: {mirroring_info}
Dinamika Ekonomi Kewilayahan: {regional_info}

=====================
STRUKTUR EXECUTIVE BRIEF:
=====================
### [BUAT JUDUL EXECUTIVE BRIEF YANG MENCERMINKAN STATUS EARLY WARNING SAAT INI]

**1. ASESMEN RISIKO DAN POSISI STRATEGIS BAPPENAS**
(Analisis perkembangan terkini dari makro nasional, hasil simulasi guncangan eksternal (Rupiah & Minyak), SERTA evaluasi kinerja ekspor-impor dan indikasi kebocoran/asimetri data perdagangan. Berdasarkan data early warning di atas, deklarasikan posisi/stance Bappenas dengan tegas: apakah situasi ini terkendali, waspada, atau krisis? Apa *root cause* permasalahannya?).

**2. DINAMIKA EKONOMI KEWILAYAHAN DAN SPASIAL**
(Lakukan analisis mendalam berdasarkan data 'Dinamika Ekonomi Kewilayahan'. Soroti laju pertumbuhan ekonomi (LPE) antar wilayah yang jomplang, dominasi kontribusi PDRB daerah terhadap nasional, serta sebaran klasifikasi tipologi wilayah. Berikan insight spasial terkait provinsi mana yang menjadi motor penggerak dan mana yang tertinggal, serta implikasinya terhadap pemerataan pembangunan nasional).

**3. STRATEGI DAN SOLUSI KOMPREHENSIF**
(Jabarkan grand strategy atau solusi menyeluruh dari Bappenas untuk menyelesaikan permasalahan makro, eksternal, dan ketimpangan wilayah yang teridentifikasi di poin 1 dan 2. Jelaskan bagaimana Bappenas akan mengorkestrasi penyelesaian masalah ini secara lintas sektor dan spasial).

**4. REKOMENDASI KEBIJAKAN LINTAS K/L & PEMDA**
(Berikan arahan kebijakan yang jelas dan *actionable* untuk diimplementasikan oleh Kementerian/Lembaga terkait dan Pemerintah Daerah. Bagi secara tegas berdasarkan horizon waktu):
* **A. Jangka Pendek (Mitigasi, Stabilisasi & Penertiban):** (Tindakan taktis dan intervensi segera untuk meredam syok eksternal, menjaga target PDB tahun berjalan, stabilisasi harga, menjaga batas defisit fiskal, serta intervensi cepat bagi daerah yang mengalami perlambatan ekstrem).
* **B. Jangka Menengah & Panjang (Reformasi Struktural):** (Kebijakan struktural untuk memperkuat fundamental ekonomi makro, resiliensi rantai pasok, penguatan kemandirian sektor unggulan daerah/transformasi wilayah berdasarkan Tipologi Klassen, serta harmonisasi sistem pencatatan devisa hasil ekspor dan audit kepatuhan dengan negara mitra).
"""
                        # MENGGUNAKAN STREAM=TRUE AGAR ANTI-TIMEOUT DAN ANTI-KEPOTONG
                        res = model.generate_content(
                            prompt, generation_config=generation_config, safety_settings=safety_settings, stream=True
                        )
                        
                        out_text = ""
                        
                        # --- PERBAIKAN: EKSTRAKSI TEKS AMAN (ANTI ERROR FINISH_REASON 1) ---
                        for chunk in res: 
                            try:
                                # Hanya ambil teks jika potongan datanya valid dan tidak kosong
                                if chunk.text:
                                    out_text += chunk.text
                            except Exception:
                                pass # Abaikan dengan tenang jika server mengirim chunk kosong
                        # ------------------------------------------------------------------
                        
                        out_text = out_text.strip()
                        
                        # Bersihkan tag markdown bawaan AI jika ada
                        if out_text.startswith("```markdown"):
                            out_text = out_text.replace("```markdown", "", 1)
                        if out_text.endswith("```"):
                            out_text = out_text[::-1].replace("```"[::-1], "", 1)[::-1]
                        out_text = out_text.strip()

                        st.session_state.policy_cache[signature] = out_text
                        
                        # Simpan ke cache agar tidak perlu generate ulang jika halaman direfresh
                        try:
                            import pickle
                            with open(CACHE_FILE, "wb") as f: 
                                pickle.dump(st.session_state.policy_cache, f)
                        except Exception:
                            pass 

                        st.session_state[editor_key] = out_text
                        st.success("Sintesis Selesai secara Penuh!")
                        st.rerun()
                        
                except Exception as e: 
                    st.error(f"Error AI: {e}")

    if editor_key in st.session_state:
        st.markdown("---")
        st.session_state[editor_key] = st.text_area("✍️ Ruang Editor Eksekutif:", value=st.session_state[editor_key], height=500)
        with st.expander("🔍 Pratinjau Sintesis Akhir", expanded=True):
            st.markdown(st.session_state[editor_key])
        final_policy_text = st.session_state[editor_key]

    st.markdown('</div>', unsafe_allow_html=True)

    if final_policy_text:
        st.markdown("<br><hr style='border:1px dashed #ccc;'><br>", unsafe_allow_html=True)
        st.markdown("#### 📑 Export Executive Brief")
        try:
            import markdown
            html_policy = markdown.markdown(final_policy_text)
            html_policy = html_policy.replace("<ul>", "<ul class='premium-list'>").replace("<li>", "<li>").replace("<strong>", "<strong class='highlight-text'>")
            html_template = f"""
            <!DOCTYPE html>
            <html lang="id">
            <head>
                <meta charset="UTF-8">
                <title>Executive Brief - Kementerian PPN/Bappenas</title>
                <style>
                    @import url('[https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap](https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap)');
                    body {{ font-family: 'Plus Jakarta Sans', sans-serif; background-color: #f1f5f9; color: #334155; line-height: 1.8; margin: 0; padding: 40px 20px; }}
                    .document-wrapper {{ max-width: 850px; margin: 0 auto; background: #ffffff; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); overflow: hidden; }}
                    .header-banner {{ background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); padding: 40px 50px; text-align: center; border-bottom: 5px solid #eab308; }}
                    .header-banner h1 {{ color: #ffffff; font-size: 32px; margin: 0 0 10px 0; }}
                    .header-banner p {{ color: #cbd5e1; font-size: 15px; margin: 0; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; }}
                    .content-area {{ padding: 50px; }}
                    .content-area h1, .content-area h3 {{ font-size: 24px; color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
                    .content-area h2 {{ color: #1e3a8a; font-size: 20px; margin-top: 35px; display: flex; align-items: center; gap: 10px; }}
                    .content-area h2::before {{ content: ''; display: block; width: 6px; height: 24px; background-color: #eab308; border-radius: 4px; }}
                    .premium-list li {{ margin-bottom: 12px; font-size: 15px; }}
                    .highlight-text {{ color: #0f172a; font-weight: 700; background-color: #fef9c3; padding: 0 4px; border-radius: 4px; }}
                    .footer-note {{ text-align: center; padding: 30px; background-color: #f8fafc; color: #94a3b8; font-size: 13px; border-top: 1px solid #e2e8f0; }}
                </style>
            </head>
            <body>
                <div class="document-wrapper">
                    <div class="header-banner"><h1>Executive Brief</h1><p>Sintesis Makroekonomi, Kewilayahan & Lintas Sektoral</p></div>
                    <div class="content-area">{html_policy}</div>
                    <div class="footer-note">Dihasilkan oleh Macro AI Command Center - Kementerian PPN/Bappenas RI<br><em>Dokumen Internal Terbatas</em></div>
                </div>
            </body>
            </html>
            """
            st.download_button(label="📥 Unduh Executive Summary (Format HTML Elegan)", data=html_template, file_name="Executive_Brief_Bappenas.html", mime="text/html", type="primary")
        except Exception as e: st.warning(f"Gagal menyiapkan dokumen HTML: {e}")
