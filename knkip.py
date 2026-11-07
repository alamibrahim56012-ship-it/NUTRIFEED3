"""
NUTRIFEED
Sistem Cerdas Formulasi Pakan Ayam Broiler melalui Optimasi Nutrisi dan Biaya
Berbasis Linear Programming dengan Pemanfaatan Bahan Pakan Lokal
 
Jalankan:  streamlit run app.py
"""
 
import io
import os
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog
 
# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN & TAMPILAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="NUTRIFEED | Optimasi Pakan Broiler",
    page_icon="🐔",
    layout="wide",
)
 
# [PERBAIKAN] Paksa tema TERANG. Penyebab tampilan hitam: Streamlit mengikuti
# mode gelap Windows/browser. Program membuat .streamlit/config.toml otomatis
# (sekali saja) agar tabel & komponen bawaan juga ikut terang.
if st.get_option("theme.base") != "light":
    try:
        _cfg = os.path.join(os.getcwd(), ".streamlit", "config.toml")
        if not os.path.exists(_cfg):
            os.makedirs(os.path.dirname(_cfg), exist_ok=True)
            with open(_cfg, "w", encoding="utf-8") as _fh:
                _fh.write(
                    '[theme]\nbase = "light"\nprimaryColor = "#2E7D32"\n'
                    'backgroundColor = "#F1F8F2"\nsecondaryBackgroundColor = "#E0F2E4"\n'
                    'textColor = "#1B4332"\n'
                )
    except OSError:
        pass
    st.warning(
        "Tema terang baru dibuat. **Hentikan program (Ctrl+C di terminal), lalu jalankan ulang** "
        "`streamlit run app.py` agar seluruh tampilan (termasuk tabel) menjadi hijau-putih."
    )
 
# TEMA NUTRIFEED: HIJAU DAN PUTIH
st.markdown("""
<style>
 
:root { color-scheme: light; }
 
/* Latar belakang utama */
.stApp {
    background-color: #F1F8F2;
    color: #1B4332;
}
 
/* [PERBAIKAN] Bar atas (Deploy / menu) yang tadinya hitam */
header[data-testid="stHeader"] {
    background-color: #F1F8F2 !important;
    border-bottom: 1px solid #C8E6C9;
}
[data-testid="stToolbar"] *,
[data-testid="stHeader"] * {
    color: #1B5E20 !important;
}
[data-testid="stDecoration"] { display: none; }
 
/* Menu titik tiga & popup pilihan */
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li,
ul[role="listbox"] {
    background-color: #FFFFFF !important;
    color: #1B4332 !important;
}
li[role="option"]:hover,
li[aria-selected="true"] {
    background-color: #E8F5E9 !important;
}
 
/* Area konten */
.block-container {
    padding-top: 1.5rem;
}
 
/* [PERBAIKAN] Semua tulisan menjadi hijau tua (bukan putih/hitam) */
.stApp p, .stApp label, .stApp li, .stApp span,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stMarkdownContainer"] {
    color: #1B4332;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #4F6F5A !important;
}
 
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #E0F2E4;
    border-right: 2px solid #A5D6A7;
}
 
[data-testid="stSidebar"] * {
    color: #1B4332 !important;
}
 
/* Header sidebar */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #1B5E20 !important;
}
 
/* Tab navigasi */
.stTabs [data-baseweb="tab-list"] {
    background-color: #DCEFE0;
    border-radius: 12px;
    padding: 6px;
    gap: 5px;
}
 
.stTabs [data-baseweb="tab"] {
    background-color: #E8F5E9;
    color: #1B5E20;
    border-radius: 8px;
}
/* [PERBAIKAN] Teks tab yang tidak aktif tadinya hampir tak terbaca */
.stTabs [data-baseweb="tab"] p {
    color: #1B5E20 !important;
    font-weight: 600;
}
 
.stTabs [aria-selected="true"] {
    background-color: #2E7D32 !important;
    color: white !important;
}
.stTabs [aria-selected="true"] p {
    color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    background-color: transparent !important;
}
 
/* Judul dan teks */
h1, h2, h3, h4 {
    color: #1B5E20 !important;
}
 
/* Tombol */
.stButton > button,
.stDownloadButton > button {
    background-color: #2E7D32;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: bold;
}
.stButton > button p,
.stDownloadButton > button p {
    color: #FFFFFF !important;
}
 
.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #1B5E20;
    color: white;
    border: 1px solid #81C784;
}
 
/* Input dan selectbox */
.stTextInput input,
.stNumberInput input {
    background-color: white !important;
    color: #1B4332 !important;
    border: 1px solid #81C784;
    border-radius: 8px;
}
 
/* [PERBAIKAN] Kotak input angka & tombol +/- yang tadinya hitam */
div[data-baseweb="input"],
div[data-baseweb="base-input"] {
    background-color: white !important;
    border-color: #81C784 !important;
}
.stNumberInput button {
    background-color: #E8F5E9 !important;
    color: #1B5E20 !important;
    border-color: #A5D6A7 !important;
}
.stNumberInput button svg {
    fill: #1B5E20 !important;
}
 
/* [PERBAIKAN] Selectbox (Fase pertumbuhan) yang tadinya hitam */
div[data-baseweb="select"] > div {
    background-color: white !important;
    border-color: #81C784 !important;
}
div[data-baseweb="select"] * {
    color: #1B4332 !important;
}
div[data-baseweb="select"] svg {
    fill: #1B5E20 !important;
}
 
/* [PERBAIKAN] Pilihan radio (Mode bahan pakan) */
label[data-baseweb="radio"] > div:first-child {
    background-color: #FFFFFF !important;
    border: 2px solid #2E7D32 !important;
}
label[data-baseweb="radio"]:has(input:checked) > div:first-child {
    background-color: #2E7D32 !important;
}
 
/* [PERBAIKAN] Slider hijau */
div[data-baseweb="slider"] div[role="slider"] {
    background-color: #2E7D32 !important;
    box-shadow: none !important;
}
[data-testid="stSliderThumbValue"],
[data-testid="stTickBarMin"],
[data-testid="stTickBarMax"] {
    color: #1B5E20 !important;
}
 
/* Tabel */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background-color: white;
    border-radius: 10px;
    border: 1px solid #C8E6C9;
}
 
/* Kotak informasi */
[data-testid="stAlert"] {
    border-radius: 10px;
    background-color: #E8F5E9 !important;
    border: 1px solid #A5D6A7;
}
[data-testid="stAlert"] * {
    color: #1B4332 !important;
}
 
/* Metric */
[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border-left: 5px solid #2E7D32;
}
[data-testid="stMetricLabel"] p { color: #4F6F5A !important; }
[data-testid="stMetricValue"] { color: #1B5E20 !important; }
 
/* [PERBAIKAN] Grafik dibungkus kartu putih */
[data-testid="stPlotlyChart"] {
    background-color: #FFFFFF;
    border: 1px solid #C8E6C9;
    border-radius: 14px;
    padding: 6px;
    box-shadow: 0 2px 10px rgba(46, 125, 50, .10);
}
 
/* Garis pemisah */
hr {
    border-color: #A5D6A7;
}
 
/* [PERBAIKAN] Tulisan header foto: memakai <div> (bukan <h1>/<p>) karena Streamlit
   membuang class & style dari tag <h1>/<p> di dalam st.markdown */
.hero-img .judul,
.hero-img .subjudul,
.hero-img .tagline {
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.45);
}
.hero-img .judul {
    color: #FFFFFF !important;          /* NUTRIFEED -> putih */
    font-size: 42px;
    font-weight: 800;
    line-height: 1.2;
    margin: 0 0 12px 0;
}
.hero-img .subjudul {
    color: #FFFFFF !important;
    font-size: 19px;
    font-weight: 700;
    margin: 0 0 14px 0;
}
.hero-img .tagline {
    color: #FFE082 !important;          /* kuning emas terang */
    font-size: 15px;
    font-weight: 700;
}
 
</style>
""", unsafe_allow_html=True)
 
GREEN = "#2E7D32"
GREEN_L = "#81C784"
ORANGE = "#F57C00"
BLUE = "#1565C0"
GREY = "#78909C"
 
# [PERBAIKAN] Urutan warna hijau untuk grafik pie
PALET_HIJAU = [
    "#2E7D32", "#66BB6A", "#F57C00", "#26A69A", "#9CCC65", "#1B5E20",
    "#FFB74D", "#4DB6AC", "#AED581", "#00796B", "#FFCC80", "#81C784",
    "#558B2F", "#80CBC4", "#C5E1A5",
]
 
 
def tema_grafik(fig):
    """[PERBAIKAN] Tema grafik Plotly: latar putih, teks hijau tua, tanpa elemen hitam."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#1B4332"),
        title_font=dict(color="#1B5E20", size=17),
        legend=dict(font=dict(color="#1B4332"), bgcolor="rgba(255,255,255,0.85)"),
        hoverlabel=dict(bgcolor="#1B5E20", font=dict(color="#FFFFFF")),
    )
    fig.update_xaxes(gridcolor="#E3F1E6", linecolor="#A5D6A7", zerolinecolor="#A5D6A7",
                     tickfont=dict(color="#4F6F5A"), title_font=dict(color="#1B5E20"))
    fig.update_yaxes(gridcolor="#E3F1E6", linecolor="#A5D6A7", zerolinecolor="#A5D6A7",
                     tickfont=dict(color="#4F6F5A"), title_font=dict(color="#1B5E20"))
    return fig
 
 
def tampil(fig):
    """Tampilkan grafik dengan tema hijau-putih (theme=None agar Streamlit tidak menimpa warna)."""
    st.plotly_chart(tema_grafik(fig), width="stretch", theme=None)
 
st.markdown(
    f"""
    <style>
    .block-container {{padding-top: 1.2rem;}}
    .hero {{
        background: linear-gradient(120deg, #1B5E20 0%, #2E7D32 55%, #66BB6A 100%);
        padding: 1.4rem 1.8rem; border-radius: 16px; color: white; margin-bottom: 1rem;
    }}
    .hero h1 {{margin: 0; font-size: 2.1rem; letter-spacing: 1px; color: white;}}
    .hero p {{margin: .2rem 0 0 0; opacity: .95; font-size: 1rem; color: white;}}
    .kpi {{
        background: #ffffff; border-left: 6px solid {GREEN}; border-radius: 12px;
        padding: .9rem 1.1rem; box-shadow: 0 2px 10px rgba(0,0,0,.08);
    }}
    .kpi .t {{font-size: .8rem; color: #607D8B; text-transform: uppercase; letter-spacing: .5px;}}
    .kpi .v {{font-size: 1.65rem; font-weight: 700; color: #1B5E20; line-height: 1.25;}}
    .kpi .s {{font-size: .8rem; color: #78909C;}}
    .kpi.o {{border-left-color: {ORANGE};}} .kpi.o .v {{color: {ORANGE};}}
    .kpi.b {{border-left-color: {BLUE};}} .kpi.b .v {{color: {BLUE};}}
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
def kpi(title, value, sub="", cls=""):
    st.markdown(
        f'<div class="kpi {cls}"><div class="t">{title}</div>'
        f'<div class="v">{value}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )
 
 
def rp(x, d=0):
    return "Rp " + f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
 
 
# ----------------------------------------------------------------------------
# DATA BAHAN PAKAN (CONTOH / ILUSTRATIF - GANTI DENGAN DATA RIIL PENELITIAN)
# Harga: Rp/kg | ME: kkal/kg | nutrien lain: % bahan | Min/Max: % dalam ransum
# ----------------------------------------------------------------------------
DEFAULT_FEED = pd.DataFrame(
    [
        # Bahan, Kategori, Harga, ME, PK, LK, SK, Ca, P, Lys, Met, Min, Max
        ["Jagung kuning", "Konvensional", 6500, 3350, 8.5, 3.8, 2.5, 0.02, 0.28, 0.26, 0.18, 0, 65],
        ["Bungkil kedelai", "Konvensional", 9500, 2500, 44.0, 1.5, 6.5, 0.30, 0.65, 2.80, 0.60, 0, 40],
        ["Minyak sawit", "Konvensional", 14000, 8000, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 5],
        ["DCP (Dikalsium fosfat)", "Konvensional", 9000, 0, 0.0, 0.0, 0.0, 24.0, 18.0, 0.0, 0.0, 0, 3],
        ["Kapur (CaCO3)", "Konvensional", 1500, 0, 0.0, 0.0, 0.0, 38.0, 0.0, 0.0, 0.0, 0, 3],
        ["DL-Metionin", "Konvensional", 60000, 0, 58.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.0, 0, 0.5],
        ["L-Lisin HCl", "Konvensional", 25000, 0, 95.0, 0.0, 0.0, 0.0, 0.0, 78.0, 0.0, 0, 0.5],
        ["Premix (vitamin & mineral)", "Konvensional", 40000, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4],
        ["Dedak padi", "Lokal", 3500, 2400, 12.5, 11.0, 11.0, 0.07, 1.20, 0.50, 0.25, 0, 20],
        ["Tepung ikan lokal", "Lokal", 11000, 2800, 52.0, 8.0, 1.0, 5.00, 3.00, 3.80, 1.40, 0, 10],
        ["Tepung maggot BSF", "Lokal", 9000, 3000, 42.0, 15.0, 7.0, 5.00, 0.90, 2.40, 0.80, 0, 15],
        ["Bungkil kelapa", "Lokal", 3500, 1540, 21.0, 9.0, 14.0, 0.20, 0.60, 0.60, 0.30, 0, 15],
        ["Bungkil inti sawit", "Lokal", 2800, 1600, 16.0, 8.0, 16.0, 0.25, 0.60, 0.60, 0.25, 0, 15],
        ["Tepung gaplek (singkong)", "Lokal", 4000, 3000, 2.5, 0.7, 3.5, 0.15, 0.08, 0.10, 0.03, 0, 20],
        ["Tepung daun lamtoro", "Lokal", 2500, 1500, 25.0, 4.0, 15.0, 1.50, 0.25, 1.30, 0.30, 0, 5],
    ],
    columns=["Bahan", "Kategori", "Harga", "ME", "PK", "LK", "SK", "Ca", "P", "Lys", "Met", "Min", "Max"],
)
 
NUTR = [
    ("ME", "Energi Metabolis", "kkal/kg"),
    ("PK", "Protein Kasar", "%"),
    ("LK", "Lemak Kasar", "%"),
    ("SK", "Serat Kasar", "%"),
    ("Ca", "Kalsium", "%"),
    ("P", "Fosfor Total", "%"),
    ("Lys", "Lisin", "%"),
    ("Met", "Metionin", "%"),
]
NUTR_LABEL = {k: f"{n} ({u})" for k, n, u in NUTR}
 
# Standar kebutuhan nutrisi (min, max) per fase - dapat diubah pengguna
REQ = {
    "Starter (0-21 hari)": {
        "ME": (3000, 3300), "PK": (21.0, 24.0), "LK": (3.0, 8.0), "SK": (0.0, 5.0),
        "Ca": (0.90, 1.10), "P": (0.60, 1.00), "Lys": (1.20, 1.50), "Met": (0.50, 0.80),
    },
    "Grower (22-35 hari)": {
        "ME": (3100, 3350), "PK": (19.0, 22.0), "LK": (3.0, 8.0), "SK": (0.0, 6.0),
        "Ca": (0.85, 1.05), "P": (0.55, 0.95), "Lys": (1.05, 1.35), "Met": (0.45, 0.75),
    },
    "Finisher (36-42 hari)": {
        "ME": (3150, 3400), "PK": (18.0, 21.0), "LK": (3.0, 8.5), "SK": (0.0, 6.0),
        "Ca": (0.80, 1.00), "P": (0.50, 0.90), "Lys": (0.95, 1.25), "Met": (0.40, 0.70),
    },
}
 
 
# ----------------------------------------------------------------------------
# MODEL LINEAR PROGRAMMING
# ----------------------------------------------------------------------------
def solve_lp(feed: pd.DataFrame, req: dict, allow_local=True, price_override=None):
    """
    Minimisasi  Z = sum(harga_i * x_i / 100)   [Rp per kg pakan]
    dengan x_i = persentase bahan i dalam ransum (0-100)
    Kendala:  sum(x_i) = 100
              min_n <= sum(a_ni * x_i)/100 <= max_n   untuk tiap nutrien n
              L_i <= x_i <= U_i
    """
    f = feed.copy().reset_index(drop=True)
    num_cols = ["Harga", "ME", "PK", "LK", "SK", "Ca", "P", "Lys", "Met", "Min", "Max"]
    f[num_cols] = f[num_cols].astype(float)
    if price_override:
        for k, v in price_override.items():
            f.loc[f["Bahan"] == k, "Harga"] = v
    if not allow_local:
        f.loc[f["Kategori"] == "Lokal", "Max"] = 0
        f.loc[f["Kategori"] == "Lokal", "Min"] = 0
 
    n = len(f)
    c = f["Harga"].values / 100.0
 
    A_ub, b_ub, meta = [], [], []
    for key, label, unit in NUTR:
        lo, hi = req[key]
        coef = f[key].values / 100.0
        if lo is not None and lo > 0:
            A_ub.append(-coef)
            b_ub.append(-lo)
            meta.append((key, "min", lo))
        if hi is not None:
            A_ub.append(coef)
            b_ub.append(hi)
            meta.append((key, "max", hi))
 
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    A_eq = np.ones((1, n))
    b_eq = np.array([100.0])
    bounds = list(zip(f["Min"].values.astype(float), f["Max"].values.astype(float)))
 
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    out = {"status": res.status, "message": res.message, "feed": f, "meta": meta}
    if res.status == 0:
        x = res.x
        out.update(
            x=x,
            cost=res.fun,
            nutr={k: float(np.dot(f[k].values, x) / 100.0) for k, _, _ in NUTR},
            shadow=-res.ineqlin.marginals,  # kenaikan biaya per 1 unit pengetatan kendala
            reduced=res.lower.marginals,  # reduced cost variabel di batas bawah
            res=res,
        )
    return out
 
 
# ----------------------------------------------------------------------------
# SIDEBAR - PARAMETER
# ----------------------------------------------------------------------------
# HEADER NUTRIFEED DENGAN FOTO AYAM
with open("gambar/header_ayam.jpg", "rb") as img:
    img_base64 = base64.b64encode(img.read()).decode("utf-8")
 
# [PERBAIKAN] Lapisan gelap hitam diganti lapisan hijau; teks diberi !important
# agar tidak tertimpa aturan warna judul (sebelumnya tulisan NUTRIFEED jadi hijau & tak terbaca)
st.markdown(
    f"""
    <div class="hero-img" style="
        background-image: linear-gradient(90deg, rgba(20,83,45,0.88), rgba(46,125,50,0.35)),
        url('data:image/jpeg;base64,{img_base64}');
        background-size: cover;
        background-position: center;
        min-height: 230px;
        padding: 40px;
        border-radius: 18px;
        margin-bottom: 20px;
        box-shadow: 0 6px 20px rgba(27,94,32,0.25);
    ">
        <div class="judul">
            🐔 NUTRIFEED
        </div>
        <div class="subjudul">
            INOVASI PAKAN AYAM BROILER BERBASIS BAHAN PAKAN LOKAL
        </div>
        <div class="tagline">
            OPTIMALISASI NUTRISI • EFISIENSI BIAYA • KETAHANAN PANGAN
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
with st.sidebar:
    st.header("⚙️ Parameter Formulasi")
    phase = st.selectbox("Fase pertumbuhan", list(REQ.keys()))
    batch_kg = st.number_input("Jumlah pakan yang dibuat (kg)", 10, 100000, 1000, step=100)
    fcr = st.number_input("FCR (rasio konversi pakan)", 1.0, 3.0, 1.6, step=0.05)
    pop = st.number_input("Populasi ayam (ekor)", 100, 200000, 5000, step=100)
    bobot = st.number_input("Bobot panen rata-rata (kg/ekor)", 0.5, 4.0, 2.0, step=0.1)
 
    st.subheader("Mode bahan pakan")
    mode = st.radio(
        "Bahan yang boleh dipakai",
        ["Lokal + Konvensional (NUTRIFEED)", "Konvensional saja (pembanding)"],
    )
    allow_local = mode.startswith("Lokal")
 
    st.subheader("Kebutuhan nutrisi (min - max)")
    req = {}
    for key, name, unit in NUTR:
        lo0, hi0 = REQ[phase][key]
        step = 10.0 if key == "ME" else 0.05
        c1, c2 = st.columns(2)
        lo = c1.number_input(f"{key} min", value=float(lo0), step=step, key=f"{phase}{key}lo", help=f"{name} ({unit})")
        hi = c2.number_input(f"{key} max", value=float(hi0), step=step, key=f"{phase}{key}hi")
        req[key] = (lo, hi)
 
# ----------------------------------------------------------------------------
# TAB
# ----------------------------------------------------------------------------
tabs = st.tabs(
    [
        "📊 Dashboard",
        "🧪 Formulasi & Nutrisi",
        "🔍 Analisis Sensitivitas",
        "📈 Simulasi Harga",
        "🗂️ Data Bahan",
        "📐 Model LP",
        "ℹ️ Tentang",
    ]
)
 
# --- Tab data bahan dibuat lebih dulu agar hasil edit dipakai seluruh dashboard
with tabs[4]:
    st.subheader("Data harga & kandungan nutrisi bahan pakan")
    st.caption(
        "Data bawaan bersifat **ilustratif**. Ganti dengan harga pasar setempat dan hasil analisis "
        "proksimat laboratorium agar hasil optimasi mencerminkan kondisi nyata. Kolom dapat diedit langsung."
    )
    feed = st.data_editor(
        DEFAULT_FEED,
        num_rows="dynamic",
        width="stretch",
        height=520,
        column_config={
            "Kategori": st.column_config.SelectboxColumn(options=["Lokal", "Konvensional"]),
            "Harga": st.column_config.NumberColumn("Harga (Rp/kg)", format="%d"),
            "ME": st.column_config.NumberColumn("ME (kkal/kg)"),
            "Min": st.column_config.NumberColumn("Min (%)"),
            "Max": st.column_config.NumberColumn("Max (%)"),
        },
    )
    feed = feed.dropna(subset=["Bahan"]).fillna(0)
 
# --- Optimasi utama + pembanding
sol = solve_lp(feed, req, allow_local=allow_local)
base = solve_lp(feed, req, allow_local=False)
full = solve_lp(feed, req, allow_local=True)
 
if sol["status"] != 0:
    for t in (0, 1, 2, 3, 5):
        with tabs[t]:
            st.error(
                "❌ Model **tidak feasible** (tidak ada kombinasi bahan yang memenuhi semua kendala). "
                "Longgarkan batas nutrisi di sidebar, perbesar batas maksimum bahan, atau tambahkan bahan baru."
            )
            st.caption(f"Pesan solver: {sol['message']}")
else:
    f = sol["feed"]
    f["Porsi"] = sol["x"]
    used = f[f["Porsi"] > 1e-6].copy()
    used["Biaya/kg pakan"] = used["Harga"] * used["Porsi"] / 100
    used["Kg per batch"] = used["Porsi"] / 100 * batch_kg
    used["Biaya batch"] = used["Kg per batch"] * used["Harga"]
    cost_kg = sol["cost"]
    local_share = used.loc[used["Kategori"] == "Lokal", "Porsi"].sum()
 
    base_ok = base["status"] == 0
    saving_kg = (base["cost"] - cost_kg) if base_ok else None
 
    total_feed_farm = pop * bobot * fcr
    cost_per_kg_bw = cost_kg * fcr
 
    # ----------------------------------------------------------------- Dashboard
    with tabs[0]:
        c = st.columns(4)
        with c[0]:
            kpi("Biaya pakan minimum", f"{rp(cost_kg)}/kg", f"{rp(cost_kg * 1000)} per ton")
        with c[1]:
            kpi("Biaya batch", rp(cost_kg * batch_kg), f"untuk {batch_kg:,} kg pakan", "b")
        with c[2]:
            kpi("Porsi bahan lokal", f"{local_share:.1f}%", f"{(used['Kategori'] == 'Lokal').sum()} bahan lokal terpilih")
        with c[3]:
            if base_ok and allow_local:
                kpi("Penghematan vs konvensional", f"{rp(saving_kg)}/kg",
                    f"{saving_kg / base['cost'] * 100:.1f}% lebih murah", "o")
            else:
                kpi("Biaya per kg bobot hidup", rp(cost_per_kg_bw), f"FCR {fcr}", "o")
 
        st.write("")
        a, b = st.columns([1.1, 1])
        with a:
            fig = px.pie(
                used, names="Bahan", values="Porsi", hole=0.5,
                color_discrete_sequence=PALET_HIJAU,
                title=f"Komposisi formulasi optimal - {phase}",
            )
            fig.update_traces(textinfo="label+percent", textposition="outside",
                              marker=dict(line=dict(color="#FFFFFF", width=2)))
            fig.update_layout(showlegend=False, margin=dict(t=60, b=10))
            tampil(fig)
        with b:
            fig = px.bar(
                used.sort_values("Porsi"), x="Porsi", y="Bahan", orientation="h",
                color="Kategori", text=used.sort_values("Porsi")["Porsi"].map("{:.2f}%".format),
                color_discrete_map={"Lokal": GREEN, "Konvensional": ORANGE},
                title="Porsi bahan (%)",
            )
            fig.update_layout(margin=dict(t=60, b=10), xaxis_title="% dalam ransum", yaxis_title="")
            tampil(fig)
 
        st.markdown("#### 💰 Perbandingan skenario biaya")
        comp = []
        if base_ok:
            comp.append(("Konvensional saja", base["cost"]))
        if full["status"] == 0:
            comp.append(("Lokal + Konvensional", full["cost"]))
        if comp:
            cdf = pd.DataFrame(comp, columns=["Skenario", "Biaya (Rp/kg)"])
            fig = px.bar(cdf, x="Skenario", y="Biaya (Rp/kg)", color="Skenario", text_auto=",.0f",
                         color_discrete_sequence=[GREY, GREEN])
            fig.update_layout(showlegend=False, height=330, margin=dict(t=20))
            tampil(fig)
            if base_ok and full["status"] == 0:
                sv = base["cost"] - full["cost"]
                st.success(
                    f"Pemanfaatan bahan lokal menurunkan biaya sebesar **{rp(sv)}/kg** "
                    f"({sv / base['cost'] * 100:.1f}%). Untuk kebutuhan peternakan "
                    f"**{total_feed_farm:,.0f} kg pakan** (populasi {pop:,} ekor), potensi penghematan "
                    f"sekitar **{rp(sv * total_feed_farm)}** per siklus."
                )
 
        st.info(
            f"📌 **Rekomendasi**: untuk fase *{phase}*, campurkan "
            + ", ".join(f"**{r.Bahan}** {r.Porsi:.1f}%" for r in used.sort_values("Porsi", ascending=False).itertuples())
            + f". Estimasi biaya {rp(cost_kg)}/kg pakan atau {rp(cost_per_kg_bw)} per kg bobot hidup."
        )
 
    # ----------------------------------------------------------------- Formulasi
    with tabs[1]:
        st.subheader("Rekomendasi formulasi pakan")
        show = used[["Bahan", "Kategori", "Harga", "Porsi", "Kg per batch", "Biaya/kg pakan", "Biaya batch"]].copy()
        show = show.sort_values("Porsi", ascending=False)
        st.dataframe(
            show.style.format(
                {"Harga": "{:,.0f}", "Porsi": "{:.3f}", "Kg per batch": "{:,.2f}",
                 "Biaya/kg pakan": "{:,.2f}", "Biaya batch": "{:,.0f}"}
            ),
            width="stretch", hide_index=True,
        )
        t1, t2 = st.columns(2)
        t1.metric("Total porsi", f"{show['Porsi'].sum():.2f}%")
        t2.metric("Total biaya batch", rp(show["Biaya batch"].sum()))
 
        st.subheader("Kandungan nutrisi vs standar")
        rows = []
        for key, name, unit in NUTR:
            lo, hi = req[key]
            val = sol["nutr"][key]
            ok = (val >= lo - 1e-6) and (val <= hi + 1e-6)
            rows.append([name, unit, lo, hi, val, "✅ Terpenuhi" if ok else "❌ Tidak"])
        ndf = pd.DataFrame(rows, columns=["Nutrien", "Satuan", "Min", "Max", "Hasil formulasi", "Status"])
        st.dataframe(ndf.style.format({"Min": "{:,.2f}", "Max": "{:,.2f}", "Hasil formulasi": "{:,.3f}"}),
                     width="stretch", hide_index=True)
 
        # Persentase terhadap batas minimum (visual)
        fig = go.Figure()
        names = [n for _, n, _ in NUTR]
        pos = []
        for key, _, _ in NUTR:
            lo, hi = req[key]
            pos.append((sol["nutr"][key] - lo) / (hi - lo) * 100 if hi > lo else 100)
        fig.add_trace(go.Bar(x=names, y=pos, marker_color=[GREEN if 0 <= p <= 100 else "#C62828" for p in pos],
                             text=[f"{p:.0f}%" for p in pos], textposition="outside"))
        fig.add_hrect(y0=0, y1=100, fillcolor=GREEN_L, opacity=0.15, line_width=0)
        fig.update_layout(title="Posisi nutrisi dalam rentang standar (0% = batas min, 100% = batas max)",
                          yaxis_title="% rentang", height=380, yaxis_range=[-10, 125])
        tampil(fig)
 
        # Unduh laporan
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            pd.DataFrame(
                {
                    "Parameter": ["Fase", "Mode", "Biaya (Rp/kg)", "Biaya per ton (Rp)", "Biaya batch (Rp)",
                                  "Porsi lokal (%)", "FCR", "Biaya per kg bobot hidup (Rp)"],
                    "Nilai": [phase, mode, round(cost_kg, 2), round(cost_kg * 1000, 0),
                              round(cost_kg * batch_kg, 0), round(local_share, 2), fcr, round(cost_per_kg_bw, 2)],
                }
            ).to_excel(w, sheet_name="Ringkasan", index=False)
            show.to_excel(w, sheet_name="Formulasi", index=False)
            ndf.to_excel(w, sheet_name="Nutrisi", index=False)
        st.download_button("⬇️ Unduh laporan (Excel)", buf.getvalue(),
                           file_name=f"NUTRIFEED_{phase.split()[0]}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
 
    # ----------------------------------------------------------------- Sensitivitas
    with tabs[2]:
        st.subheader("Kendala aktif & harga bayangan (shadow price)")
        st.caption("Harga bayangan = tambahan biaya (Rp/kg pakan) bila kendala diperketat 1 satuan. "
                   "Nilai besar menunjukkan kendala yang paling 'mahal' bagi formulasi.")
        sh = []
        for (key, kind, val), sp in zip(sol["meta"], sol["shadow"]):
            unit = dict((k, u) for k, _, u in NUTR)[key]
            sh.append([NUTR_LABEL[key], "Minimum" if kind == "min" else "Maksimum", val, sp, "Aktif" if abs(sp) > 1e-9 else "Tidak aktif"])
        shd = pd.DataFrame(sh, columns=["Nutrien", "Jenis", "Batas", "Shadow price (Rp/kg per satuan)", "Status"])
        active = shd[shd["Status"] == "Aktif"]
        if len(active):
            st.dataframe(active.style.format({"Batas": "{:,.2f}", "Shadow price (Rp/kg per satuan)": "{:,.4f}"}),
                         width="stretch", hide_index=True)
            fig = px.bar(active, x="Shadow price (Rp/kg per satuan)", y="Nutrien", color="Jenis",
                         orientation="h", title="Kendala paling mempengaruhi biaya",
                         color_discrete_map={"Minimum": GREEN, "Maksimum": ORANGE})
            tampil(fig)
        else:
            st.write("Tidak ada kendala nutrisi aktif.")
 
        st.subheader("Harga break-even bahan yang belum terpilih")
        st.caption("Harga maksimal agar bahan mulai masuk formulasi (hanya jika harga bahan diturunkan).")
        be = []
        for i, r in f.iterrows():
            if r["Porsi"] < 1e-6 and r["Max"] > 0:
                rc = sol["reduced"][i] * 100  # Rp/kg harga
                be.append([r["Bahan"], r["Kategori"], r["Harga"], max(r["Harga"] - rc, 0), rc])
        if be:
            bd = pd.DataFrame(be, columns=["Bahan", "Kategori", "Harga saat ini", "Harga break-even", "Penurunan dibutuhkan"])
            st.dataframe(bd.style.format({"Harga saat ini": "{:,.0f}", "Harga break-even": "{:,.0f}", "Penurunan dibutuhkan": "{:,.0f}"}),
                         width="stretch", hide_index=True)
        else:
            st.write("Semua bahan aktif sudah dipakai dalam formulasi.")
 
    # ----------------------------------------------------------------- Simulasi harga
    with tabs[3]:
        st.subheader("Simulasi perubahan harga bahan")
        ing = st.selectbox("Pilih bahan", list(feed["Bahan"]), index=0)
        p0 = float(feed.loc[feed["Bahan"] == ing, "Harga"].iloc[0])
        rng = st.slider("Rentang perubahan harga (%)", -50, 100, (-40, 60))
        pts = np.linspace(rng[0], rng[1], 21)
        recs = []
        for pct in pts:
            s = solve_lp(feed, req, allow_local=allow_local, price_override={ing: p0 * (1 + pct / 100)})
            if s["status"] == 0:
                share = float(s["x"][s["feed"]["Bahan"] == ing].sum())
                recs.append([pct, p0 * (1 + pct / 100), s["cost"], share])
        if recs:
            sd = pd.DataFrame(recs, columns=["Perubahan (%)", "Harga bahan", "Biaya (Rp/kg)", "Porsi bahan (%)"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sd["Harga bahan"], y=sd["Biaya (Rp/kg)"], name="Biaya pakan",
                                     mode="lines+markers", line=dict(color=GREEN, width=3)))
            fig.add_trace(go.Scatter(x=sd["Harga bahan"], y=sd["Porsi bahan (%)"], name=f"Porsi {ing} (%)",
                                     yaxis="y2", mode="lines+markers", line=dict(color=ORANGE, dash="dash")))
            fig.update_layout(
                xaxis_title=f"Harga {ing} (Rp/kg)", yaxis_title="Biaya pakan (Rp/kg)",
                yaxis2=dict(title="Porsi (%)", overlaying="y", side="right",
                            tickfont=dict(color="#4F6F5A"), title_font=dict(color="#1B5E20")),
                height=450, legend=dict(orientation="h", y=1.1),
            )
            tampil(fig)
            st.dataframe(sd.style.format({"Perubahan (%)": "{:+.0f}", "Harga bahan": "{:,.0f}",
                                          "Biaya (Rp/kg)": "{:,.1f}", "Porsi bahan (%)": "{:.2f}"}),
                         width="stretch", hide_index=True)
        else:
            st.warning("Tidak ada skenario feasible pada rentang ini.")
 
    # ----------------------------------------------------------------- Model LP
    with tabs[5]:
        st.subheader("Formulasi matematis Linear Programming")
        st.markdown("**Variabel keputusan:** $x_i$ = persentase bahan pakan ke-$i$ dalam ransum (% bobot).")
        st.markdown("**Fungsi tujuan (minimisasi biaya):**")
        st.latex(r"\min Z = \sum_{i=1}^{n} \frac{h_i}{100}\, x_i")
        st.markdown("**Kendala:**")
        st.latex(r"\sum_{i=1}^{n} x_i = 100")
        st.latex(r"N_n^{\min} \le \frac{1}{100}\sum_{i=1}^{n} a_{ni}\, x_i \le N_n^{\max}, \quad n=\text{ME, PK, LK, SK, Ca, P, Lys, Met}")
        st.latex(r"L_i \le x_i \le U_i, \qquad x_i \ge 0")
        st.markdown(
            "dengan $h_i$ harga bahan (Rp/kg), $a_{ni}$ kandungan nutrien $n$ pada bahan $i$, "
            "$L_i,U_i$ batas minimum/maksimum penggunaan bahan. Diselesaikan dengan algoritma **HiGHS** "
            "(`scipy.optimize.linprog`)."
        )
        st.markdown(f"**Ukuran model saat ini:** {len(f)} variabel, {len(sol['meta'])} kendala pertidaksamaan, 1 kendala persamaan.")
        st.metric("Status solver", "Optimal ✅")
 
# ----------------------------------------------------------------------------
# TENTANG
# ----------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Latar belakang")
    st.write(
        "Pakan merupakan komponen biaya terbesar dalam usaha peternakan ayam broiler. Ketergantungan pada "
        "bahan pakan komersial meningkatkan biaya produksi, sementara potensi bahan pakan lokal belum "
        "dimanfaatkan secara optimal. Diperlukan optimasi formulasi pakan agar kebutuhan nutrisi terpenuhi "
        "dengan biaya yang efisien."
    )
    st.subheader("Tujuan")
    st.markdown(
        "- Mengoptimalkan kombinasi bahan pakan lokal dan konvensional\n"
        "- Memperhitungkan kebutuhan nutrisi ayam broiler\n"
        "- Mengestimasi biaya formulasi pakan minimum\n"
        "- Memberikan rekomendasi formulasi pakan berbasis data"
    )
    st.subheader("Tahapan metode")
    st.markdown(
        "1. **Pengumpulan data** harga & kandungan nutrisi bahan lokal/konvensional\n"
        "2. **Identifikasi kebutuhan nutrisi** per fase pertumbuhan\n"
        "3. **Pemodelan Linear Programming** (variabel, fungsi tujuan, kendala)\n"
        "4. **Optimasi formulasi** dengan solver HiGHS\n"
        "5. **Pengembangan dashboard** interaktif (Streamlit)"
    )
    st.subheader("Manfaat")
    st.markdown(
        "- Mengoptimalkan biaya formulasi pakan\n- Mendorong pemanfaatan bahan pakan lokal\n"
        "- Membantu memenuhi kebutuhan nutrisi ayam broiler\n- Mendukung keputusan peternak berbasis data"
    )
    st.caption("⚠️ Data bahan & standar nutrisi bawaan bersifat contoh. Validasi dengan data lapangan, "
               "uji proksimat, dan pedoman resmi (misalnya SNI/strain guide) sebelum digunakan di lapangan.")
 