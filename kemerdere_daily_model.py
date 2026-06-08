#!/usr/bin/env python3
"""
Kemerdere GÜNLÜK Hidrolojik Tahmin Modeli
=========================================
Karakurt AYLIK uygulamasının günlük muadili. Hedef: günlük giriş akımı (m³/s),
'Toplam Su (m³)/86400' (Su_Zaman_Serisi_KMR.xlsx). Girdiler GÜNLÜK:
  - Open-Meteo arşivi: günlük yağış / sıcaklık / PET (alan-ağırlıklı, yükseklik bantları)
  - MODIS MOD10A1 günlük kar örtüsü oranı (SCF), GEE ile (anahtarsız, kayıtlı proje)
Model: HİBRİT
  - Model A (otoregresif, Q_lag'li)  -> kısa vade (gün 1-3), özyinelemeli tahmin
  - Model B (hava/kar-temelli)        -> orta vade (gün 4-16), tek-seferde
Her ikisi RF+GBM ensemble. Doğrulama: su-yılı çapraz doğrulama (günlük).

Çalıştırma:  streamlit run kemerdere_daily_model.py
"""

import os, io, json, hashlib, calendar, base64
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xml.etree.ElementTree as ET
import zipfile
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black",
    "ytick.color": "black", "axes.edgecolor": "black",
})
st.set_page_config(page_title="Kemerdere Günlük Akım Tahmin Modeli", page_icon="🏔️", layout="wide")

# ── AUTH ──────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("<div style='text-align:center'><h2>🔐 Kemerdere Akım Tahmin Modeli Girişi</h2></div>", unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        with st.form("login_form"):
            username = st.text_input("Kullanıcı adı")
            password = st.text_input("Şifre", type="password")
            submitted = st.form_submit_button("Giriş", use_container_width=True)
            if submitted:
                users = st.secrets["auth"]["users"]
                for u in users:
                    if u["username"] == username and u["password"] == password:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.rerun()
                st.error("Kullanıcı adı veya şifre hatalı.")
    st.stop()
# ── END AUTH ──────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FLOW = os.path.join(SCRIPT_DIR, "Su_Zaman_Serisi_KMR.xlsx")
DEFAULT_KMZ = os.path.join(SCRIPT_DIR, "3_HEPPs.kmz")
DEFAULT_RIVER = next((os.path.join(SCRIPT_DIR, _f) for _f in (os.listdir(SCRIPT_DIR) if os.path.isdir(SCRIPT_DIR) else [])
                      if _f.lower().endswith((".kmz", ".kml"))
                      and any(k in _f.lower() for k in ("river", "nehir", "akarsu", "stream"))), None)
TARGET = "Q_m3s"

# =====================================================================
# GEOMETRİ / NOKTA YARDIMCILARI  (generic sürümden port)
# =====================================================================
def parse_kmz_polygon(src):
    """KMZ/KML'den ilk poligonu (lat,lon köşeleri) ve bbox'u çıkarır. src: dosya yolu
    ya da .read()/.seek() destekleyen yüklenmiş dosya."""
    try:
        if hasattr(src, "read"):
            file_bytes = src.read(); src.seek(0)
        else:
            with open(src, "rb") as f: file_bytes = f.read()
        if zipfile.is_zipfile(io.BytesIO(file_bytes)):
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                kmls = [f for f in z.namelist() if f.endswith('.kml')]
                if not kmls: return None
                kml_content = z.read(kmls[0])
        else:
            kml_content = file_bytes
        root = ET.fromstring(kml_content)
        ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        for pm in root.findall(f".//{ns}Placemark"):
            ring = pm.find(f".//{ns}Polygon//{ns}coordinates")
            if ring is None:
                ring = pm.find(f".//{ns}LinearRing/{ns}coordinates")
            if ring is not None and ring.text:
                verts = []
                for tok in ring.text.strip().split():
                    parts = tok.split(',')
                    if len(parts) >= 2:
                        verts.append((float(parts[1]), float(parts[0])))  # (lat, lon)
                if len(verts) >= 3:
                    lats = [v[0] for v in verts]; lons = [v[1] for v in verts]
                    return {"verts": verts, "bbox": (min(lons), min(lats), max(lons), max(lats))}
    except Exception as e:
        st.sidebar.error(f"Poligon Ayrıştırma Hatası: {e}")
    return None

def parse_kmz_lines(src):
    """KMZ/KML'den çizgi (LineString) geometrileri — dereler/akarsular. src: yol ya da yüklenmiş dosya.
    Döndürür: [{"name":..., "coords":[(lat,lon),...]}, ...] ya da None."""
    lines = []
    try:
        if hasattr(src, "read"):
            file_bytes = src.read(); src.seek(0)
        else:
            with open(src, "rb") as f: file_bytes = f.read()
        if zipfile.is_zipfile(io.BytesIO(file_bytes)):
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                kmls = [f for f in z.namelist() if f.endswith('.kml')]
                if not kmls: return None
                kml_content = z.read(kmls[0])
        else:
            kml_content = file_bytes
        root = ET.fromstring(kml_content)
        ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        for pm in root.findall(f".//{ns}Placemark"):
            nm = pm.find(f"{ns}name"); name = nm.text.strip() if (nm is not None and nm.text) else "Dere"
            for ls in pm.findall(f".//{ns}LineString/{ns}coordinates"):
                if ls.text:
                    coords = []
                    for tok in ls.text.strip().split():
                        p = tok.split(',')
                        if len(p) >= 2: coords.append((float(p[1]), float(p[0])))  # (lat,lon)
                    if len(coords) >= 2: lines.append({"name": name, "coords": coords})
    except Exception as e:
        st.sidebar.error(f"Dereler KMZ Hatası: {e}")
    return lines if lines else None

def basin_hash(verts):
    s = json.dumps([[round(float(v[0]), 5), round(float(v[1]), 5)] for v in verts])
    return hashlib.md5(s.encode()).hexdigest()[:10]

def polygon_area_km2(verts):
    if not verts or len(verts) < 3: return None
    lats = np.array([v[0] for v in verts], float); lons = np.array([v[1] for v in verts], float)
    lat0 = np.radians(lats.mean()); R = 6371.0
    x = np.radians(lons) * R * np.cos(lat0); y = np.radians(lats) * R
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def _poly_path(verts):
    from matplotlib.path import Path as MPath
    return MPath([(v[1], v[0]) for v in verts])

def generate_grid_points(poly, target_n=16):
    min_lon, min_lat, max_lon, max_lat = poly["bbox"]; path = _poly_path(poly["verts"]); pts = []
    for res in range(3, 80):
        gx = np.linspace(min_lon, max_lon, res); gy = np.linspace(min_lat, max_lat, res)
        GX, GY = np.meshgrid(gx, gy)
        inside = path.contains_points(np.column_stack([GX.ravel(), GY.ravel()]))
        pts = [(float(la), float(lo)) for lo, la, ins in zip(GX.ravel(), GY.ravel(), inside) if ins]
        if len(pts) >= target_n: return pts
    return pts if pts else [((min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0)]

def thiessen_weights(points, poly, res=120):
    min_lon, min_lat, max_lon, max_lat = poly["bbox"]; path = _poly_path(poly["verts"])
    gx = np.linspace(min_lon, max_lon, res); gy = np.linspace(min_lat, max_lat, res)
    GX, GY = np.meshgrid(gx, gy)
    inside = path.contains_points(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    plon = np.array([p[1] for p in points]); plat = np.array([p[0] for p in points])
    ys, xs = np.where(inside)
    if len(ys) == 0: return np.ones(len(points)) / len(points)
    cell_lon = gx[xs]; cell_lat = gy[ys]
    d2 = (cell_lon[:, None] - plon[None, :]) ** 2 + (cell_lat[:, None] - plat[None, :]) ** 2
    nearest = np.argmin(d2, axis=1)
    counts = np.array([np.sum(nearest == k) for k in range(len(points))], float)
    return counts / counts.sum() if counts.sum() > 0 else np.ones(len(points)) / len(points)

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_point_elevations(coords):
    try:
        lats = ",".join(str(c[0]) for c in coords); lons = ",".join(str(c[1]) for c in coords)
        r = requests.get("https://api.open-meteo.com/v1/elevation",
                         params={"latitude": lats, "longitude": lons}, timeout=30)
        if r.status_code == 200:
            el = r.json().get("elevation")
            if el and len(el) == len(coords): return np.array(el, float)
    except Exception:
        pass
    return None

def build_elevation_bands(grid_points, elevations, n_bands=4):
    grid_points = list(grid_points); elev = np.asarray(elevations, float)
    n_bands = int(max(1, min(n_bands, len(grid_points))))
    edges = np.quantile(elev, np.linspace(0, 1, n_bands + 1)); edges[0] -= 1; edges[-1] += 1
    bands = []
    for bi in range(n_bands):
        mask = (elev >= edges[bi]) & (elev < edges[bi + 1])
        if mask.sum() == 0: continue
        idxs = np.where(mask)[0]; med = np.median(elev[idxs])
        rep = idxs[np.argmin(np.abs(elev[idxs] - med))]
        bands.append({"name": f"Bant {bi+1} (~{int(med)} m)",
                      "lat": float(grid_points[rep][0]), "lon": float(grid_points[rep][1]),
                      "area_fraction": float(mask.sum()) / len(grid_points)})
    tot = sum(b["area_fraction"] for b in bands) or 1.0
    for b in bands: b["area_fraction"] /= tot
    return bands

def auto_generate_points(poly, snow_dominated, n=16, n_bands=4):
    if snow_dominated:
        grid = generate_grid_points(poly, target_n=max(n, n_bands * 4))
        elev = fetch_point_elevations(grid)
        if elev is not None:
            bands = build_elevation_bands(grid, elev, n_bands=n_bands)
            if bands: return bands
    grid = generate_grid_points(poly, target_n=n)
    w = thiessen_weights(grid, poly)
    return [{"name": f"Nokta {i+1}", "lat": float(p[0]), "lon": float(p[1]),
             "area_fraction": float(w[i])} for i, p in enumerate(grid)]

def points_to_kmz_bytes(bands, basin_area_km2=None):
    import xml.sax.saxutils as _sx
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2">',
             '<Document>', '<name>Kemerdere Model Noktaları</name>']
    for b in bands:
        af = float(b.get("area_fraction", 0.0))
        desc = (f"Alan payı: {af:.4f} ({af*basin_area_km2:.2f} km2)" if basin_area_km2 else f"Alan payı: {af:.4f}")
        parts += ['<Placemark>', f'<name>{_sx.escape(str(b["name"]))}</name>',
                  f'<description>{_sx.escape(desc)}</description>',
                  f'<Point><coordinates>{b["lon"]:.6f},{b["lat"]:.6f},0</coordinates></Point>', '</Placemark>']
    parts.append('</Document></kml>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", "\n".join(parts))
    return buf.getvalue()

# =====================================================================
# GEE — GÜNLÜK MODIS SCF (MOD10A1)
# =====================================================================
@st.cache_resource(show_spinner=False)
def init_gee(project=None):
    try:
        import ee
    except Exception as e:
        return False, f"earthengine-api kurulu değil: {e}"
    try:
        sa = None
        try: sa = st.secrets.get("GEE_SERVICE_ACCOUNT")
        except Exception: sa = None
        if not sa:
            sa_path = os.environ.get("GEE_SERVICE_ACCOUNT_JSON_PATH")
            if sa_path and os.path.isfile(sa_path):
                with open(sa_path) as f:
                    sa = f.read()
        if sa:
            info = json.loads(sa) if isinstance(sa, str) else dict(sa)
            creds = ee.ServiceAccountCredentials(info["client_email"], key_data=json.dumps(info))
            ee.Initialize(creds, project=project or info.get("project_id"))
        elif project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        return True, "ok"
    except Exception as e:
        return False, str(e)

def build_modis_scf_daily_gee(verts, start_date, end_date, snow_ndsi=40, product="MODIS/061/MOD10A1"):
    """GEE'de MOD10A1 NDSI_Snow_Cover'dan havza-ortalama GÜNLÜK SCF (0-1). Bulut maskeli.
    Ay-ay FeatureCollection (gün başına reduceRegion). Disk cache: modis_scf_daily_{hash}.csv."""
    cache_file = os.path.join(SCRIPT_DIR, f"modis_scf_daily_{basin_hash(verts)}.csv")
    start_ts = pd.Timestamp(start_date).normalize(); end_ts = pd.Timestamp(end_date).normalize()
    if os.path.exists(cache_file):
        try:
            dfc = pd.read_csv(cache_file); dfc["Date"] = pd.to_datetime(dfc["Date"])
            # ARALIK-bazlı kontrol (hava cache'i gibi): cache başlangıcı<=istenen ve sonu>=istenen ise kullan.
            # MODIS'te bazı günler görüntü YOK -> o günler eksik olur; bu NORMALDİR ve refetch tetiklememeli
            # (eski 'her gün mevcut olmalı' kuralı bu yüzden her run'da yeniden çekiyordu).
            if not dfc.empty and dfc["Date"].min() <= start_ts and dfc["Date"].max() >= end_ts:
                return dfc.sort_values("Date").reset_index(drop=True)
        except Exception:
            pass
    import ee
    geom = ee.Geometry.Polygon([[[float(v[1]), float(v[0])] for v in verts]])
    ic = ee.ImageCollection(product).select("NDSI_Snow_Cover")
    def snow_img(img):
        valid = img.lte(100); snow = img.gte(snow_ndsi).And(valid)
        return snow.updateMask(valid).rename("snow")
    months = pd.date_range(start_ts.replace(day=1), end_ts, freq="MS")
    rows = []
    prog = st.progress(0.0, text="GEE'den GÜNLÜK SCF çekiliyor (bir kez, sonra cache)...")
    for mi, m0 in enumerate(months):
        base = ee.Date.fromYMD(int(m0.year), int(m0.month), 1)
        dim = calendar.monthrange(int(m0.year), int(m0.month))[1]
        def daily(d):
            s = base.advance(ee.Number(d), "day"); e = s.advance(1, "day")
            img = ic.filterDate(s, e).map(snow_img).mean()
            val = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=500, maxPixels=1e9).get("snow")
            return ee.Feature(None, {"date": s.format("YYYY-MM-dd"), "scf": val})
        fc = ee.FeatureCollection(ee.List.sequence(0, dim - 1).map(daily))
        try:
            for f in fc.getInfo()["features"]:
                p = f["properties"]; sc = p.get("scf")
                rows.append({"Date": pd.Timestamp(p["date"]), "SCF_basin": float(sc) if sc is not None else np.nan})
        except Exception:
            pass
        prog.progress((mi + 1) / len(months), text=f"GEE günlük SCF: {m0.strftime('%Y-%m')} ({mi+1}/{len(months)})")
    prog.empty()
    df = pd.DataFrame(rows)
    if df.empty: return df
    df = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].sort_values("Date").reset_index(drop=True)
    try: df.to_csv(cache_file, index=False)
    except Exception: pass
    return df

def scf_doy_climatology(df_scf):
    if df_scf is None or df_scf.empty: return {}
    s = df_scf.dropna(subset=["SCF_basin"]).copy(); s["doy"] = s["Date"].dt.dayofyear
    return s.groupby("doy")["SCF_basin"].mean().to_dict()

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_current_scf(verts, today_str, snow_ndsi=40, days_back=30, product="MODIS/061/MOD10A1"):
    """Son ~days_back günde havza ortalama EN GÜNCEL gözlenen SCF (0-1) — GEE.
    Tahmin SCF'sini haritadaki güncel kara sabitlemek için. None olabilir."""
    try:
        import ee
    except Exception:
        return None
    try:
        geom = ee.Geometry.Polygon([[[float(v[1]), float(v[0])] for v in verts]])
        end = ee.Date(today_str); start = end.advance(-int(days_back), "day")
        ic = ee.ImageCollection(product).select("NDSI_Snow_Cover").filterDate(start, end)
        def daily(img):
            valid = img.lte(100); snow = img.gte(snow_ndsi).And(valid)
            v = snow.updateMask(valid).rename("snow").reduceRegion(
                ee.Reducer.mean(), geom, 500, maxPixels=1e9).get("snow")
            return ee.Feature(None, {"date": img.date().format("YYYY-MM-dd"), "scf": v})
        feats = ee.FeatureCollection(ic.map(daily)).getInfo()["features"]
        rows = [(f["properties"]["date"], f["properties"]["scf"]) for f in feats
                if f["properties"].get("scf") is not None]
        if not rows: return None
        rows.sort()
        last = rows[-3:]   # son 3 geçerli gün ortalaması (tek-gün MODIS bulut gürültüsünü yumuşat)
        return float(np.mean([s for _, s in last])), last[-1][0]   # (SCF, en güncel tarih)
    except Exception:
        return None

# =====================================================================
# OPEN-METEO GÜNLÜK HAVA  (process_raw_api_list generic'ten port)
# =====================================================================
def process_raw_api_list(api_response, _bands):
    """Open-Meteo çok-noktalı GÜNLÜK yanıtını birleşik tabloya çevirir (Date, T_i,P_i,PET_i)."""
    if isinstance(api_response, dict) and "error" in api_response:
        st.error(f"🚨 Open-Meteo Reddi: {api_response.get('reason','Limit')}"); st.stop()
    results_list = [api_response] if isinstance(api_response, dict) else api_response
    zone_records = []
    for idx, zd in enumerate(results_list):
        if not isinstance(zd, dict) or "daily" not in zd:
            st.error(f"🚨 İstasyon engeli: {zd.get('reason') if isinstance(zd,dict) else 'Hatalı JSON'}"); st.stop()
        src = zd["daily"]
        df_d = pd.DataFrame({"Date": pd.to_datetime(src["time"]), "T_mean": src["temperature_2m_mean"],
                             "P": src["precipitation_sum"], "PET": src["et0_fao_evapotranspiration"]}).ffill()
        df_d = df_d.rename(columns={"T_mean": f"T_{idx}", "P": f"P_{idx}", "PET": f"PET_{idx}"})
        zone_records.append(df_d[["Date", f"T_{idx}", f"P_{idx}", f"PET_{idx}"]])
    df_merged = zone_records[0]
    for dn in zone_records[1:]: df_merged = pd.merge(df_merged, dn, on="Date", how="inner")
    return df_merged

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_daily_archive(coords_str, _bands, start_str, end_str, cache_file=None):
    if cache_file and os.path.exists(cache_file):
        try:
            dfc = pd.read_csv(cache_file); dfc["Date"] = pd.to_datetime(dfc["Date"])
            if dfc["Date"].min() <= pd.Timestamp(start_str) and dfc["Date"].max() >= pd.Timestamp(end_str):
                return dfc
        except Exception: pass
    params = {"latitude": [b["lat"] for b in _bands], "longitude": [b["lon"] for b in _bands],
              "start_date": start_str, "end_date": end_str,
              "daily": ["temperature_2m_mean", "precipitation_sum", "et0_fao_evapotranspiration"], "timezone": "auto"}
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=120)
    if r.status_code == 429: st.error("⏳ Open-Meteo hız sınırı. 60 sn bekleyin."); st.stop()
    dfw = process_raw_api_list(r.json(), _bands)
    if cache_file:
        try: dfw.to_csv(cache_file, index=False)
        except Exception: pass
    return dfw

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily_forecast(coords_str, _bands, past_days=14):
    params = {"latitude": [b["lat"] for b in _bands], "longitude": [b["lon"] for b in _bands],
              "daily": ["temperature_2m_mean", "precipitation_sum", "et0_fao_evapotranspiration"],
              "past_days": past_days, "forecast_days": 16, "timezone": "auto"}
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=60)
    if r.status_code == 429: st.error("⏳ Tahmin limiti. 1 dk bekleyin."); st.stop()
    return process_raw_api_list(r.json(), _bands)

# =====================================================================
# METRİKLER / KAR HAZNESİ / MODELLER  (generic'ten port)
# =====================================================================
def nse(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    d = np.sum((obs - obs.mean()) ** 2)
    return 1.0 - np.sum((obs - sim) ** 2) / d if d > 0 else np.nan

def log_nse(obs, sim, eps=0.01):
    o, s = np.log(np.asarray(obs, float) + eps), np.log(np.asarray(sim, float) + eps)
    d = np.sum((o - o.mean()) ** 2)
    return 1.0 - np.sum((o - s) ** 2) / d if d > 0 else np.nan

def kge(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    if obs.std() == 0 or sim.std() == 0: return -1e9
    r = np.corrcoef(obs, sim)[0, 1]
    return 1.0 - np.sqrt((r - 1) ** 2 + (sim.std()/obs.std() - 1) ** 2 + (sim.mean()/obs.mean() - 1) ** 2)

def pbias(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    return 100.0 * np.sum(sim - obs) / np.sum(obs) if np.sum(obs) != 0 else np.nan

def all_metrics(obs, sim):
    obs = np.asarray(obs, float)
    if len(obs) < 2: return {"NSE": np.nan, "logNSE": np.nan, "KGE": np.nan, "PBIAS_%": np.nan}
    return {"NSE": nse(obs, sim), "logNSE": log_nse(obs, sim), "KGE": kge(obs, sim), "PBIAS_%": pbias(obs, sim)}

PEAK_QUANTILE = 0.90
def peak_metrics(obs, sim, q=PEAK_QUANTILE):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    if len(obs) < 10: return {"Peak_NSE": np.nan, "Peak_PBIAS_%": np.nan, "Peak_capture": np.nan}
    thr = np.quantile(obs, q); mask = obs >= thr
    if mask.sum() < 3: return {"Peak_NSE": np.nan, "Peak_PBIAS_%": np.nan, "Peak_capture": np.nan}
    o, s = obs[mask], sim[mask]
    pn = 1.0 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2) if np.sum((o - o.mean())**2) > 0 else np.nan
    return {"Peak_NSE": pn, "Peak_PBIAS_%": 100.0*np.sum(s-o)/np.sum(o), "Peak_capture": s.mean()/o.mean() if o.mean()>0 else np.nan}

def peak_sample_weight(y, q=PEAK_QUANTILE, hi=4.0, lo=1.0):
    y = np.asarray(y, float); thr = np.quantile(y, q); w = np.ones_like(y) * lo; above = y >= thr
    if above.sum() > 0:
        ymax = y.max()
        w[above] = lo + (hi-lo)*(y[above]-thr)/(ymax-thr) if ymax > thr else hi
    return w

def y_forward(y): return np.sqrt(np.clip(np.asarray(y, float), 0, None))
def y_inverse(z): return np.clip(np.asarray(z, float), 0, None) ** 2

def run_snow_bucket(P_series, T_series, Tsnow=1.0, Tmelt=0.0, ddf=4.0):
    """GÜNLÜK derece-gün kar haznesi. ddf birimi: mm/°C/GÜN."""
    snow = 0.0; snowpack, melt_out = [], []
    for P, T in zip(np.asarray(P_series, float), np.asarray(T_series, float)):
        if T <= Tsnow: snow += P; melt = 0.0
        else: melt = min(ddf * max(T - Tmelt, 0.0), snow); snow -= melt
        snowpack.append(snow); melt_out.append(melt)
    return np.array(snowpack), np.array(melt_out)

def water_year(dates):
    d = pd.to_datetime(dates)
    return np.where(d.dt.month >= 10, d.dt.year + 1, d.dt.year)

def make_rf(): return RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=12, max_features=0.5, random_state=42, n_jobs=-1)
def make_gbm(): return GradientBoostingRegressor(n_estimators=350, learning_rate=0.03, max_depth=3, min_samples_leaf=20, subsample=0.7, random_state=42)
MODELS = {"Random Forest": make_rf, "Gradient Boosting": make_gbm}

# =====================================================================
# GÜNLÜK VERİ HATTI: yükleme + özellik mühendisliği
# =====================================================================
# Akım birimi -> m³/s çarpanı (GENERİK: her formattaki günlük akım dosyası için)
UNIT_TO_M3S = {
    "m³/gün (günlük hacim)": 1.0 / 86400.0,
    "m³/s (debi)": 1.0,
    "hm³/gün": 1e6 / 86400.0,
    "L/s": 1.0 / 1000.0,
    "1000 m³/gün": 1000.0 / 86400.0,
}

def load_daily_flow(source, sheet=None, date_col=None, q_col=None, unit_factor=1.0/86400.0, start=None):
    """GENERİK günlük akım yükleyici -> DataFrame[Date, Q_m3s].
    sheet/date_col/q_col verilmezse otomatik seçilir (ilk sayfa; tarih=ilk/'tarih' içeren;
    akım='toplam su' içeren ya da son sütun). unit_factor: kaynağı m³/s'ye çevirir.
    start: 'YYYY-MM-DD' (devreye-alma vb. ilk dönemi atmak için) ya da None=tümü."""
    src = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    df = pd.read_excel(src, sheet_name=sheet if sheet is not None else 0)
    df.columns = [str(c).strip() for c in df.columns]
    if date_col is None or date_col not in df.columns:
        date_col = next((c for c in df.columns if any(k in c.lower() for k in ("tarih", "date", "zaman"))), df.columns[0])
    if q_col is None or q_col not in df.columns:
        q_col = next((c for c in df.columns if "toplam su" in c.lower()), df.columns[-1])
    date = pd.to_datetime(df[date_col], errors="coerce")
    q = pd.to_numeric(df[q_col], errors="coerce") * float(unit_factor)
    out = pd.DataFrame({"Date": date, "Q_m3s": q}).dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if start:
        out = out[out["Date"] >= pd.Timestamp(start)].reset_index(drop=True)
    return out

FEAT_W = ["P_sum", "T_mean", "PET_sum", "P_roll3", "P_roll7", "P_roll15", "P_roll30", "P_roll60", "P_roll90",
          "P_lag1", "P_lag2", "wbal_roll30", "API", "T_roll7", "snowpack", "melt", "melt_lag1", "melt_roll7",
          "res95", "res85", "sin_doy", "cos_doy"]
SCF_FEATS = ["SCF", "SCF_lag1", "SCF_lag7"]
AR_FEATS = ["Q_lag1", "Q_lag2", "Q_lag3", "Q_lag7"]

def feats_weather(df): return FEAT_W + [f for f in SCF_FEATS if f in df.columns]
def feats_ar(df): return feats_weather(df) + AR_FEATS

def engineer_daily_features(df, snow_params, weights, scf_doy_clim=None):
    df = df.sort_values("Date").reset_index(drop=True).copy()
    nb = len(weights)
    df["P_sum"] = sum(df[f"P_{i}"] * weights[i] for i in range(nb))
    df["T_mean"] = sum(df[f"T_{i}"] * weights[i] for i in range(nb))
    df["PET_sum"] = sum(df[f"PET_{i}"] * weights[i] for i in range(nb))
    for w_ in (3, 7, 15, 30): df[f"P_roll{w_}"] = df["P_sum"].rolling(w_, min_periods=1).sum()
    df["T_roll7"] = df["T_mean"].rolling(7, min_periods=1).mean()
    sp = snow_params or {}
    snowpack, melt = run_snow_bucket(df["P_sum"], df["T_mean"], Tsnow=sp.get("Tsnow", 1.0),
                                     Tmelt=sp.get("Tmelt", 0.0), ddf=sp.get("ddf", 4.0))
    df["snowpack"] = snowpack; df["melt"] = melt
    df["melt_lag1"] = df["melt"].shift(1); df["melt_roll7"] = df["melt"].rolling(7, min_periods=1).sum()
    for w_ in (60, 90): df[f"P_roll{w_}"] = df["P_sum"].rolling(w_, min_periods=1).sum()
    df["P_lag1"] = df["P_sum"].shift(1); df["P_lag2"] = df["P_sum"].shift(2)
    df["wbal_roll30"] = (df["P_sum"] - df["PET_sum"]).rolling(30, min_periods=1).sum()  # su dengesi (P-PET)
    _p = df["P_sum"].values; _api = 0.0; _aa = np.empty(len(_p))   # API: çürütülmüş antesedan yağış (nem vekili)
    for _i in range(len(_p)): _api = 0.92 * _api + _p[_i]; _aa[_i] = _api
    df["API"] = _aa
    # Lineer rezervuar HAFIZA (gözlenen Q kullanmaz): runoff=0.3P+0.8melt -> S=k·S+runoff. Baz akış/depolama vekili,
    # Model B'ye 'hafıza' verir (resesyon temsili) -> kör test belirgin iyileşir.
    _ro = 0.3 * df["P_sum"].values + 0.8 * df["melt"].values
    for _k in (0.95, 0.85):
        _S = 0.0; _ss = np.empty(len(_ro))
        for _i in range(len(_ro)): _S = _k * _S + _ro[_i]; _ss[_i] = _S
        df[f"res{int(_k*100)}"] = _ss
    doy = df["Date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25); df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    if "SCF_basin" in df.columns:
        scf = df["SCF_basin"].astype(float).copy()
        if scf_doy_clim:
            scf = scf.fillna(df["Date"].dt.dayofyear.map(scf_doy_clim))
        scf = scf.fillna(scf.median()).fillna(0.0)
        df["SCF"] = scf; df["SCF_lag1"] = scf.shift(1); df["SCF_lag7"] = scf.shift(7)
    if "Q_m3s" in df.columns:
        for L in (1, 2, 3, 7): df[f"Q_lag{L}"] = df["Q_m3s"].shift(L)
    return df

# =====================================================================
# HİBRİT MODEL: eğitim + su-yılı CV
# =====================================================================
def loo_water_year_cv(df_feat, feats, model_ctor, peak_focus=False):
    df = df_feat.copy(); df["WY"] = water_year(df["Date"])
    obs, sim, per = [], [], {}
    for wy in sorted(df["WY"].unique()):
        tr = df[df["WY"] != wy]; te = df[df["WY"] == wy]
        if len(te) < 30 or len(tr) < 200: continue
        m = model_ctor(); sw = peak_sample_weight(tr[TARGET].values) if peak_focus else None
        m.fit(tr[feats], y_forward(tr[TARGET]), sample_weight=sw)
        pred = y_inverse(m.predict(te[feats]))
        obs.extend(te[TARGET].values); sim.extend(pred); per[int(wy)] = nse(te[TARGET].values, pred)
    metrics = all_metrics(obs, sim); metrics.update(peak_metrics(obs, sim))
    return metrics, per

def train_models(df_feat, feats, peak_focus):
    trained, results = {}, {}
    for name, ctor in MODELS.items():
        cvm, per = loo_water_year_cv(df_feat, feats, ctor, peak_focus)
        m = ctor(); sw = peak_sample_weight(df_feat[TARGET].values) if peak_focus else None
        m.fit(df_feat[feats], y_forward(df_feat[TARGET]), sample_weight=sw)
        trained[name] = m; results[name] = {"cv": cvm, "per_year": per}
    ns = {n: max(0.01, results[n]["cv"].get("NSE", 0.01) if np.isfinite(results[n]["cv"].get("NSE", np.nan)) else 0.01) for n in trained}
    tot = sum(ns.values()); ens = {n: ns[n] / tot for n in trained}
    return {"trained": trained, "results": results, "ens": ens, "feats": feats}

@st.cache_resource(show_spinner=False)
def train_hybrid(_df_feat, peak_focus, cache_key):
    """A (otoregresif) ve B (hava-temelli) ensemble'larını eğitir."""
    dfA = _df_feat.dropna(subset=feats_ar(_df_feat) + [TARGET]).reset_index(drop=True)
    dfB = _df_feat.dropna(subset=feats_weather(_df_feat) + [TARGET]).reset_index(drop=True)
    A = train_models(dfA, feats_ar(_df_feat), peak_focus)
    B = train_models(dfB, feats_weather(_df_feat), peak_focus)
    return A, B

@st.cache_resource(show_spinner=False)
def calibrate_snow_daily(_df_raw, _weights, _scf_clim, cache_key):
    """ddf/Tsnow'u GERÇEKTEN kalibre eder: küçük grid aramayla, hava-temelli (B) modelin
    kronolojik %70/%30 doğrulama NSE'sini maksimize eden kombinasyonu seçer.
    Döndürür: (snow_params, best_nse)."""
    best = {"ddf": 4.0, "Tsnow": 1.0, "Tmelt": 0.0}; best_s = -1e9
    for ddf in (2.0, 3.0, 4.0, 6.0, 8.0, 10.0):
        for Tsnow in (0.0, 1.0, 2.0):
            for Tmelt in (-1.0, 0.0, 1.0):
                if Tmelt > Tsnow:   # fiziksel: erime eşiği <= kar eşiği
                    continue
                sp = {"ddf": ddf, "Tsnow": Tsnow, "Tmelt": Tmelt}
                d = engineer_daily_features(_df_raw, sp, _weights, _scf_clim)
                feats = feats_weather(d)
                d = d.dropna(subset=feats + [TARGET]).sort_values("Date").reset_index(drop=True)
                if len(d) < 300:
                    continue
                k = int(len(d) * 0.7); tr = d.iloc[:k]; te = d.iloc[k:]
                if len(tr) < 200 or len(te) < 60:
                    continue
                m = GradientBoostingRegressor(n_estimators=80, learning_rate=0.08, max_depth=3,
                                              subsample=0.8, random_state=42)
                m.fit(tr[feats], y_forward(tr[TARGET]))
                s = nse(te[TARGET].values, y_inverse(m.predict(te[feats])))
                if s > best_s:
                    best_s, best = s, sp
    return best, best_s

def predict_ensemble(model_pack, X):
    pred = np.zeros(len(X))
    for n, m in model_pack["trained"].items():
        pred += model_pack["ens"][n] * y_inverse(m.predict(X[model_pack["feats"]]))
    return pred

def hybrid_forecast(feat_live, A, B, today, Q0, n_fc=16):
    """feat_live: sürekli günlük özellikler (bağlam + tahmin). Döndürür: tahmin günlerinin DataFrame'i
    Q_A (özyinelemeli), Q_B (hava-temelli), Q_hybrid (gün1-3 A, gün4-16 B)."""
    fc = feat_live[feat_live["Date"] >= today].head(n_fc).reset_index(drop=True)
    if fc.empty: return fc
    # Model B: hava-temelli, tek seferde
    qB = predict_ensemble(B, fc)
    # Model A: özyinelemeli (Q_lag'leri Q0 ile tohumla)
    qhist = [float(Q0)] * 8
    qA = []
    for i in range(len(fc)):
        row = fc.iloc[[i]].copy()
        row["Q_lag1"] = qhist[-1]; row["Q_lag2"] = qhist[-2]
        row["Q_lag3"] = qhist[-3]; row["Q_lag7"] = qhist[-7]
        p = float(predict_ensemble(A, row)[0])
        qA.append(p); qhist.append(p)
    qA = np.array(qA)
    qhyb = qB.copy(); qhyb[:3] = qA[:3]
    out = pd.DataFrame({"Date": fc["Date"], "Q_A": qA, "Q_B": qB, "Q_hybrid": qhyb,
                        "P_sum": fc["P_sum"], "SCF": fc.get("SCF", np.nan)})
    out["gun"] = np.arange(1, len(out) + 1)
    out["reliability"] = np.where(out["gun"] <= 3, "Yüksek (1-3, otoregresif)",
                          np.where(out["gun"] <= 7, "Orta (4-7)", "Belirsiz (8-16)"))
    return out

def q_doy_envelope(df_obs, dates, col=TARGET, window=7):
    """Gözlenen günlük seriden (col) gün-of-yıl (±window) min/ort/max zarfı; `dates`'e hizalı."""
    if df_obs is None or df_obs.empty or col not in df_obs.columns:
        nanarr = np.full(len(list(dates)), np.nan); return nanarr, nanarr.copy(), nanarr.copy()
    s = df_obs.dropna(subset=[col]).copy(); s["doy"] = s["Date"].dt.dayofyear
    mn, me, mx = [], [], []
    for d in pd.to_datetime(dates):
        diff = np.abs(((s["doy"] - d.dayofyear + 182) % 365) - 182)
        sub = s.loc[diff <= window, col]
        if len(sub): mn.append(sub.min()); me.append(sub.mean()); mx.append(sub.max())
        else: mn.append(np.nan); me.append(np.nan); mx.append(np.nan)
    return np.array(mn), np.array(me), np.array(mx)

def chrono_split_daily(df_feat, feats, split_ts, peak_focus=False):
    """Kronolojik train/test: split_ts öncesi eğitim, sonrası KÖR test. RF+GBM eşit-ağırlık ensemble."""
    tr = df_feat[df_feat["Date"] < split_ts].dropna(subset=feats + [TARGET])
    te = df_feat[df_feat["Date"] >= split_ts].dropna(subset=feats + [TARGET])
    if len(tr) < 100 or len(te) < 30: return None
    ptr = np.zeros(len(tr)); pte = np.zeros(len(te)); k = len(MODELS)
    for ctor in MODELS.values():
        m = ctor(); sw = peak_sample_weight(tr[TARGET].values) if peak_focus else None
        m.fit(tr[feats], y_forward(tr[TARGET]), sample_weight=sw)
        ptr += y_inverse(m.predict(tr[feats])) / k; pte += y_inverse(m.predict(te[feats])) / k
    trm = all_metrics(tr[TARGET].values, ptr)
    tem = all_metrics(te[TARGET].values, pte); tem.update(peak_metrics(te[TARGET].values, pte))
    return {"train_obs": tr[TARGET].values, "train_sim": ptr, "test_obs": te[TARGET].values, "test_sim": pte,
            "train_m": trm, "test_m": tem, "n_tr": len(tr), "n_te": len(te),
            "tr_start": tr["Date"].min(), "te_start": te["Date"].min(), "te_end": te["Date"].max()}

@st.cache_resource(show_spinner=False)
def train_test_split_eval(_df_feat, split_str, peak_focus, cache_key):
    sts = pd.Timestamp(split_str)
    return {"A": chrono_split_daily(_df_feat, feats_ar(_df_feat), sts, peak_focus),
            "B": chrono_split_daily(_df_feat, feats_weather(_df_feat), sts, peak_focus)}

def ensemble_importance(model_pack):
    feats = model_pack["feats"]; imp = np.zeros(len(feats)); n = 0
    for m in model_pack["trained"].values():
        fi = getattr(m, "feature_importances_", None)
        if fi is not None and len(fi) == len(feats): imp += np.asarray(fi); n += 1
    return (imp / n if n else imp), feats

def plot_feature_importance(model_pack, ax, title):
    imp, feats = ensemble_importance(model_pack)
    order = np.argsort(imp); names = np.array(feats)[order]; vals = imp[order]
    snow = {"snowpack", "melt", "melt_lag1", "melt_roll7", "SCF", "SCF_lag1", "SCF_lag7"}
    qset = {"Q_lag1", "Q_lag2", "Q_lag3", "Q_lag7"}
    colors = ["#d62728" if n in qset else "#1d9e75" if n in snow else "#888780" for n in names]
    ax.barh(names, vals, color=colors, alpha=0.9); ax.set_title(title, weight="bold", fontsize=11)
    ax.grid(True, axis="x", ls=":", alpha=0.5)

# =====================================================================
# GÜNCEL MODIS KAR HARİTASI (GIBS WMS) — Karakurt tarzı statik figür
# =====================================================================
@st.cache_data(show_spinner=False, ttl=86400)
def fetch_modis_snow_image(bbox, date_str, _pad=0.05):
    """NASA GIBS WMS'ten verilen tarih için MODIS Terra NDSI kar örtüsü PNG'si (anahtarsız)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    min_lon -= _pad; min_lat -= _pad; max_lon += _pad; max_lat += _pad
    params = {"SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0",
              "LAYERS": "MODIS_Terra_NDSI_Snow_Cover", "FORMAT": "image/png", "TRANSPARENT": "TRUE",
              "CRS": "EPSG:4326", "TIME": date_str,
              "BBOX": f"{min_lat},{min_lon},{max_lat},{max_lon}", "WIDTH": "768", "HEIGHT": "768"}
    try:
        r = requests.get("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi", params=params, timeout=30)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
            return r.content
    except Exception:
        pass
    return None

def render_basin_modis_map(poly, bands, today, streams=None):
    """Güncel MODIS kar örtüsü + havza sınırı + model noktaları (Karakurt tarzı)."""
    bb = poly["bbox"]; pad = 0.05
    extent = [bb[0]-pad, bb[2]+pad, bb[1]-pad, bb[3]+pad]
    verts = poly["verts"]; poly_lon = [v[1] for v in verts]; poly_lat = [v[0] for v in verts]
    modis_date = today.strftime("%Y-%m-%d"); img = fetch_modis_snow_image(bb, modis_date)
    if img is None:
        for back in range(1, 9):
            alt = (today - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
            img = fetch_modis_snow_image(bb, alt)
            if img is not None: modis_date = alt; break
    is_today = (modis_date == today.strftime("%Y-%m-%d"))
    lbl = "bugün" if is_today else "en güncel mevcut kare"
    st.subheader(f"🛰️ Güncel Kar Örtüsü + Havza + Noktalar — {modis_date} ({lbl})")
    fig, ax = plt.subplots(figsize=(8, 8))
    if img is not None:
        import matplotlib.image as mpimg
        ax.imshow(mpimg.imread(io.BytesIO(img)), extent=extent, origin="upper", aspect="auto")
    else:
        st.caption("ℹ️ Son 8 gün için MODIS karesi alınamadı (bulut/gecikme/ağ) — yalnızca havza+noktalar gösteriliyor.")
    if streams:
        for si, s in enumerate(streams):
            ax.plot([c[1] for c in s["coords"]], [c[0] for c in s["coords"]], color="#2eb6ff",
                    lw=1.0, alpha=0.9, zorder=4, label="Dereler" if si == 0 else None)
    ax.plot(poly_lon, poly_lat, color="#e24b4a", lw=2.5, label="Havza sınırı", zorder=5)
    ax.scatter([b["lon"] for b in bands], [b["lat"] for b in bands], c="#ffd400", edgecolors="black",
               s=80, marker="o", label=f"{len(bands)} model noktası", zorder=6)
    for b in bands:
        ax.annotate(b["name"], (b["lon"], b["lat"]), fontsize=6, color="white",
                    xytext=(3, 3), textcoords="offset points", zorder=7)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Boylam (°D)"); ax.set_ylabel("Enlem (°K)")
    ax.legend(loc="upper right", fontsize=8); ax.grid(True, ls=":", alpha=0.3)
    ax.set_title(f"MODIS Terra NDSI Kar Örtüsü — {modis_date}", fontsize=11, weight="bold")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("Açık/beyaz: yüksek kar örtüsü (NDSI). Kırmızı çizgi: havza sınırı. Sarı: model noktaları. Kaynak: NASA GIBS.")

# =====================================================================
# İNTERAKTİF YAĞIŞ ANİMASYONU (IDW yüzeyi) + son 1-2-3 gün MODIS (Karakurt tarzı)
# =====================================================================
def idw_surface(pts_lon, pts_lat, vals, bbox, res=80, power=3.0, pad=0.05):
    min_lon, min_lat, max_lon, max_lat = bbox
    min_lon -= pad; min_lat -= pad; max_lon += pad; max_lat += pad
    gx = np.linspace(min_lon, max_lon, res); gy = np.linspace(min_lat, max_lat, res)
    GX, GY = np.meshgrid(gx, gy)
    pts_lon = np.asarray(pts_lon); pts_lat = np.asarray(pts_lat); vals = np.asarray(vals, float)
    num = den = None
    for k in range(len(vals)):
        d = np.sqrt((GX - pts_lon[k])**2 + (GY - pts_lat[k])**2); d = np.where(d < 1e-9, 1e-9, d)
        if k == 0: num = vals[k] / d**power; den = 1.0 / d**power
        else: num += vals[k] / d**power; den += 1.0 / d**power
    return GX, GY, num / den, (min_lon, min_lat, max_lon, max_lat)

def surface_to_png(GX, GY, Z, verts, vmax=None, cmap_name="YlGnBu", gamma=0.6, n_levels=7):
    from matplotlib.path import Path as MPath
    from PIL import Image
    poly_path = MPath([(v[1], v[0]) for v in verts])
    mask = poly_path.contains_points(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    Zm = np.where(mask, Z, np.nan)
    vmax = max(vmax or (np.nanmax(Zm) if np.isfinite(np.nanmax(Zm)) else 1.0), 1.0)
    norm = np.clip(Zm / vmax, 0, 1) ** gamma
    levels = np.linspace(0, 1, n_levels + 1); norm_q = np.full_like(norm, np.nan); valid = ~np.isnan(norm)
    norm_q[valid] = levels[np.clip(np.digitize(norm[valid], levels) - 1, 0, n_levels - 1)] + (0.5 / n_levels)
    rgba = matplotlib.colormaps[cmap_name](norm_q)
    rgba[..., 3] = np.where(np.isnan(Zm), 0.0, 0.55 + 0.40 * np.clip(norm_q, 0, 1))
    rgba = np.flipud(rgba)
    buf = io.BytesIO(); Image.fromarray((np.nan_to_num(rgba) * 255).astype(np.uint8), "RGBA").save(buf, format="PNG")
    return buf.getvalue()

@st.cache_data(show_spinner="Yağış yüzeyleri hazırlanıyor (bir kez)...", ttl=3600)
def build_rain_surfaces(pts_lon, pts_lat, daily_vals, bbox, verts_t, vmax, res=60, power=3.0):
    verts = [tuple(v) for v in verts_t]; frames = []; bounds = None
    for vals in daily_vals:
        GX, GY, Z, ext = idw_surface(list(pts_lon), list(pts_lat), list(vals), tuple(bbox), res=res, power=power)
        frames.append(base64.b64encode(surface_to_png(GX, GY, Z, verts, vmax=vmax)).decode())
        bounds = [[ext[1], ext[0]], [ext[3], ext[2]]]
    return frames, bounds

@st.fragment
def render_rain_animation_map(poly, bands, df_live, today, streams=None):
    """İnteraktif folium: yağış yüzeyi (gün gün animasyon) + havza + noktalar + son 1-2-3 gün MODIS kar.
    @st.fragment: gün kaydırıcısı/autoplay yalnızca bu haritayı yeniler — tüm uygulamayı DEĞİL (hız)."""
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.info("İnteraktif harita için: `pip install folium streamlit-folium`"); return
    bb = poly["bbox"]; verts = poly["verts"]; NBp = len(bands)
    fc16 = df_live[df_live["Date"] >= today].sort_values("Date").head(16).reset_index(drop=True)
    if fc16.empty:
        st.info("Tahmin verisi yok (harita çizilemedi)."); return
    st.subheader("🌧️ İnteraktif Yağış Yüzeyi Animasyonu + MODIS Kar (son 1-2-3 gün)")
    day_labels = [pd.Timestamp(d).strftime("%d %b %Y") for d in fc16["Date"]]
    pts_lon = [b["lon"] for b in bands]; pts_lat = [b["lat"] for b in bands]
    all_p = [float(fc16.iloc[r].get(f"P_{i}", 0.0)) for r in range(len(fc16)) for i in range(NBp)]
    vmax_all = max(max(all_p) if all_p else 1.0, 1.0)
    if "rain_idx" not in st.session_state: st.session_state["rain_idx"] = 0
    cur_idx = min(st.session_state["rain_idx"], len(day_labels) - 1)
    sel_label = st.select_slider("📅 Yağış yüzeyi günü (gün gün)", options=day_labels, value=day_labels[cur_idx])
    sel_idx = day_labels.index(sel_label); st.session_state["rain_idx"] = sel_idx
    center_lat = (bb[1] + bb[3]) / 2; center_lon = (bb[0] + bb[2]) / 2
    span = max(bb[3] - bb[1], (bb[2] - bb[0]) * 0.7)
    zoom_lvl = 13 if span < 0.07 else 12 if span < 0.12 else 11 if span < 0.22 else 10 if span < 0.65 else 9
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_lvl, tiles=None)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery", name="Uydu", control=False).add_to(fmap)
    try:
        daily_vals = tuple(tuple(float(fc16.iloc[r].get(f"P_{i}", 0.0)) for i in range(NBp)) for r in range(len(fc16)))
        verts_t = tuple(tuple(v) for v in verts)
        frames_b64, img_bounds = build_rain_surfaces(tuple(pts_lon), tuple(pts_lat), daily_vals, tuple(bb), verts_t, round(vmax_all, 3))
        folium.raster_layers.ImageOverlay(image="data:image/png;base64," + frames_b64[sel_idx],
            bounds=img_bounds, opacity=0.75, name=f"Yağış Yüzeyi ({sel_label})", interactive=False, zindex=1).add_to(fmap)
    except Exception as _e:
        st.caption(f"Yağış yüzeyi üretilemedi: {_e}")
    if streams:
        fg_str = folium.FeatureGroup(name=f"Dereler ({len(streams)})", show=True)
        for s in streams:
            folium.PolyLine(s["coords"], color="#2eb6ff", weight=2, opacity=0.85, popup=s["name"]).add_to(fg_str)
        fg_str.add_to(fmap)
    fg_basin = folium.FeatureGroup(name="Havza Sınırı", show=True)
    folium.Polygon([(v[0], v[1]) for v in verts], color="#e24b4a", weight=3, fill=True, fill_opacity=0.0).add_to(fg_basin)
    fg_basin.add_to(fmap)
    fg_pts = folium.FeatureGroup(name=f"{NBp} Model Noktası", show=True)
    for i, b in enumerate(bands):
        pv = float(fc16.iloc[sel_idx].get(f"P_{i}", 0.0))
        folium.CircleMarker([b["lat"], b["lon"]], radius=5, color="black", fill=True, fill_color="#ffd400",
                            fill_opacity=0.9, popup=f"{b['name']}<br>{sel_label}: {pv:.1f} mm").add_to(fg_pts)
    fg_pts.add_to(fmap)
    def add_modis_layer(day_offset, label):
        dt = (today - pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
        url = ("https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_NDSI_Snow_Cover/default/"
               + dt + "/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png")
        folium.TileLayer(tiles=url, attr="NASA GIBS / MODIS", name=f"MODIS Kar — {label} ({dt})",
                         overlay=True, control=True, show=(day_offset == 1), opacity=0.85,
                         max_native_zoom=8, max_zoom=18).add_to(fmap)
    add_modis_layer(1, "1 gün önce"); add_modis_layer(2, "2 gün önce"); add_modis_layer(3, "3 gün önce")
    folium.LayerControl(collapsed=False).add_to(fmap)
    cpa, cpb = st.columns([1, 4])
    with cpa:
        autoplay = st.checkbox("▶ Otomatik oynat", value=False, key="rain_autoplay",
                               help="Gün gün ilerler (~1 sn/gün). Durdurmak için tiki kaldırın.")
    with cpb:
        st.markdown("**🌍 İnteraktif Harita** — kaydırıcıyla günü değiştir ya da oynat; sağ üstten **MODIS kar (1/2/3 gün önce)** katmanlarını aç/kapat.")
    st_folium(fmap, height=540, use_container_width=True, returned_objects=[],
              center=[center_lat, center_lon], zoom=zoom_lvl, key="rain_map")
    st.caption(f"⚠️ Yağış yüzeyi {NBp} noktadan IDW ile üretildi (tahmin; gerçek ölçüm değil). Renk skalası 16 günün "
               f"maksimumuna ({vmax_all:.0f} mm) sabit. MODIS kar katmanları gecikme/bulut nedeniyle bugünkü kare genelde boş — "
               "son 1-3 gün sunulur.")
    if autoplay:
        import time as _t; _t.sleep(1.0)
        st.session_state["rain_idx"] = (sel_idx + 1) % len(day_labels)
        st.rerun(scope="fragment")  # yalnızca fragment'i yeniden çiz (tüm app değil)

# =====================================================================
# ARAYÜZ
# =====================================================================
st.title("🏔️ Kemerdere Havzası — Günlük Akım Tahmin Modeli")
st.caption("Hibrit ML (otoregresif + hava/kar-temelli) • Günlük P/T/PET (Open-Meteo) + MODIS günlük SCF (GEE)")

if st.sidebar.button("Çıkış", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.pop("username", None)
    st.rerun()
st.sidebar.divider()
st.sidebar.header("📁 Girdiler")
flow_up = st.sidebar.file_uploader("Günlük akım (.xlsx) — boşsa Su_Zaman_Serisi_KMR.xlsx", type=["xlsx"])
kmz_up = st.sidebar.file_uploader("Havza poligonu (.kmz/.kml) — boşsa 3_HEPPs.kmz", type=["kmz", "kml"])
river_up = st.sidebar.file_uploader("Dereler/Akarsular (.kmz/.kml) — opsiyonel", type=["kmz", "kml"])
gee_project = st.sidebar.text_input("🌍 GEE Proje Kimliği", value=os.environ.get("EARTHENGINE_PROJECT", ""),
    help="Boşsa kayıtlı proje kullanılır (earthengine set_project). Boş/başarısızsa SCF olmadan devam edilir.")

# ---- Akım dosyasını bytes olarak çöz + GENERİK format seçimi ----
flow_bytes = None
if flow_up is not None:
    flow_bytes = flow_up.getvalue()
elif os.path.exists(DEFAULT_FLOW):
    with open(DEFAULT_FLOW, "rb") as _f: flow_bytes = _f.read()

flow_sheet = flow_date_col = flow_q_col = None
flow_unit_factor = 1.0 / 86400.0; flow_start = "2016-01-01"
if flow_bytes is not None:
    with st.sidebar.expander("📑 Akım dosyası formatı (generic)", expanded=False):
        try:
            _sheets = pd.ExcelFile(io.BytesIO(flow_bytes)).sheet_names
            _si = _sheets.index("Zaman Serisi") if "Zaman Serisi" in _sheets else 0
            flow_sheet = st.selectbox("Sayfa", _sheets, index=_si)
            _cols = [str(c).strip() for c in pd.read_excel(io.BytesIO(flow_bytes), sheet_name=flow_sheet, nrows=5).columns]
            _di = next((i for i, c in enumerate(_cols) if any(k in c.lower() for k in ("tarih", "date", "zaman"))), 0)
            flow_date_col = st.selectbox("Tarih sütunu", _cols, index=_di)
            _qi = next((i for i, c in enumerate(_cols) if "toplam su" in c.lower()), len(_cols) - 1)
            flow_q_col = st.selectbox("Akım sütunu", _cols, index=_qi)
            flow_unit_factor = UNIT_TO_M3S[st.selectbox("Akım birimi", list(UNIT_TO_M3S.keys()), index=0)]
            flow_start = st.text_input("Başlangıç tarihi (YYYY-MM-DD, boş=tümü)", value="2016-01-01").strip() or None
        except Exception as _e:
            st.error(f"Akım dosyası okunamadı: {_e}")

st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Havza / Nokta")
pt_mode = st.sidebar.radio("Nokta şeması",
    ["Tek nokta — havza ortalaması ⭐", "Yükseklik bantları (kar-baskın)", "Grid / Thiessen"], index=0,
    help="Bu havzada TEK nokta (havza merkezi), çok-noktalı/bant şemalarından belirgin daha iyi sonuç verdi "
         "(kör test ~0.80 vs ~0.60): yüksek soğuk noktalar fazla kar/erime üretip yaz resesyonunu abartıyor.")
snow_dom = pt_mode.startswith("Yükseklik")
single_point = pt_mode.startswith("Tek nokta")
n_bands, n_points = 3, 16
if snow_dom:
    n_bands = st.sidebar.slider("Yükseklik bandı sayısı", 2, 6, 3, 1)
elif pt_mode.startswith("Grid"):
    n_points = st.sidebar.slider("Grid nokta sayısı", 4, 30, 12, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("❄️ Kar / Kalibrasyon")
auto_cal = st.sidebar.checkbox("ddf/Tsnow'u otomatik kalibre et", value=True,
    help="Kar parametrelerini grid-aramayla, hava-temelli modelin doğrulama NSE'sini "
         "maksimize edecek şekilde seçer. Kapatırsan elle ayarlarsın.")
if auto_cal:
    manual_snow = None
else:
    _ddf = st.sidebar.slider("ddf (derece-gün, mm/°C/gün)", 1.0, 12.0, 4.0, 0.5)
    _tsnow = st.sidebar.slider("Tsnow (°C)", -2.0, 3.0, 1.0, 0.5)
    manual_snow = {"ddf": _ddf, "Tsnow": _tsnow, "Tmelt": 0.0}
peak_focus = st.sidebar.checkbox("Pik akışlara odaklan", value=True)
split_year_tt = st.sidebar.selectbox("Train/Test bölme yılı (test bu yıldan itibaren KÖR)", [2018, 2019, 2020, 2021], index=2)
Q0 = st.sidebar.number_input("Bugünkü Q₀ (m³/s) — canlı tahmin için", min_value=0.0, value=1.5, step=0.1)

# ---- Girdileri çöz (yükleme ya da varsayılan disk dosyaları) ----
kmz_src = kmz_up if kmz_up is not None else (DEFAULT_KMZ if os.path.exists(DEFAULT_KMZ) else None)

if flow_bytes is None or kmz_src is None:
    st.info("Başlamak için günlük akım Excel'i ve havza KMZ'sini yükleyin "
            "(ya da bu klasöre `Su_Zaman_Serisi_KMR.xlsx` ve `3_HEPPs.kmz` koyun).")
    st.stop()

poly = parse_kmz_polygon(kmz_src)
if not poly:
    st.error("Havza poligonu okunamadı."); st.stop()
basin_area = polygon_area_km2(poly["verts"]) or 0.0
BHASH = basin_hash(poly["verts"])
river_src = river_up if river_up is not None else DEFAULT_RIVER
STREAMS = parse_kmz_lines(river_src) if river_src else None
if STREAMS:
    st.sidebar.caption(f"🏞️ {len(STREAMS)} dere/akarsu yüklendi (haritalarda gösterilir).")

with st.spinner("Havza noktaları üretiliyor..."):
    if single_point:
        _clat = float(np.mean([v[0] for v in poly["verts"]])); _clon = float(np.mean([v[1] for v in poly["verts"]]))
        BANDS = [{"name": "Havza merkezi", "lat": _clat, "lon": _clon, "area_fraction": 1.0}]
    else:
        BANDS = auto_generate_points(poly, snow_dom, n=n_points, n_bands=n_bands)
WEIGHTS = [b["area_fraction"] for b in BANDS]
NB = len(BANDS)
COORDS_STR = json.dumps([(b["lat"], b["lon"]) for b in BANDS])
CHASH = hashlib.md5(COORDS_STR.encode()).hexdigest()[:10]  # koordinat-bazlı (hava cache; kar paramından bağımsız)

df_flow = load_daily_flow(flow_bytes, flow_sheet, flow_date_col, flow_q_col, flow_unit_factor, flow_start)
if df_flow.empty: st.error("Akım verisi okunamadı (format seçimlerini kontrol edin)."); st.stop()
flow_start, flow_end = df_flow["Date"].min(), df_flow["Date"].max()
wx_start = (flow_start - pd.Timedelta(days=120)).strftime("%Y-%m-%d")   # snowpack/lag ısınması

c0, c1, c2, c3 = st.columns(4)
c0.metric("Havza alanı", f"{basin_area:.1f} km²")
c1.metric("Model noktası", f"{NB} bant")
c2.metric("Akım aralığı", f"{flow_start:%Y-%m} – {flow_end:%Y-%m}")
c3.metric("Gözlem günü", f"{len(df_flow):,}")
st.sidebar.download_button("📥 Üretilen noktalar (KMZ)", points_to_kmz_bytes(BANDS, basin_area),
                           f"kemerdere_points_{BHASH}.kmz", "application/vnd.google-earth.kmz")

# ---- Günlük hava (tarihsel) ----
with st.spinner(f"Günlük hava çekiliyor ({NB} nokta)..."):
    df_w = fetch_daily_archive(COORDS_STR, BANDS, wx_start, flow_end.strftime("%Y-%m-%d"),
                               cache_file=os.path.join(SCRIPT_DIR, f"weather_daily_{CHASH}.csv"))

# ---- Günlük SCF (GEE) ----
df_scf = pd.DataFrame()
gee_ok, gee_msg = init_gee(gee_project.strip() or None)
if gee_ok:
    try:
        df_scf = build_modis_scf_daily_gee(poly["verts"], wx_start, flow_end.strftime("%Y-%m-%d"))
    except Exception as e:
        st.warning(f"GEE SCF çekilemedi: {e}. SCF olmadan devam ediliyor.")
else:
    st.warning(f"GEE başlatılamadı ({gee_msg}). SCF olmadan devam ediliyor — sol menüden GEE proje kimliği girin.")
scf_clim = scf_doy_climatology(df_scf)

# ---- Tarihsel özellik tablosu ----
df_raw = df_w.copy()
if not df_scf.empty:
    df_raw = pd.merge(df_raw, df_scf, on="Date", how="left")
df_raw = pd.merge(df_raw, df_flow, on="Date", how="left")

# ---- Kar parametreleri: GERÇEK kalibrasyon (grid) ya da elle ----
if auto_cal:
    cal_key = hashlib.md5((str(df_raw.shape) + str(flow_end) + CHASH + str(bool(not df_scf.empty))).encode()).hexdigest()
    with st.spinner("Kar parametreleri kalibre ediliyor (ddf, Tsnow grid-araması)..."):
        snow_params, snow_cal_nse = calibrate_snow_daily(df_raw, WEIGHTS, scf_clim, cal_key)
    st.sidebar.success(f"❄️ Kalibre: ddf={snow_params['ddf']:.0f}, Tsnow={snow_params['Tsnow']:.0f}°C, "
                       f"Tmelt={snow_params['Tmelt']:.0f}°C (doğrulama NSE={snow_cal_nse:.2f})")
else:
    snow_params, snow_cal_nse = manual_snow, None

df_feat = engineer_daily_features(df_raw, snow_params, WEIGHTS, scf_clim)
df_hist = df_feat.dropna(subset=feats_weather(df_feat) + [TARGET]).reset_index(drop=True)
if len(df_hist) < 200:
    st.error(f"Eğitim verisi çok kısa ({len(df_hist)} gün)."); st.stop()

ck = hashlib.md5((str(df_hist.shape) + str(flow_end) + str(snow_params) + str(peak_focus) + CHASH +
                  str(bool(not df_scf.empty))).encode()).hexdigest()
with st.spinner("Hibrit model eğitiliyor (A: otoregresif, B: hava-temelli)..."):
    A, B = train_hybrid(df_feat, peak_focus, ck)

# ---- Canlı sürekli seri (bağlam + 16 günlük tahmin) ----
today = pd.Timestamp(datetime.now().date())
wy0 = pd.Timestamp(today.year if today.month >= 10 else today.year - 1, 10, 1)
df_ctx = fetch_daily_archive(COORDS_STR, BANDS, wy0.strftime("%Y-%m-%d"),
                             (today - pd.Timedelta(days=11)).strftime("%Y-%m-%d"))
df_fcw = fetch_daily_forecast(COORDS_STR, BANDS)
df_live = pd.concat([df_ctx, df_fcw]).drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)
df_live["SCF_basin"] = df_live["Date"].dt.dayofyear.map(scf_clim) if scf_clim else np.nan
# Tahmin SCF'sini HARİTADAKİ GÜNCEL kara sabitle: en güncel gözlenen MODIS SCF'sinden
# başlat, ~10 günde gün-of-yıl iklimatolojisine harmanla (yoksa iklimatoloji kullanılır).
cur_scf, cur_scf_date = (None, None)
_cur = fetch_current_scf(poly["verts"], today.strftime("%Y-%m-%d")) if gee_ok else None
if _cur is not None:
    cur_scf, cur_scf_date = _cur
    fmask = df_live["Date"] >= today
    nfd = int(fmask.sum())
    if nfd:
        clim_f = (df_live.loc[fmask, "Date"].dt.dayofyear.map(scf_clim).astype(float).fillna(cur_scf).values
                  if scf_clim else np.full(nfd, cur_scf))
        w = np.clip(np.arange(nfd) / 10.0, 0.0, 1.0)  # gün0=güncel gözlem -> ~gün10 iklimatoloji
        df_live.loc[fmask, "SCF_basin"] = (1.0 - w) * cur_scf + w * clim_f
feat_live = engineer_daily_features(df_live, snow_params, WEIGHTS, scf_clim)
fc = hybrid_forecast(feat_live, A, B, today, Q0, n_fc=16)

tab1, tab2, tab3 = st.tabs(["📈 Canlı 16-Günlük Tahmin", "🔬 Tarihsel Doğrulama", "📋 Tablo / İndir"])

with tab1:
    render_basin_modis_map(poly, BANDS, today, STREAMS)
    st.markdown("---")
    render_rain_animation_map(poly, BANDS, df_live, today, STREAMS)
    st.markdown("---")
    st.subheader("16 Günlük Hibrit Hidrograf")
    st.caption("Gün 1-3: Model A (otoregresif, Q₀'dan özyinelemeli). Gün 4-16: Model B (hava/kar-temelli). "
               "Tahmin SCF'si gün-of-yıl iklimatolojisinden (gözlem değil).")
    if fc.empty:
        st.warning("Tahmin üretilemedi (Open-Meteo forecast boş).")
    else:
        omn, ome, omx = q_doy_envelope(df_flow, fc["Date"], window=7)
        fig, axl = plt.subplots(figsize=(15, 4.8))
        axl.fill_between(fc["Date"], omn, omx, color="#bbbbbb", alpha=0.30,
                         label="Gözlenen aralık (tarihsel min–max, ±7 gün)", zorder=1)
        axl.plot(fc["Date"], ome, color="#555555", ls="-.", lw=1.6, label="Tarihsel ortalama", zorder=2)
        axl.plot(fc["Date"], fc["Q_B"], color="#9bbfe0", ls="--", lw=2, marker="s", ms=4, label="Model B (hava-temelli)", zorder=4)
        axl.plot(fc["Date"][:7], fc["Q_A"][:7], color="#1f77b4", lw=1.6, marker="o", ms=4, alpha=0.7, label="Model A (otoregresif)", zorder=4)
        axl.plot(fc["Date"], fc["Q_hybrid"], color="#d62728", lw=2.8, marker="o", ms=5, label="HİBRİT (operasyonel)", zorder=5)
        axl.axhline(Q0, color="gray", ls=":", label=f"Q₀={Q0:.2f}", zorder=3)
        axl.axvspan(fc["Date"].iloc[0], fc["Date"].iloc[min(2, len(fc)-1)], color="#1f77b4", alpha=0.06)
        axl.set_ylabel("Debi (m³/s)"); axl.grid(True, ls=":"); axl.legend(fontsize=8, ncol=2, loc="upper right")
        qmax = max(float(fc[["Q_A", "Q_B", "Q_hybrid"]].max().max()), Q0, float(np.nanmax(omx)) if np.isfinite(np.nanmax(omx)) else 0.0)
        axl.set_ylim(0, qmax * 1.5)
        axp = axl.twinx(); axp.bar(fc["Date"], fc["P_sum"], width=0.7, color="#2c7fb8", alpha=0.5, label="Yağış")
        axp.set_ylim(max(float(fc["P_sum"].max()), 1.0) * 2.8, 0); axp.set_ylabel("Yağış (mm/gün)", color="#2c7fb8")
        axl.xaxis.set_major_formatter(mdates.DateFormatter('%d %b')); plt.setp(axl.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        m = st.columns(4)
        m[0].metric("Toplam hacim (16g)", f"{fc['Q_hybrid'].sum()*86400/1e6:.2f} hm³")
        m[1].metric("Maks debi", f"{fc['Q_hybrid'].max():.2f} m³/s")
        m[2].metric("Ort debi", f"{fc['Q_hybrid'].mean():.2f} m³/s")
        m[3].metric("Min debi", f"{fc['Q_hybrid'].min():.2f} m³/s")

        # --- 16 günlük HAVA: tahmin vs uzun dönem (gün-of-yıl) ort + min/max ---
        fcw = feat_live[feat_live["Date"] >= today].sort_values("Date").head(16)[["Date", "T_mean", "P_sum"]].reset_index(drop=True)
        if not fcw.empty:
            st.markdown("---"); st.subheader("🌤️ 16 Günlük Hava — Tahmin vs Uzun Dönem (gün-of-yıl, ±7g)")
            tmn, tme, tmx = q_doy_envelope(df_feat, fcw["Date"], col="T_mean", window=7)
            pmn, pme, pmx = q_doy_envelope(df_feat, fcw["Date"], col="P_sum", window=7)
            figw, axt = plt.subplots(figsize=(15, 4.2)); axpr = axt.twinx()
            # Sıcaklık (sol eksen)
            axt.fill_between(fcw["Date"], tmn, tmx, color="#f0c0c0", alpha=0.35, label="Tarihsel sıcaklık aralığı (min–max)", zorder=1)
            axt.plot(fcw["Date"], tme, color="#a04040", ls="-.", lw=1.4, label="Tarihsel ort. sıcaklık", zorder=2)
            axt.plot(fcw["Date"], fcw["T_mean"], color="#d62728", marker="o", ms=4, lw=2, label="Tahmin sıcaklık", zorder=4)
            axt.axhline(0, color="black", alpha=0.3, lw=1)
            axt.set_ylabel("Sıcaklık (°C)", color="darkred", weight="bold"); axt.grid(True, ls=":", alpha=0.4)
            # Yağış (sağ eksen)
            axpr.fill_between(fcw["Date"], pmn, pmx, color="#c0d0f0", alpha=0.30, label="Tarihsel yağış aralığı (min–max)", zorder=1)
            axpr.plot(fcw["Date"], pme, color="#4060a0", ls="-.", lw=1.4, label="Tarihsel ort. yağış", zorder=2)
            axpr.bar(fcw["Date"], fcw["P_sum"], width=0.6, color="#1f77b4", alpha=0.55, label="Tahmin yağış", zorder=3)
            axpr.set_ylabel("Yağış (mm/gün)", color="darkblue", weight="bold")
            axt.xaxis.set_major_formatter(mdates.DateFormatter('%d %b')); plt.setp(axt.get_xticklabels(), rotation=30, ha="right")
            ht, lt = axt.get_legend_handles_labels(); hp, lp = axpr.get_legend_handles_labels()
            axt.legend(ht + hp, lt + lp, fontsize=7, ncol=2, loc="upper left")
            figw.tight_layout(); st.pyplot(figw); plt.close(figw)
            st.caption("Havza ortalaması. Kırmızı: sıcaklık, mavi: yağış. Kesik çizgi = uzun dönem (gün-of-yıl) ortalaması, "
                       "bant = tarihsel min–max (±7 gün penceresi, 2015-2021).")

        if "SCF" in fc.columns and fc["SCF"].notna().any():
            st.markdown("---"); st.subheader("🛰️ Tahmin Dönemi Kar Örtüsü (SCF)")
            smn, sme, smx = q_doy_envelope(df_scf, fc["Date"], col="SCF_basin", window=7)
            figfsc, afsc = plt.subplots(figsize=(15, 2.8))
            if np.isfinite(smx).any():
                afsc.fill_between(fc["Date"], smn, smx, color="#bbbbbb", alpha=0.30,
                                  label="Gözlenen SCF aralığı (tarihsel min–max, ±7 gün)", zorder=1)
                afsc.plot(fc["Date"], sme, color="#555555", ls="-.", lw=1.4, label="Tarihsel ort. SCF", zorder=2)
            afsc.fill_between(fc["Date"], 0, fc["SCF"], color="#1f9e9e", alpha=0.18, zorder=3)
            afsc.plot(fc["Date"], fc["SCF"], color="#1f9e9e", marker="h", lw=2, label="Tahmin SCF (gün-of-yıl iklimatolojisi)", zorder=4)
            afsc.set_ylabel("SCF (0-1)"); afsc.set_ylim(0, 1); afsc.grid(True, ls=":"); afsc.legend(fontsize=8, loc="upper right")
            afsc.xaxis.set_major_formatter(mdates.DateFormatter('%d %b')); plt.setp(afsc.get_xticklabels(), rotation=30, ha="right")
            figfsc.tight_layout(); st.pyplot(figfsc); plt.close(figfsc)
            if cur_scf is not None:
                st.caption(f"Başlangıç, **güncel gözlenen MODIS SCF = {cur_scf:.2f}** ({cur_scf_date}) değerine sabitlendi "
                           "(haritayla tutarlı); sonra ~10 günde gün-of-yıl iklimatolojisine harmanlanır.")
            else:
                st.caption("Güncel MODIS SCF çekilemedi → gün-of-yıl iklimatolojisi kullanıldı (haritadaki güncel kardan farklı olabilir).")

with tab2:
    st.subheader("Tarihsel Doğrulama (Günlük)")
    cvtab = []
    for nm, pack in [("A — Otoregresif", A), ("B — Hava-temelli", B)]:
        cv = {}
        # ensemble CV ~ ortalama (model bazında); basitçe modellerin CV'sini birleştir
        for k in ["NSE", "logNSE", "KGE", "PBIAS_%", "Peak_NSE"]:
            vals = [pack["results"][m2]["cv"].get(k, np.nan) for m2 in pack["results"]]
            cvtab.append({"Model": nm, "Metrik": k, "Değer": np.nanmean(vals)})
    cvdf = pd.DataFrame(cvtab).pivot(index="Model", columns="Metrik", values="Değer")
    st.dataframe(cvdf.style.format("{:.3f}"), use_container_width=True)
    st.caption("Su-yılı çapraz doğrulama (her su yılı sırayla test). A doğal olarak yüksek NSE "
               "(dünkü akımı kullanır); B 'gerçek' meteorolojik beceriyi gösterir.")

    # gözlenen vs tahmin (full-fit modellerle); A için Q_lag'leri tam olan satırlar
    dfB_hist = df_hist
    dfA_hist = df_feat.dropna(subset=feats_ar(df_feat) + [TARGET]).reset_index(drop=True)
    predB_h = predict_ensemble(B, dfB_hist)
    predA_h = predict_ensemble(A, dfA_hist) if len(dfA_hist) else None
    fig2, ax2 = plt.subplots(figsize=(15, 4.2))
    ax2.plot(dfB_hist["Date"], dfB_hist[TARGET], color="black", lw=1.4, alpha=0.6, label="Gözlenen")
    if predA_h is not None:
        ax2.plot(dfA_hist["Date"], predA_h, color="#1f77b4", lw=0.9, alpha=0.75, label="Model A")
    ax2.plot(dfB_hist["Date"], predB_h, color="#d62728", lw=0.9, alpha=0.75, label="Model B")
    ax2.set_ylabel("Debi (m³/s)"); ax2.grid(True, ls=":"); ax2.legend(fontsize=8, loc="upper right")
    fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    fig3, (a, b) = plt.subplots(1, 2, figsize=(15, 4.5))
    mx = float(dfB_hist[TARGET].max()) * 1.1
    if predA_h is not None:
        a.scatter(dfA_hist[TARGET], predA_h, s=6, alpha=0.3, color="#1f77b4")
    a.plot([0, mx], [0, mx], "r--"); a.set_title("Model A"); a.set_xlabel("Gözlenen"); a.set_ylabel("Tahmin")
    a.set_xlim(0, mx); a.set_ylim(0, mx); a.grid(True, ls=":")
    b.scatter(dfB_hist[TARGET], predB_h, s=6, alpha=0.3, color="#d62728")
    b.plot([0, mx], [0, mx], "r--"); b.set_title("Model B"); b.set_xlabel("Gözlenen"); b.set_ylabel("Tahmin")
    b.set_xlim(0, mx); b.set_ylim(0, mx); b.grid(True, ls=":")
    fig3.tight_layout(); st.pyplot(fig3); plt.close(fig3)

    if df_scf is not None and not df_scf.empty and df_scf["SCF_basin"].notna().any():
        st.markdown("---"); st.subheader("🛰️ Günlük MODIS Kar Örtüsü Oranı (SCF) — Gözlenen")
        _sc = df_scf.dropna(subset=["SCF_basin"]).sort_values("Date")
        figs, axsc = plt.subplots(figsize=(15, 3.0))
        axsc.fill_between(_sc["Date"], 0, _sc["SCF_basin"], color="#1f9e9e", alpha=0.15)
        axsc.plot(_sc["Date"], _sc["SCF_basin"], color="#1f9e9e", lw=0.8)
        axsc.set_ylabel("SCF (0-1)"); axsc.set_ylim(0, 1)
        # ay çentikleri (minor) + yıl etiketleri (major)
        axsc.xaxis.set_major_locator(mdates.YearLocator())
        axsc.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axsc.xaxis.set_minor_locator(mdates.MonthLocator())  # ay çentikleri (etiketsiz)
        axsc.tick_params(axis="x", which="minor", length=4)
        axsc.tick_params(axis="x", which="major", length=9, labelsize=10)
        axsc.grid(True, which="major", axis="both", ls=":", alpha=0.5)
        axsc.grid(True, which="minor", axis="x", ls=":", alpha=0.25)
        axsc.set_title("Havza ortalama günlük kar örtüsü oranı (MODIS MOD10A1, GEE)", fontsize=10, weight="bold")
        figs.tight_layout(); st.pyplot(figs); plt.close(figs)
        st.caption(f"{len(_sc)} günlük gözlem (bulutlu günler hariç). Tahmin ufkunda gün-of-yıl iklimatolojisi kullanılır.")

    # --- Kronolojik TRAIN / TEST (kör test) ---
    st.markdown("---"); st.subheader(f"🧪 Train / Test — kör test: {split_year_tt} ve sonrası")
    tt = train_test_split_eval(df_feat, f"{split_year_tt}-01-01", peak_focus, ck + str(split_year_tt))
    tt_rows = []
    for nm in ["A", "B"]:
        rr = tt.get(nm)
        if rr:
            tt_rows.append({"Model": ("A — Otoregresif" if nm == "A" else "B — Hava-temelli"),
                            "Train NSE": rr["train_m"]["NSE"], "Test NSE": rr["test_m"]["NSE"],
                            "Test KGE": rr["test_m"]["KGE"], "Test PBIAS_%": rr["test_m"]["PBIAS_%"],
                            "Test Pik_NSE": rr["test_m"].get("Peak_NSE", np.nan),
                            "n_train": rr["n_tr"], "n_test": rr["n_te"]})
    if tt_rows:
        st.dataframe(pd.DataFrame(tt_rows).set_index("Model").style.format({
            "Train NSE": "{:.3f}", "Test NSE": "{:.3f}", "Test KGE": "{:.3f}", "Test PBIAS_%": "{:+.1f}",
            "Test Pik_NSE": "{:.3f}", "n_train": "{:.0f}", "n_test": "{:.0f}"}), use_container_width=True)
        for nm, color in [("A", "#1f77b4"), ("B", "#d62728")]:
            rr = tt.get(nm)
            if not rr: continue
            fl = feats_ar(df_feat) if nm == "A" else feats_weather(df_feat)
            lbl = "Model A (otoregresif)" if nm == "A" else "Model B (hava-temelli)"
            te_df = df_feat[df_feat["Date"] >= pd.Timestamp(f"{split_year_tt}-01-01")].dropna(subset=fl + [TARGET])
            figtt, (t1, t2) = plt.subplots(1, 2, figsize=(15, 4.0))
            t1.plot(te_df["Date"], rr["test_obs"], color="black", lw=1.2, alpha=0.6, label="Gözlenen (test)")
            t1.plot(te_df["Date"], rr["test_sim"], color=color, lw=1.0, alpha=0.85, label=f"{lbl} tahmin")
            t1.set_title(f"Kör test dönemi — {lbl} — NSE {rr['test_m']['NSE']:.2f}")
            t1.set_ylabel("Debi (m³/s)"); t1.legend(fontsize=8); t1.grid(True, ls=":")
            mx = max(float(np.max(rr["train_obs"])), float(np.max(rr["test_obs"]))) * 1.1
            t2.scatter(rr["train_obs"], rr["train_sim"], s=6, alpha=0.22, color="#1d9e75", label=f"Eğitim (NSE {rr['train_m']['NSE']:.2f})")
            t2.scatter(rr["test_obs"], rr["test_sim"], s=9, alpha=0.45, color=color, label=f"Kör test (NSE {rr['test_m']['NSE']:.2f})")
            t2.plot([0, mx], [0, mx], "r--"); t2.set_xlim(0, mx); t2.set_ylim(0, mx)
            t2.set_xlabel("Gözlenen"); t2.set_ylabel("Tahmin"); t2.set_title(f"{lbl} — Eğitim vs Kör Test")
            t2.legend(fontsize=8); t2.grid(True, ls=":")
            figtt.tight_layout(); st.pyplot(figtt); plt.close(figtt)
        st.caption("Su-yılı CV (üstte) + bağımsız kronolojik kör test (burada): test dönemi eğitimde HİÇ kullanılmadı. "
                   "Model A test NSE'si yüksek (Q_lag/dünkü akım baskın); Model B 'gerçek' meteorolojik beceriyi gösterir.")
    else:
        st.info("Train/test için yeterli veri yok — bölme yılını küçültün.")

    # --- Özellik önemi ---
    st.markdown("---"); st.subheader("⭐ Özellik Önemi (RF+GBM ortalaması)")
    figfi, (fa, fb) = plt.subplots(1, 2, figsize=(15, 5.5))
    plot_feature_importance(A, fa, "Model A (otoregresif)")
    plot_feature_importance(B, fb, "Model B (hava-temelli)")
    figfi.tight_layout(); st.pyplot(figfi); plt.close(figfi)
    st.caption("🔴 Q-lag (otoregresif) • 🟢 kar/SCF (snowpack, melt, SCF) • ⚪ yağış/sıcaklık/mevsim.")

with tab3:
    st.subheader("16 Günlük Tahmin Tablosu")
    if not fc.empty:
        show = fc[["Date", "gun", "Q_hybrid", "Q_A", "Q_B", "P_sum", "SCF", "reliability"]].copy()
        st.dataframe(show.style.format({"Q_hybrid": "{:.2f}", "Q_A": "{:.2f}", "Q_B": "{:.2f}",
                                        "P_sum": "{:.1f}", "SCF": "{:.2f}"}), use_container_width=True)
        st.download_button("📥 Tahmin (CSV)", show.to_csv(index=False).encode("utf-8"),
                           "kemerdere_16gun_tahmin.csv", "text/csv")
    st.markdown("---")
    st.subheader("Tarihsel gözlem + model (CSV)")
    dfA_hist2 = df_feat.dropna(subset=feats_ar(df_feat) + [TARGET]).reset_index(drop=True)
    hist_out = dfA_hist2[["Date", TARGET]].copy()
    hist_out["Q_A_pred"] = predict_ensemble(A, dfA_hist2)
    hist_out["Q_B_pred"] = predict_ensemble(B, dfA_hist2)
    st.download_button("📥 Tarihsel (CSV)", hist_out.to_csv(index=False).encode("utf-8"),
                       "kemerdere_tarihsel_model.csv", "text/csv")
