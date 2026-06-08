# AGENTS.md — Kemerdere Daily Hydrological Model

## Overview

Single-file Streamlit app (`kemerdere_daily_model.py`, 1180 lines) for daily streamflow prediction in the Kemerdere basin using hybrid ML (RF + GBM ensemble).

## Run

```bash
streamlit run kemerdere_daily_model.py
```

## Required Files

- `Su_Zaman_Serisi_KMR.xlsx` — daily flow observations (must have columns with "tarih"/"date" and "toplam su")
- `3_HEPPs.kmz` — basin polygon (KMZ/KML)
- `rivers.kml` — optional river lines for map display

## Dependencies

Core: `numpy`, `pandas`, `requests`, `matplotlib`, `scikit-learn`, `streamlit`
Optional (interactive maps): `folium`, `streamlit-folium`
Optional (snow cover): `earthengine-api` (GEE)

## External APIs (no keys needed)

- **Open-Meteo**: daily weather archive + forecast (P, T, PET)
- **NASA GIBS**: MODIS NDSI snow cover images
- **Google Earth Engine**: MODIS daily SCF (optional, requires project ID)

## Architecture

- No build system, no tests, no lint config — this is a standalone research app
- All logic in one file: geometry helpers, API fetches, ML training, plotting, Streamlit UI
- Caching via `@st.cache_data` and `@st.cache_resource`
- Model A: autoregressive (Q_lag1-7), Model B: weather-only; hybrid uses A for days 1-3, B for days 4-16
