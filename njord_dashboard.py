from __future__ import annotations

from pathlib import Path
from datetime import date
import base64

import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.features import DivIcon
from streamlit_folium import st_folium
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from njord.config import REGIONS

st.set_page_config(page_title="AeolusDT-Njord Wind Farm Twin", layout="wide")


# Helpers
def dtstr(d: date) -> str:
    return pd.to_datetime(d).strftime("%Y-%m-%d")

def mtime_str(p: Path) -> str:
    if not p.exists():
        return "missing"
    return pd.to_datetime(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")

def hours_in_period(start: str, end: str) -> float:
    t0 = pd.to_datetime(start)
    t1 = pd.to_datetime(end) + pd.Timedelta(days=1)
    return float((t1 - t0) / pd.Timedelta(hours=1))

def get_time_name(ds: xr.Dataset) -> str:
    for cand in ("time", "valid_time"):
        if cand in ds.dims or cand in ds.coords:
            return cand
    raise KeyError(f"No time-like coord found. Coords={list(ds.coords)} Dims={list(ds.dims)}")

def b64_png(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


# Cached loaders
@st.cache_data(show_spinner=False)
def load_energy_cf(start: str, end: str, height: int) -> pd.DataFrame:
    p = Path(f"data_lake/gold/reports/energy_cf_{start}_to_{end}_{height}m.csv")
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

@st.cache_data(show_spinner=False)
def find_latest_inflow_path(region: str, start: str, end: str, height: int) -> Path | None:
    inflow_dir = Path("data_lake/gold/inflow") / region
    pat = f"inflow_u{height}v{height}_{start}_to_{end}_*.parquet"
    files = sorted(inflow_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

@st.cache_data(show_spinner=False)
def load_layout(region: str, start: str, end: str, height: int) -> pd.DataFrame:
    p = find_latest_inflow_path(region, start, end, height)
    if p is None:
        return pd.DataFrame()
    df = pd.read_parquet(p)
    layout = df[["turbine_id", "lat", "lon"]].drop_duplicates().sort_values("turbine_id")
    return layout

@st.cache_data(show_spinner=False)
def load_baseline_farm(region: str, start: str, end: str, height: int) -> tuple[pd.DataFrame, Path | None]:
    p = Path(f"data_lake/gold/power/{region}/baseline_power_inflow_u{height}v{height}_{start}_to_{end}_r3c4_sp1.2km.parquet")
    if not p.exists():
        return pd.DataFrame(), None
    df = pd.read_parquet(p)
    df["time"] = pd.to_datetime(df["time"])
    if "p_turbine_mw" in df.columns:
        farm = df.groupby("time")["p_turbine_mw"].sum().reset_index().rename(columns={"p_turbine_mw": "p_farm_base_mw"})
        return farm, p
    if "p_farm_mw" in df.columns:
        farm = df.rename(columns={"p_farm_mw": "p_farm_base_mw"})[["time", "p_farm_base_mw"]]
        return farm, p
    for c in ["p_farm_base_mw", "p_farm_free_mw"]:
        if c in df.columns:
            return df[["time", c]].rename(columns={c: "p_farm_base_mw"}), p
    return pd.DataFrame(), p

@st.cache_data(show_spinner=False)
def load_wake(region: str, start: str, end: str, height: int) -> tuple[pd.DataFrame, Path | None]:
    p = Path(f"data_lake/gold/power/{region}/wake_power_inflow_u{height}v{height}_{start}_to_{end}_r3c4_sp1.2km.parquet")
    if not p.exists():
        return pd.DataFrame(), None
    df = pd.read_parquet(p)
    df["time"] = pd.to_datetime(df["time"])
    return df, p

@st.cache_data(show_spinner=False)
def load_density(region: str, start: str, end: str, height: int) -> tuple[pd.DataFrame, Path | None]:
    p = Path(f"data_lake/gold/power/{region}/density_farm_inflow_u{height}v{height}_{start}_to_{end}_r3c4_sp1.2km.parquet")
    if not p.exists():
        return pd.DataFrame(), None
    df = pd.read_parquet(p)
    df["time"] = pd.to_datetime(df["time"])
    return df, p


# Wind overlay loaders
@st.cache_data(show_spinner=False)
def wind_nc_path(region: str, start: str, end: str, height: int) -> Path:
    return Path(f"data_lake/bronze/era5/{region}/era5_single_levels_u{height}v{height}_{start}_to_{end}.nc")

@st.cache_data(show_spinner=False)
def load_wind_times(region: str, start: str, end: str, height: int) -> list[str]:
    p = wind_nc_path(region, start, end, height)
    if not p.exists():
        return []
    ds = xr.open_dataset(p)
    tname = get_time_name(ds)
    times = pd.to_datetime(ds[tname].values)
    return [t.strftime("%Y-%m-%d %H:%M") for t in times]

@st.cache_data(show_spinner=False)
def legend_png_path(vmin: float, vmax: float) -> Path:
    out = Path("viz/dashboard")
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"wind_legend_{int(vmin)}_{int(vmax)}.png"
    if p.exists():
        return p

    fig = plt.figure(figsize=(3.6, 0.45), dpi=200)
    ax = fig.add_axes([0.05, 0.45, 0.90, 0.35]) 
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), cax=ax, orientation="horizontal")
    cb.set_label("Wind speed (m/s)", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2)
    fig.savefig(p, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return p

@st.cache_data(show_spinner=False)
def build_wind_raster(region: str, start: str, end: str, height: int, time_label: str,
                      vmin: float, vmax: float, arrow_scale: float, arrow_max_deg: float) -> dict:
    p = wind_nc_path(region, start, end, height)
    if not p.exists():
        return {"rgba": None, "bounds": None, "arrows": [], "legend_b64": None}

    ds = xr.open_dataset(p)
    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    if "longitude" in ds.coords:
        ds = ds.sortby("longitude")

    tname = get_time_name(ds)
    times = pd.to_datetime(ds[tname].values)
    wanted = pd.to_datetime(time_label)
    idx = int(np.argmin(np.abs(times - wanted)))

    u_name = f"u{height}"
    v_name = f"v{height}"
    if u_name not in ds.data_vars or v_name not in ds.data_vars:
        return {"rgba": None, "bounds": None, "arrows": [], "legend_b64": None}

    u = ds[u_name].isel({tname: idx}).values.astype(float)
    v = ds[v_name].isel({tname: idx}).values.astype(float)
    speed = np.sqrt(u**2 + v**2)

    lats = ds["latitude"].values
    lons = ds["longitude"].values
    latmin, latmax = float(lats.min()), float(lats.max())
    lonmin, lonmax = float(lons.min()), float(lons.max())
    bounds = [[latmin, lonmin], [latmax, lonmax]]

    # RGBA raster
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap("viridis")
    rgba = (cmap(norm(speed)) * 255).astype(np.uint8)
    rgba[..., 3] = 150 

    arrows = []
    step_i = max(1, len(lats) // 8)
    step_j = max(1, len(lons) // 8)

    # Windspeed vector arrow scaling
    base_deg_per_ms = 0.0025  
    k = base_deg_per_ms * arrow_scale

    for ii in range(0, len(lats), step_i):
        for jj in range(0, len(lons), step_j):
            lat0 = float(lats[ii])
            lon0 = float(lons[jj])
            uij = float(u[ii, jj])
            vij = float(v[ii, jj])

            dlat = vij * k
            dlon = uij * k

            # capped arrow length 
            L = float(np.sqrt(dlat**2 + dlon**2))
            if L > arrow_max_deg and L > 0:
                s = arrow_max_deg / L
                dlat *= s
                dlon *= s

            lat1 = lat0 + dlat
            lon1 = lon0 + dlon
            arrows.append(((lat0, lon0), (lat1, lon1)))

    leg_path = legend_png_path(vmin, vmax)
    legend_b64 = b64_png(leg_path)

    return {"rgba": rgba, "bounds": bounds, "arrows": arrows, "legend_b64": legend_b64}

def add_legend_to_map(m: folium.Map, legend_b64: str) -> None:
    if not legend_b64:
        return
    html = f"""
    <div style="
        position: fixed;
        bottom: 18px;
        left: 18px;
        z-index: 9999;
        background: rgba(255,255,255,0.75);
        padding: 6px 8px;
        border-radius: 8px;
        box-shadow: 0 1px 10px rgba(0,0,0,0.15);
    ">
      <img src="data:image/png;base64,{legend_b64}" style="display:block; width:260px;">
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


# Map builder
def build_map(region_sel: str, layout_df: pd.DataFrame, show_turbines: bool,
              overlay_mode: str, overlay_payload: dict | None) -> folium.Map:

    m = folium.Map(
        location=[63.5, 6.0],
        zoom_start=4,
        tiles="CartoDB positron",
        attr="© OpenStreetMap, © CARTO",
    )

    for key, r in REGIONS.items():
        box = [(r.lat_min, r.lon_min), (r.lat_max, r.lon_min), (r.lat_max, r.lon_max), (r.lat_min, r.lon_max)]
        folium.Polygon(
            locations=box,
            color="#2c7fb8" if key != region_sel else "#d95f0e",
            weight=2,
            fill=True,
            fill_opacity=0.08 if key != region_sel else 0.14,
            tooltip=f"{r.name} ({key})",
        ).add_to(m)

        latc = (r.lat_min + r.lat_max) / 2.0
        lonc = (r.lon_min + r.lon_max) / 2.0
        folium.map.Marker(
            [latc, lonc],
            icon=DivIcon(
                icon_size=(250, 36),
                icon_anchor=(0, 0),
                html=f'<div style="font-size: 13px; font-weight: 700; color: #111;">{r.name}</div>',
            ),
        ).add_to(m)

    if overlay_mode != "Off" and overlay_payload and overlay_payload.get("rgba") is not None:
        folium.raster_layers.ImageOverlay(
            image=overlay_payload["rgba"],
            bounds=overlay_payload["bounds"],
            opacity=0.45,
            interactive=False,
            cross_origin=False,
            zindex=3,
        ).add_to(m)

        add_legend_to_map(m, overlay_payload.get("legend_b64"))

        if overlay_mode == "Speed + arrows":
            for (lat0, lon0), (lat1, lon1) in overlay_payload.get("arrows", []):
                folium.PolyLine(
                    locations=[(lat0, lon0), (lat1, lon1)],
                    weight=3,
                    opacity=0.95,
                    color="#111",
                ).add_to(m)

    if show_turbines and not layout_df.empty:
        for _, row in layout_df.iterrows():
            folium.CircleMarker(
                location=(float(row["lat"]), float(row["lon"])),
                radius=4,
                weight=1,
                fill=True,
                fill_opacity=0.9,
                tooltip=f"Turbine {int(row['turbine_id'])}",
            ).add_to(m)

    return m

# UI
st.title("AeolusDT-Njord Offshore Wind Farm Digital Twin")

with st.sidebar:
    st.header("Controls")
    d0 = st.date_input("Start date", value=date(2020, 1, 1))
    d1 = st.date_input("End date", value=date(2020, 1, 31))
    height = st.selectbox("Wind height (m)", [100, 10], index=0)
    region = st.selectbox("Region", ["tampen_box_a", "utsira_nord_box_b", "sn2_box_c"], index=0)
    show_turbs = st.checkbox("Show turbine points", value=True)

    st.divider()
    st.subheader("Wind overlay")
    overlay_mode = st.selectbox("Overlay mode", ["Off", "Speed raster", "Speed + arrows"], index=1)
    vmin, vmax = 0.0, 30.0
    arrow_scale = st.slider("Arrow scale", min_value=0.5, max_value=6.0, value=2.5, step=0.5)
    arrow_max_deg = st.slider("Arrow max length", min_value=0.01, max_value=0.12, value=0.06, step=0.01)

start = dtstr(d0)
end = dtstr(d1)

overlay_payload = None
if overlay_mode != "Off":
    time_labels = load_wind_times(region, start, end, height)
    if not time_labels:
        st.sidebar.warning(
            "No wind NetCDF found for this selection.\n"
            "Expected:\n"
            f"data_lake/bronze/era5/{region}/era5_single_levels_u{height}v{height}_{start}_to_{end}.nc"
        )
    else:
        # --- Wind replay controls ---
        if "wind_idx" not in st.session_state:
            st.session_state.wind_idx = 0

        colp, coln = st.sidebar.columns(2)
        if colp.button("⏮ Prev"):
            st.session_state.wind_idx = max(0, st.session_state.wind_idx - 1)
        if coln.button("Next ⏭"):
            st.session_state.wind_idx = min(len(time_labels) - 1, st.session_state.wind_idx + 1)

        st.session_state.wind_idx = st.sidebar.slider(
            "Overlay timestep index",
            min_value=0,
            max_value=len(time_labels) - 1,
            value=int(st.session_state.wind_idx),
            step=1,
        )

        t_sel = time_labels[int(st.session_state.wind_idx)]
        st.sidebar.caption(f"Selected: {t_sel}")

        overlay_payload = build_wind_raster(
            region=region, start=start, end=end, height=height, time_label=t_sel,
            vmin=vmin, vmax=vmax, arrow_scale=arrow_scale, arrow_max_deg=arrow_max_deg
        )

colA, colB = st.columns([1.2, 1.0], gap="large")

with colA:
    st.subheader("Regions and turbine layout map")
    layout = load_layout(region, start, end, height)
    m = build_map(region, layout, show_turbs, overlay_mode, overlay_payload)
    st_folium(m, width=900, height=520)

with colB:
    st.subheader("Energy and capacity factor summary")
    rep = load_energy_cf(start, end, height)
    if rep.empty:
        st.info("No energy report found yet. Run: python scripts/report_energy_cf.py --start ... --end ... --height ...")
    else:
        st.dataframe(rep, use_container_width=True)

st.divider()

base, base_path = load_baseline_farm(region, start, end, height)
wake, wake_path = load_wake(region, start, end, height)
dens, dens_path = load_density(region, start, end, height)

st.subheader("KPIs (for selected region)")

if base.empty:
    st.warning("Baseline power parquet not found for this selection.")
else:
    ts = base.copy()
    if not wake.empty:
        ts = ts.merge(wake[["time", "p_farm_wake_mw", "wake_loss_pct"]], on="time", how="left")
    if not dens.empty:
        ts = ts.merge(dens[["time", "p_farm_rho_mw", "delta_pct"]], on="time", how="left")

    region_row = None
    if not rep.empty and "region" in rep.columns:
        rr = rep[rep["region"] == region]
        if not rr.empty:
            region_row = rr.iloc[0].to_dict()

    c1, c2, c3, c4 = st.columns(4)
    if region_row:
        c1.metric("Capacity factor (baseline)", f"{region_row['cf_baseline']:.3f}")
        c2.metric("Capacity factor (wake)", f"{region_row['cf_wake']:.3f}")
        c3.metric("Capacity factor (density)", f"{region_row['cf_density']:.3f}")
        c4.metric("Energy (baseline, MWh)", f"{region_row['energy_baseline_mwh']:.0f}")
    else:
        H = hours_in_period(start, end)
        p_cap = 60.0
        e_base = float(ts["p_farm_base_mw"].sum())
        cf_base = e_base / (p_cap * H)
        c1.metric("Capacity factor (baseline)", f"{cf_base:.3f}")
        c4.metric("Energy (baseline, MWh)", f"{e_base:.0f}")

    c5, c6, c7, c8 = st.columns(4)
    if "wake_loss_pct" in ts.columns:
        c5.metric("Mean wake loss %", f"{float(ts['wake_loss_pct'].mean()):.2f}%")
        c6.metric("Max wake loss %", f"{float(ts['wake_loss_pct'].max()):.2f}%")
    else:
        c5.metric("Mean wake loss %", "n/a")
        c6.metric("Max wake loss %", "n/a")

    if "delta_pct" in ts.columns:
        producing = ts["p_farm_base_mw"] > 0
        mean_uplift = float(ts.loc[producing, "delta_pct"].mean()) if producing.any() else 0.0
        c7.metric("Mean density uplift %", f"{mean_uplift:.2f}%")
        c8.metric("Max density uplift %", f"{float(ts['delta_pct'].max()):.2f}%")
    else:
        c7.metric("Mean density uplift %", "n/a")
        c8.metric("Max density uplift %", "n/a")

    st.subheader("Time series (farm power output)")
    tmin = ts["time"].min()
    tmax = ts["time"].max()
    sel = st.slider(
        "Zoom window",
        min_value=tmin.to_pydatetime(),
        max_value=tmax.to_pydatetime(),
        value=(tmin.to_pydatetime(), tmax.to_pydatetime()),
    )
    t0, t1 = pd.to_datetime(sel[0]), pd.to_datetime(sel[1])
    z = ts[(ts["time"] >= t0) & (ts["time"] <= t1)].set_index("time")

    cols = ["p_farm_base_mw"]
    for c in ["p_farm_wake_mw", "p_farm_rho_mw"]:
        if c in z.columns:
            cols.append(c)

    st.line_chart(z[cols], height=340, use_container_width=True)

    with st.expander("Data provenance (files used)"):
        inflow_p = find_latest_inflow_path(region, start, end, height)
        wf = wind_nc_path(region, start, end, height)
        st.write("Wind file:", str(wf))
        st.write("Wind modified:", mtime_str(wf))
        st.write("Inflow file:", str(inflow_p) if inflow_p else "missing")
        if inflow_p:
            st.write("Inflow modified:", mtime_str(inflow_p))
        st.write("Baseline file:", str(base_path) if base_path else "missing")
        if base_path:
            st.write("Baseline modified:", mtime_str(base_path))
        st.write("Wake file:", str(wake_path) if wake_path else "missing")
        if wake_path:
            st.write("Wake modified:", mtime_str(wake_path))
        st.write("Density file:", str(dens_path) if dens_path else "missing")
        if dens_path:
            st.write("Density modified:", mtime_str(dens_path))

