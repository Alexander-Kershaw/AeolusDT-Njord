from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
import folium
from folium.features import DivIcon
from streamlit_folium import st_folium
from datetime import date

from njord.config import REGIONS

st.set_page_config(page_title="AeolusDT-Njord Wind Farm Twins", layout="wide")


# Helpers
def dtstr(d: date) -> str:
    return pd.to_datetime(d).strftime("%Y-%m-%d")

def fmt_ts(ts: float) -> str:
    # ts is epoch seconds
    return pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M:%S")

def mtime_str(p: Path) -> str:
    if not p.exists():
        return "missing"
    return pd.to_datetime(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")

def hours_in_period(start: str, end: str) -> float:
    # inclusive day range, hourly data
    t0 = pd.to_datetime(start)
    t1 = pd.to_datetime(end) + pd.Timedelta(days=1)
    return float((t1 - t0) / pd.Timedelta(hours=1))


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



def region_map(region_sel: str, layout_df: pd.DataFrame, show_turbines: bool) -> folium.Map:
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

start = dtstr(d0)
end = dtstr(d1)

colA, colB = st.columns([1.2, 1.0], gap="large")

with colA:
    st.subheader("Farm regions and turbine layout map")
    layout = load_layout(region, start, end, height)
    m = region_map(region, layout, show_turbs)
    st_folium(m, width=900, height=520)

with colB:
    st.subheader("Energy and capacity factor summary")
    rep = load_energy_cf(start, end, height)
    if rep.empty:
        st.info("No energy report found yet. Run: python scripts/report_energy_cf.py --start ... --end ... --height ...")
    else:
        st.dataframe(rep, use_container_width=True)

st.divider()


# Load time series
base, base_path = load_baseline_farm(region, start, end, height)
wake, wake_path = load_wake(region, start, end, height)
dens, dens_path = load_density(region, start, end, height)


# KPIs
st.subheader("KPIs (for selected region)")

if base.empty:
    st.warning("Baseline power parquet not found for this selection.")
else:
    # Merge series on time
    ts = base.copy()
    if not wake.empty:
        ts = ts.merge(wake[["time", "p_farm_wake_mw", "wake_loss_pct"]], on="time", how="left")
    if not dens.empty:
        ts = ts.merge(dens[["time", "p_farm_rho_mw", "delta_pct"]], on="time", how="left")

    # Region row from report
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
        # fallback KPIs from time series only
        H = hours_in_period(start, end)
        p_cap = 60.0  # 12 x 5 turbines rated at 5MW
        e_base = float(ts["p_farm_base_mw"].sum())
        cf_base = e_base / (p_cap * H)
        c1.metric("Capacity factor (baseline)", f"{cf_base:.3f}")
        if "p_farm_wake_mw" in ts.columns:
            e_w = float(ts["p_farm_wake_mw"].sum())
            c2.metric("Capacity factor (wake)", f"{(e_w / (p_cap * H)):.3f}")
        if "p_farm_rho_mw" in ts.columns:
            e_d = float(ts["p_farm_rho_mw"].sum())
            c3.metric("Capacity factor (density)", f"{(e_d / (p_cap * H)):.3f}")
        c4.metric("Energy (baseline, MWh)", f"{e_base:.0f}")

    # Additional KPI stats
    c5, c6, c7, c8 = st.columns(4)
    if "wake_loss_pct" in ts.columns:
        c5.metric("Mean wake loss %", f"{float(ts['wake_loss_pct'].mean()):.2f}%")
        c6.metric("Max wake loss %", f"{float(ts['wake_loss_pct'].max()):.2f}%")
    else:
        c5.metric("Mean wake loss %", "n/a")
        c6.metric("Max wake loss %", "n/a")

    if "delta_pct" in ts.columns:
        # density uplift relies on farm producing power
        producing = ts["p_farm_base_mw"] > 0
        mean_uplift = float(ts.loc[producing, "delta_pct"].mean()) if producing.any() else 0.0
        c7.metric("Mean density uplift %", f"{mean_uplift:.2f}%")
        c8.metric("Max density uplift %", f"{float(ts['delta_pct'].max()):.2f}%")
    else:
        c7.metric("Mean density uplift %", "n/a")
        c8.metric("Max density uplift %", "n/a")

  
    # Time series with zoom window
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
    st.caption("Baseline = power curve. Wake = Jensen wake deficit. Density = baseline scaled by ρ/1.225 and capped at rated power.")


    # Provenance
    with st.expander("Data provenance"):
        inflow_p = find_latest_inflow_path(region, start, end, height)
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