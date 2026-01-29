from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from njord.config import REGIONS, DEFAULT_START, DEFAULT_END

def latest_inflow_parquet(region_key: str) -> Path:
    d = Path("data_lake/gold/inflow") / region_key
    files = sorted(d.glob("inflow_u10v10_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No inflow parquet files found in {d}")
    return files[0]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot turbine layout and inflow time series.")
    p.add_argument("--region", required=True, choices=sorted(REGIONS.keys()))
    return p.parse_args()

def main() -> int:
    args = parse_args()
    region_key = args.region
    r = REGIONS[region_key]

    inflow_path = latest_inflow_parquet(region_key)
    print("Using inflow parquet:", inflow_path)

    df = pd.read_parquet(inflow_path)
    df["time"] = pd.to_datetime(df["time"])

    layout = df[["turbine_id", "lat", "lon"]].drop_duplicates().sort_values("turbine_id")

    # Turbine layout plot 
    fig = plt.figure()
    ax = plt.gca()

    ax.plot(
        [r.lon_min, r.lon_max, r.lon_max, r.lon_min, r.lon_min],
        [r.lat_min, r.lat_min, r.lat_max, r.lat_max, r.lat_min],
    )

    ax.scatter(layout["lon"], layout["lat"])
    for _, row in layout.iterrows():
        ax.text(row["lon"], row["lat"], str(int(row["turbine_id"])), fontsize=8, ha="left", va="bottom")

    ax.set_title(f"Turbine Layout | {r.name}")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_aspect("equal", adjustable="box")

    tag = inflow_path.stem.replace("inflow_u10v10_", "")
    out_layout = Path("viz/layout") / f"{region_key}_layout_{tag}.png"
    fig.savefig(out_layout, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved layout plot", out_layout)

    # Speed time series for all turbines
    fig = plt.figure()
    ax = plt.gca()
    for tid, g in df.groupby("turbine_id"):
        ax.plot(g["time"], g["speed10"], label=str(int(tid)))

    ax.set_title(f"Speed10 time series (all turbines) | {region_key}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Wind speed (m/s)")
    ax.legend(ncol=4, fontsize=7, frameon=False)

    out_speed = Path("viz/inflow") / f"{region_key}_speed10_timeseries_{tag}.png"
    fig.savefig(out_speed, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved inflow speed plot", out_speed)

    #  u vs v scatter for turbine 0
    tid0 = int(layout["turbine_id"].iloc[0])
    g0 = df[df["turbine_id"] == tid0]

    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(g0["u10"], g0["v10"])
    ax.set_title(f"u10 vs v10 scatter | turbine {tid0} | {region_key}")
    ax.set_xlabel("u10 (m/s)")
    ax.set_ylabel("v10 (m/s)")

    out_uv = Path("viz/inflow") / f"{region_key}_t{tid0}_u10_v10_scatter_{tag}.png"
    fig.savefig(out_uv, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved u/v scatter plot", out_uv)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())