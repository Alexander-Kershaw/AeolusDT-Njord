from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from njord.config import REGIONS, DEFAULT_START, DEFAULT_END

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot turbine layout + basic inflow sanity plots.")
    p.add_argument("--region", required=True, choices=sorted(REGIONS.keys()))
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=4)
    return p.parse_args()

def main() -> int:
    args = parse_args()
    region_key = args.region
    r = REGIONS[region_key]

    inflow_path = Path(f"data_lake/gold/inflow/{region_key}/inflow_u10v10_{args.start}_to_{args.end}_r{args.rows}c{args.cols}.parquet")
    if not inflow_path.exists():
        raise FileNotFoundError(f"Missing inflow parquet: {inflow_path}")

    df = pd.read_parquet(inflow_path)
    df["time"] = pd.to_datetime(df["time"])


    # Regional windfarm layout plot (turbine grid)
    layout = df[["turbine_id", "lat", "lon"]].drop_duplicates().sort_values("turbine_id")

    fig = plt.figure()
    ax = plt.gca()

    # Region box
    ax.plot(
        [r.lon_min, r.lon_max, r.lon_max, r.lon_min, r.lon_min],
        [r.lat_min, r.lat_min, r.lat_max, r.lat_max, r.lat_min],
    )

    # Turbine points
    ax.scatter(layout["lon"], layout["lat"])
    for _, row in layout.iterrows():
        ax.text(row["lon"], row["lat"], str(int(row["turbine_id"])), fontsize=8, ha="left", va="bottom")

    ax.set_title(f"Turbine Layout | {r.name}")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_aspect("equal", adjustable="box")

    out_layout = Path("viz/layout") / f"{region_key}_layout_r{args.rows}c{args.cols}.png"
    fig.savefig(out_layout, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved Wind Farm DT layout plot", out_layout)

    # Inflow plots
    # Plot speed time series for all turbines
    fig = plt.figure()
    ax = plt.gca()

    for tid, g in df.groupby("turbine_id"):
        ax.plot(g["time"], g["speed10"], label=str(int(tid)))

    ax.set_title(f"Speed10 time series (all turbines) | {region_key}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Wind speed (m/s)")
    ax.legend(ncol=4, fontsize=7, frameon=False)

    out_speed = Path("viz/inflow") / f"{region_key}_speed10_timeseries_r{args.rows}c{args.cols}.png"
    fig.savefig(out_speed, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved inflow speed plot", out_speed)

    # u10 vs v10 scatter for a single turbine
    tid0 = int(layout["turbine_id"].iloc[0])
    g0 = df[df["turbine_id"] == tid0]

    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(g0["u10"], g0["v10"])
    ax.set_title(f"u10 vs v10 scatter | turbine {tid0} | {region_key}")
    ax.set_xlabel("u10 (m/s)")
    ax.set_ylabel("v10 (m/s)")

    out_uv = Path("viz/inflow") / f"{region_key}_t{tid0}_u10_v10_scatter.png"
    fig.savefig(out_uv, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved u/v scatter plot", out_uv)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())