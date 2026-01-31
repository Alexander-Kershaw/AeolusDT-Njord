from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from njord.config import HEIGHTS

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot wake loss VS wind direction.")
    p.add_argument("--region", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--height", type=int, default=10, choices=HEIGHTS, help="use dir_from at 10m or 100m")
    p.add_argument("--bin-deg", type=int, default=15, help="direction bin width in degrees")
    return p.parse_args()

def load_latest_inflow(region: str, start: str, end: str, height: int) -> Path:
    d = Path("data_lake/gold/inflow") / region
    pat = f"inflow_u{height}v{height}_{start}_to_{end}_*.parquet"
    files = sorted(d.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No inflow parquet for {region} matching {pat}")
    return files[0]

def load_wake_power(region: str, start: str, end: str, height: int) -> Path:
    d = Path("data_lake/gold/power") / region
    pat = f"wake_power_inflow_u{height}v{height}_{start}_to_{end}_*.parquet"
    files = sorted(d.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No wake_power parquet for {region} matching {pat}")
    return files[0]

def circmean_deg(deg: np.ndarray) -> float:
    #Circular mean for degrees 
    rad = np.deg2rad(deg)
    s = np.sin(rad).mean()
    c = np.cos(rad).mean()
    ang = np.arctan2(s, c)
    return float((np.rad2deg(ang) + 360.0) % 360.0)


def main() -> int:
    args = parse_args()
    region = args.region
    start, end = args.start, args.end
    height = args.height
    bin_deg = int(args.bin_deg)

    dir_col = f"dir_from{height}"

    inflow_path = load_latest_inflow(region, start, end, height)
    wake_path = load_wake_power(region, start, end, height)

    inflow = pd.read_parquet(inflow_path)
    inflow["time"] = pd.to_datetime(inflow["time"])
    if dir_col not in inflow.columns:
        raise KeyError(f"Missing {dir_col} in {inflow_path.name}. Columns={list(inflow.columns)}")

    wake = pd.read_parquet(wake_path)
    wake["time"] = pd.to_datetime(wake["time"])

    dir_by_time = (
        inflow.groupby("time")[dir_col]
             .apply(lambda s: circmean_deg(s.values.astype(float)))
             .reset_index()
             .rename(columns={dir_col: "dir_from_mean"})
    )

    df = wake.merge(dir_by_time, on="time", how="left")
    df = df[df["p_farm_free_mw"] > 0].copy()

    bins = np.arange(0, 360 + bin_deg, bin_deg)
    labels = bins[:-1]
    df["dir_bin"] = pd.cut(df["dir_from_mean"], bins=bins, right=False, include_lowest=True, labels=labels)

    grouped = df.groupby("dir_bin", observed=True)["wake_loss_pct"].mean().reset_index()
    grouped["dir_bin"] = grouped["dir_bin"].astype(float)

    plt.figure()
    plt.plot(grouped["dir_bin"], grouped["wake_loss_pct"])
    plt.title(f"Mean wake loss (%) VS wind direction-from | {region} | {height}m | bin={bin_deg}°")
    plt.xlabel("Wind direction-from (deg, binned)")
    plt.ylabel("Mean wake loss (%)")

    out = Path("viz/wake") / f"{region}_wake_loss_vs_direction_{start}_to_{end}_{height}m_bin{bin_deg}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()

    print("Using inflow:", inflow_path.name)
    print("Using wake :", wake_path.name)
    print("Saved plot", out)
    print("Wake loss pct summary (producing power only): mean=",
          float(df["wake_loss_pct"].mean()), "max=", float(df["wake_loss_pct"].max()))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())