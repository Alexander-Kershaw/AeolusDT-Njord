from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from njord.config import REGIONS

# Turbine params
P_RATED_MW = 5.0 # rated power
V_CI = 3.0 # cut in speed
V_RATED = 12.0 # ramp up region
V_CO = 25.0 # cout out speed

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute turbine and farm power from inflow time series.")
    p.add_argument("--region", default="all", choices=["all"] + sorted(REGIONS.keys()))
    p.add_argument("--start", default=None, help="optional: filter inflow files containing this start date string (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="optional: filter inflow files containing this end date string (YYYY-MM-DD)")
    return p.parse_args()


def latest_inflow_parquet(region_key: str, start: str | None, end: str | None) -> Path:
    d = Path("data_lake/gold/inflow") / region_key
    files = list(d.glob("inflow_u10v10_*.parquet"))
    if start:
        files = [f for f in files if start in f.name]
    if end:
        files = [f for f in files if end in f.name]
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No matching inflow parquet files in {d} (start={start}, end={end})")
    return files[0]


def power_curve_mw(v: np.ndarray) -> np.ndarray:
    """
    ---------------------------------------------------------------------------------------------------------
    Piecewise power curve with cubic ramp between cut-in and rated.
    v in m/s, output in MW.
    ---------------------------------------------------------------------------------------------------------
    """
    v = np.asarray(v, dtype=float)
    p = np.zeros_like(v)

    # Regions
    m1 = (v >= V_CI) & (v < V_RATED)
    m2 = (v >= V_RATED) & (v < V_CO)

    # Cubic ramp
    p[m1] = P_RATED_MW * ((v[m1]**3 - V_CI**3) / (V_RATED**3 - V_CI**3))
    p[m2] = P_RATED_MW

    # v < cut-in and v >= cut-out remains 0
    return p


def plot_power_curve(out_path: Path) -> None:
    v = np.linspace(0, 30, 301)
    p = power_curve_mw(v)
    plt.figure()
    plt.plot(v, p)
    plt.title("Turbine Power Curve")
    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("Power (MW)")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def run_region(region_key: str, start: str | None, end: str | None) -> None:
    inflow_path = latest_inflow_parquet(region_key, start, end)
    df = pd.read_parquet(inflow_path)
    df["time"] = pd.to_datetime(df["time"])

    df["p_turbine_mw"] = power_curve_mw(df["speed10"].values)

    farm = (
        df.groupby("time", as_index=False)["p_turbine_mw"]
          .sum()
          .rename(columns={"p_turbine_mw": "p_farm_mw"})
    )

    out_dir = Path("data_lake/gold/power") / region_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baseline_power_{inflow_path.stem}.parquet"
    farm.to_parquet(out_path, index=False)

    # Plots
    Path("viz/power").mkdir(parents=True, exist_ok=True)

    # Farm power over time
    plt.figure()
    plt.plot(farm["time"], farm["p_farm_mw"])
    plt.title(f"Baseline Farm Power (no wakes) | {region_key}")
    plt.xlabel("Time")
    plt.ylabel("Farm power (MW)")
    farm_plot = Path("viz/power") / f"{region_key}_baseline_farm_power_{inflow_path.stem}.png"
    plt.savefig(farm_plot, dpi=180, bbox_inches="tight")
    plt.close()

    # Wind speed histogram 
    plt.figure()
    plt.hist(df["speed10"].values, bins=30)
    plt.title(f"Wind speed distribution (speed10) | {region_key}")
    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("Count")
    hist_plot = Path("viz/power") / f"{region_key}_speed10_hist_{inflow_path.stem}.png"
    plt.savefig(hist_plot, dpi=180, bbox_inches="tight")
    plt.close()

    # Power curve 
    curve_plot = Path("viz/power") / "baseline_power_curve.png"
    if not curve_plot.exists():
        plot_power_curve(curve_plot)

    # Summary
    n_turb = df["turbine_id"].nunique()
    print(region_key, "turbines:", n_turb)
    print(region_key, "inflow:", inflow_path.name)
    print(region_key, "power :", out_path)
    print(region_key, "plots :", farm_plot.name, "|", hist_plot.name)
    print(region_key, "farm power stats (MW):",
          float(farm["p_farm_mw"].min()),
          float(farm["p_farm_mw"].mean()),
          float(farm["p_farm_mw"].max()))

def main() -> int:
    args = parse_args()
    if args.region == "all":
        for rk in REGIONS.keys():
            run_region(rk, args.start, args.end)
    else:
        run_region(args.region, args.start, args.end)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
