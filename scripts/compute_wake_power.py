from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from njord.config import REGIONS

# Turbine and wake parameters
P_RATED_MW = 5.0
V_CI = 3.0
V_RATED = 12.0
V_CO = 25.0

D_M = 200.0          # rotor diameter (m)
K_WAKE = 0.05        # wake expansion coefficient
CT = 0.8             # thrust coefficient

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute wake-adjusted farm power.")
    p.add_argument("--region", required=True, choices=sorted(REGIONS.keys()))
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    return p.parse_args()

def latest_inflow_parquet(region_key: str, start: str, end: str) -> Path:
    d = Path("data_lake/gold/inflow") / region_key
    files = [f for f in d.glob("inflow_u10v10_*.parquet") if start in f.name and end in f.name]
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No inflow parquet files in {d} matching start={start}, end={end}")
    return files[0]

def power_curve_mw(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    p = np.zeros_like(v)
    m1 = (v >= V_CI) & (v < V_RATED)
    m2 = (v >= V_RATED) & (v < V_CO)
    p[m1] = P_RATED_MW * ((v[m1]**3 - V_CI**3) / (V_RATED**3 - V_CI**3))
    p[m2] = P_RATED_MW
    return p

def latlon_to_xy_km(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    -------------------------------------------------------------------------------------------
    Local tangent-plane approximation:
    x = east (km), y = north (km)
    -------------------------------------------------------------------------------------------
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.deg2rad(lat0))
    x = (lon - lon0) * km_per_deg_lon
    y = (lat - lat0) * km_per_deg_lat
    return x, y

def rotate_xy(x: np.ndarray, y: np.ndarray, theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    -------------------------------------------------------------------------------------------
    Rotates coordinates by theta_deg (counterclockwise)
    -------------------------------------------------------------------------------------------
    """
    th = np.deg2rad(theta_deg)
    xr = x * np.cos(th) - y * np.sin(th)
    yr = x * np.sin(th) + y * np.cos(th)
    return xr, yr

def jensen_deficit(dx_m: float, dy_m: float) -> float:

    # Jensen wake deficit 

    if dx_m <= 0:
        return 0.0

    r0 = D_M / 2.0
    r = r0 + K_WAKE * dx_m
    if abs(dy_m) > r:
        return 0.0

    # Standard Jensen deficit formula 
    denom = (1.0 + (K_WAKE * dx_m / r0))**2
    deficit = (1.0 - np.sqrt(1.0 - CT)) / denom
    return float(max(0.0, min(deficit, 0.95)))

def wake_adjust_speeds(v_free: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """
    -------------------------------------------------------------------------------------------
    Apply wakes at one timestep.
    v_free: per-turbine free-stream speed 
    x_m: downwind coordinate (m) 
    y_m: crosswind coordinate (m)
    -------------------------------------------------------------------------------------------
    """
    n = len(v_free)
    order = np.argsort(x_m)
    v_eff = v_free.copy()

    for ii in range(n):
        i = order[ii]
        deficits = []
        for jj in range(ii):
            j = order[jj]
            dx = x_m[i] - x_m[j]
            dy = y_m[i] - y_m[j]
            d = jensen_deficit(dx, dy)
            if d > 0:
                deficits.append(d)
        if deficits:
            d_tot = float(np.sqrt(np.sum(np.square(deficits))))
            v_eff[i] = v_free[i] * (1.0 - d_tot)
    return v_eff

def main() -> int:
    args = parse_args()
    region_key = args.region

    inflow_path = latest_inflow_parquet(region_key, args.start, args.end)
    df = pd.read_parquet(inflow_path)
    df["time"] = pd.to_datetime(df["time"])

    # turbine layout
    layout = df[["turbine_id", "lat", "lon"]].drop_duplicates().sort_values("turbine_id").reset_index(drop=True)
    lat0 = float(layout["lat"].mean())
    lon0 = float(layout["lon"].mean())

    x_km, y_km = latlon_to_xy_km(layout["lat"].values, layout["lon"].values, lat0, lon0)

    times = sorted(df["time"].unique())
    n_turb = layout.shape[0]
    farm_free = []
    farm_wake = []

    # Group by time for timestep processing
    gby = df.groupby("time", sort=True)

    for t, g in gby:
        v_free = g.sort_values("turbine_id")["speed10"].values.astype(float)
        wd_from = float(g["dir_from10"].mean())  
        # Convert to downwind rotation:
        # so direction FROM wd_from means wind blows TOWARD wd_to = wd_from + 180
        wd_to = (wd_from + 180.0) % 360.0
        theta = 90.0 - wd_to

        xr_km, yr_km = rotate_xy(x_km, y_km, theta)
        x_m = xr_km * 1000.0
        y_m = yr_km * 1000.0

        v_eff = wake_adjust_speeds(v_free, x_m, y_m)

        p_free = power_curve_mw(v_free).sum()
        p_wake = power_curve_mw(v_eff).sum()

        farm_free.append(p_free)
        farm_wake.append(p_wake)

    out = pd.DataFrame({
        "time": pd.to_datetime(list(gby.groups.keys())),
        "p_farm_free_mw": farm_free,
        "p_farm_wake_mw": farm_wake,
    })
    out["wake_loss_mw"] = out["p_farm_free_mw"] - out["p_farm_wake_mw"]
    out["wake_loss_pct"] = np.where(out["p_farm_free_mw"] > 0, out["wake_loss_mw"] / out["p_farm_free_mw"] * 100.0, 0.0)

    out_dir = Path("data_lake/gold/power") / region_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"wake_power_{inflow_path.stem}.parquet"
    out.to_parquet(out_path, index=False)

    # Plot free, wake power
    Path("viz/wake").mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(out["time"], out["p_farm_free_mw"], label="Free (no wakes)")
    plt.plot(out["time"], out["p_farm_wake_mw"], label="Wake-adjusted")
    plt.title(f"Farm Power: free vs wake-adjusted | {region_key}")
    plt.xlabel("Time")
    plt.ylabel("Farm power (MW)")
    plt.legend(frameon=False)
    plot_path = Path("viz/wake") / f"{region_key}_free_vs_wake_{args.start}_to_{args.end}.png"
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close()

    # Regional summary
    print("Region:", region_key)
    print("Inflow:", inflow_path.name)
    print("Wrote:", out_path)
    print("Plot :", plot_path)
    print("Wake loss % (mean over nonzero power):",
          float(out.loc[out["p_farm_free_mw"] > 0, "wake_loss_pct"].mean()))
    print("Wake loss % (max):", float(out["wake_loss_pct"].max()))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())