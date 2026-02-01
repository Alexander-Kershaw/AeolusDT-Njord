from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from njord.config import HEIGHTS

R_D = 287.05          # J/(kg*K)
RHO0 = 1.225          # kg/m^3 (reference density)
P_RATED_MW = 5.0      # per turbine
V_CI = 3.0
V_RATED = 12.0
V_CO = 25.0

def power_curve_mw(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    p = np.zeros_like(v)
    m1 = (v >= V_CI) & (v < V_RATED)
    m2 = (v >= V_RATED) & (v < V_CO)
    p[m1] = P_RATED_MW * ((v[m1]**3 - V_CI**3) / (V_RATED**3 - V_CI**3))
    p[m2] = P_RATED_MW
    return p

def get_time_name(ds: xr.Dataset) -> str:
    for cand in ("time", "valid_time"):
        if cand in ds.dims or cand in ds.coords:
            return cand
    raise KeyError(f"No time-like coord found. Coords={list(ds.coords)} Dims={list(ds.dims)}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute density-corrected power using ERA5 sp+t2m.")
    p.add_argument("--region", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--height", type=int, default=100, choices=HEIGHTS, help="Choice of speed10 or speed100 inflow (hub height assumption 10m/100m)")
    return p.parse_args()

def latest_inflow(region: str, start: str, end: str, height: int) -> Path:
    d = Path("data_lake/gold/inflow") / region
    pat = f"inflow_u{height}v{height}_{start}_to_{end}_*.parquet"
    files = sorted(d.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No inflow parquet in {d} matching {pat}")
    return files[0]

def main() -> int:
    args = parse_args()
    region = args.region
    start = args.start
    end = args.end
    height = args.height

    speed_col = f"speed{height}"

    inflow_path = latest_inflow(region, start, end, height)
    inflow = pd.read_parquet(inflow_path)
    inflow["time"] = pd.to_datetime(inflow["time"])

    # ERA5 (sp, t2m)
    met_path = Path(f"data_lake/bronze/era5/{region}/era5_single_levels_sp_t2m_{start}_to_{end}.nc")
    if not met_path.exists():
        raise FileNotFoundError(f"Missing met file: {met_path}")

    ds = xr.open_dataset(met_path)
    tname = get_time_name(ds)

    # Precompute time index mapping for speed (since merge will be on timestamps)
    times = pd.to_datetime(ds[tname].values)

    # Turbine layout from inflow
    layout = inflow[["turbine_id", "lat", "lon"]].drop_duplicates().sort_values("turbine_id").reset_index(drop=True)

    # Interpolate sp/t2m to each turbine and build density (rho) timeseries per turbine
    records = []
    for _, trb in layout.iterrows():
        tid = int(trb["turbine_id"])
        lat = float(trb["lat"])
        lon = float(trb["lon"])

        sp_ts = ds["sp"].interp(latitude=lat, longitude=lon).values.astype(float)   # Pa
        t2m_ts = ds["t2m"].interp(latitude=lat, longitude=lon).values.astype(float) # K
        rho_ts = sp_ts / (R_D * t2m_ts)

        records.append(pd.DataFrame({
            "time": times,
            "turbine_id": tid,
            "rho": rho_ts,
        }))

    rho_df = pd.concat(records, ignore_index=True)

    # Merge density into inflow (on time+turbine_id)
    df = inflow.merge(rho_df, on=["time", "turbine_id"], how="left")
    if df["rho"].isna().any():
        missing = int(df["rho"].isna().sum())
        raise RuntimeError(f"Density merge produced {missing} missing rho values. Check time alignment.")

    # Baseline power from speed
    p_base = power_curve_mw(df[speed_col].values)

    # Density adjustment
    scale = (df["rho"].values / RHO0).astype(float)
    p_rho = p_base * scale
    p_rho = np.clip(p_rho, 0.0, P_RATED_MW)

    df["p_turbine_base_mw"] = p_base
    df["p_turbine_rho_mw"] = p_rho

    # Farm level stats
    farm = (
        df.groupby("time")[["p_turbine_base_mw", "p_turbine_rho_mw"]]
          .sum()
          .reset_index()
          .rename(columns={"p_turbine_base_mw": "p_farm_base_mw", "p_turbine_rho_mw": "p_farm_rho_mw"})
    )
    farm["delta_mw"] = farm["p_farm_rho_mw"] - farm["p_farm_base_mw"]
    farm["delta_pct"] = np.where(farm["p_farm_base_mw"] > 0, farm["delta_mw"] / farm["p_farm_base_mw"] * 100.0, 0.0)

    out_dir = Path("data_lake/gold/power") / region
    out_dir.mkdir(parents=True, exist_ok=True)

    out_detail = out_dir / f"density_power_{inflow_path.stem}.parquet"
    out_farm = out_dir / f"density_farm_{inflow_path.stem}.parquet"
    df.to_parquet(out_detail, index=False)
    farm.to_parquet(out_farm, index=False)

    Path("viz/power").mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(farm["time"], farm["p_farm_base_mw"], label="Baseline power curve")
    plt.plot(farm["time"], farm["p_farm_rho_mw"], label="Density-corrected power curve")
    plt.title(f"Farm power: baseline VS density-corrected | {region} | {height}m winds")
    plt.xlabel("Time")
    plt.ylabel("Farm power (MW)")
    plt.legend(frameon=False)
    plot_path = Path("viz/power") / f"{region}_baseline_vs_density_{start}_to_{end}_{height}m.png"
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close()

    print("Region:", region)
    print("Inflow:", inflow_path.name)
    print("Met  :", met_path.name)
    print("Wrote detail:", out_detail)
    print("Wrote farm  :", out_farm)
    print("Plot:", plot_path)

    print("rho stats (kg/m3): min={:.3f} mean={:.3f} max={:.3f}".format(df["rho"].min(), df["rho"].mean(), df["rho"].max()))
    print("farm delta pct (producing only): mean={:.3f}% max={:.3f}%".format(
        float(farm.loc[farm["p_farm_base_mw"] > 0, "delta_pct"].mean()),
        float(farm["delta_pct"].max()),
    ))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())