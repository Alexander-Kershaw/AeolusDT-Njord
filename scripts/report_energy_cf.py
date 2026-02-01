from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from njord.config import HEIGHTS

P_RATED_MW = 5.0  # Power rating per turbine

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Energy and capacity factor report for wind farm regions.")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--height", type=int, default=100, choices=HEIGHTS)
    return p.parse_args()

def hours_in_period(start: str, end: str) -> float:
    t0 = pd.to_datetime(start)
    t1 = pd.to_datetime(end) + pd.Timedelta(days=1)
    return float((t1 - t0) / pd.Timedelta(hours=1))

def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    return df

def region_turbine_count(region: str, start: str, end: str, height: int) -> int:
    inflow_dir = Path("data_lake/gold/inflow") / region
    inflow_pat = f"inflow_u{height}v{height}_{start}_to_{end}_*.parquet"
    files = sorted(inflow_dir.glob(inflow_pat), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No inflow parquet for {region} matching {inflow_pat}")
    inflow = pd.read_parquet(files[0])
    return int(inflow["turbine_id"].nunique())

def main() -> int:
    args = parse_args()
    start, end, height = args.start, args.end, args.height
    H = hours_in_period(start, end)

    regions = ["tampen_box_a", "utsira_nord_box_b", "sn2_box_c"]

    rows = []
    for region in regions:
        n_turb = region_turbine_count(region, start, end, height)
        p_cap = n_turb * P_RATED_MW

        # Baseline power
        base_path = Path(f"data_lake/gold/power/{region}/baseline_power_inflow_u{height}v{height}_{start}_to_{end}_r3c4_sp1.2km.parquet")
        base = load_parquet(base_path)

        # Find all potential baseline power cols/rows/ per-turbine/ farm-level
        if "p_farm_mw" in base.columns:
            p_base = base["p_farm_mw"].values
        elif "p_farm_base_mw" in base.columns:
            p_base = base["p_farm_base_mw"].values
        elif "p_turbine_mw" in base.columns:
            p_base = base.groupby("time")["p_turbine_mw"].sum().values
        else:
            cand = [c for c in base.columns if c.startswith("p_") and c.endswith("_mw")]
            raise KeyError(f"Cannot infer farm power in {base_path.name}. Columns={list(base.columns)} candidates={cand}")

        e_base = float(np.sum(p_base))  # MW * hours => MWh 
        cf_base = e_base / (p_cap * H) if p_cap > 0 else 0.0

        # Wake adjustment
        wake_path = Path(f"data_lake/gold/power/{region}/wake_power_inflow_u{height}v{height}_{start}_to_{end}_r3c4_sp1.2km.parquet")
        wake = load_parquet(wake_path)
        p_wake = wake["p_farm_wake_mw"].values
        e_wake = float(np.sum(p_wake))
        cf_wake = e_wake / (p_cap * H) if p_cap > 0 else 0.0

        # Density correction
        dens_farm_path = Path(f"data_lake/gold/power/{region}/density_farm_inflow_u{height}v{height}_{start}_to_{end}_r3c4_sp1.2km.parquet")
        dens = load_parquet(dens_farm_path)
        p_dens = dens["p_farm_rho_mw"].values
        e_dens = float(np.sum(p_dens))
        cf_dens = e_dens / (p_cap * H) if p_cap > 0 else 0.0

        rows.append({
            "region": region,
            "n_turbines": n_turb,
            "capacity_mw": p_cap,
            "hours": H,
            "energy_baseline_mwh": e_base,
            "cf_baseline": cf_base,
            "energy_wake_mwh": e_wake,
            "cf_wake": cf_wake,
            "energy_density_mwh": e_dens,
            "cf_density": cf_dens,
        })

    report = pd.DataFrame(rows)

    out_dir = Path("data_lake/gold/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"energy_cf_{start}_to_{end}_{height}m.csv"
    report.to_csv(out_csv, index=False)

    print(report.to_string(index=False))
    print("Wrote:", out_csv)

    # Capacity factor comparison
    Path("viz/reports").mkdir(parents=True, exist_ok=True)
    plt.figure()
    x = np.arange(len(report))
    plt.plot(report["region"], report["cf_baseline"], label="Baseline")
    plt.plot(report["region"], report["cf_wake"], label="Wake")
    plt.plot(report["region"], report["cf_density"], label="Density")
    plt.title(f"Capacity factor comparison | {start} to {end} | {height}m")
    plt.xlabel("Region")
    plt.ylabel("Capacity factor")
    plt.legend(frameon=False)
    out_plot = Path("viz/reports") / f"cf_compare_{start}_to_{end}_{height}m.png"
    plt.savefig(out_plot, dpi=180, bbox_inches="tight")
    plt.close()

    print("Plot:", out_plot)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())