from __future__ import annotations

from pathlib import Path
import sys

import cdsapi
import argparse
import pandas as pd

from njord.config import REGIONS, ACTIVE_REGION_KEY, DEFAULT_START, DEFAULT_END, HEIGHTS

PRESETS = {
    "wind10": (["10m_u_component_of_wind", "10m_v_component_of_wind"], "u10v10"),
    "wind100": (["100m_u_component_of_wind", "100m_v_component_of_wind"], "u100v100"),
    "met": (["surface_pressure", "2m_temperature"], "sp_t2m"),
}

def area_nwse(region_key: str) -> list[float]:
    """
    Area bounds: [North, West, South, East]
    """
    r = REGIONS[region_key]
    return [r.lat_max, r.lon_min, r.lat_min, r.lon_max]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download ERA5 single levels wind components for a region.")
    p.add_argument("--region", default=ACTIVE_REGION_KEY, choices=sorted(REGIONS.keys()))
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--height", type=int, default=10, choices=HEIGHTS, help="wind component height (m): 10 or 100")
    p.add_argument("--vars", default="wind10", choices=sorted(PRESETS.keys()), help="dataset preset: wind10, wind100, met(sp+t2m)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    r = REGIONS[args.region]

    variables, tag = PRESETS[args.vars]

    dates = pd.date_range(args.start, args.end, freq="D")
    years = sorted({d.strftime("%Y") for d in dates})
    months = sorted({d.strftime("%m") for d in dates})
    days = sorted({d.strftime("%d") for d in dates})

    area = [r.lat_max, r.lon_min, r.lat_min, r.lon_max]  # [N, W, S, E]
    out_dir = Path("data_lake/bronze/era5") / args.region
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"era5_single_levels_{tag}_{args.start}_to_{args.end}.nc"

    req = {
        "product_type": "reanalysis",
        "variable": variables,
        "year": years,
        "month": months,
        "day": days,
        "time": [f"{h:02d}:00" for h in range(24)],
        "format": "netcdf",
        "area": area,
    }

    print(f"Requesting ERA5: {args.region}")
    print("Preset:", args.vars, "| variables:", variables)
    print("Area [N,W,S,E]:", area)
    print("Dates:", f"{args.start} to {args.end}")
    print("Output:", out_path)

    c = cdsapi.Client()
    c.retrieve("reanalysis-era5-single-levels", req, str(out_path))
    print("Download complete", out_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())