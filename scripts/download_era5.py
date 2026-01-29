from __future__ import annotations

from pathlib import Path
import sys

import cdsapi
import argparse

from njord.config import REGIONS, ACTIVE_REGION_KEY, DEFAULT_START, DEFAULT_END

def area_nwse(region_key: str) -> list[float]:
    """
    Area bounds: [North, West, South, East]
    """
    r = REGIONS[region_key]
    return [r.lat_max, r.lon_min, r.lat_min, r.lon_max]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download ERA5 single levels u10/v10 for a region.")
    p.add_argument("--region", default=ACTIVE_REGION_KEY, choices=sorted(REGIONS.keys()))
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    return p.parse_args()


def date_range_yyyymmdd(start: str, end: str) -> tuple[str, str]:
    # CDS uses 'YYYY-MM-DD/YYYY-MM-DD'
    return (start, end)

def main() -> int:
    args = parse_args()

    out_dir = Path("data_lake/bronze/era5") / args.region
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"era5_single_levels_u10v10_{args.start}_to_{args.end}.nc"

    req = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "date": f"{args.start}/{args.end}",
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area_nwse(args.region),
    }

    print("Requesting ERA5:", args.region)
    print("Area [N,W,S,E]:", req["area"])
    print("Dates:", req["date"])
    print("Output:", out_path)

    c = cdsapi.Client()
    c.retrieve("reanalysis-era5-single-levels", req, str(out_path))

    print("Download complete", out_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())