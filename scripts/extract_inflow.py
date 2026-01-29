from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import xarray as xr

from njord.config import REGIONS, DEFAULT_START, DEFAULT_END

def get_time_name(ds: xr.Dataset) -> str:
    for cand in ("time", "valid_time"):
        if cand in ds.dims or cand in ds.coords:
            return cand
    raise KeyError(f"No time coord found. Coords={list(ds.coords)} Dims={list(ds.dims)}")

def make_grid_layout(region_key: str, n_rows: int, n_cols: int, margin_frac: float = 0.12) -> pd.DataFrame:
    """
    ------------------------------------------------------------------------------------------------------------
    Simple synthetic offshore windfarm layout:

    Within the regional bounds (3 REGIONS) n_rows x n_cols linear arrangement wind farms
    (with margin so turbine locations are not on the region borders)
    ------------------------------------------------------------------------------------------------------------
    """
    r = REGIONS[region_key]
    lat_margin = (r.lat_max - r.lat_min) * margin_frac
    lon_margin = (r.lon_max - r.lon_min) * margin_frac

    lats = np.linspace(r.lat_min + lat_margin, r.lat_max - lat_margin, n_rows)
    lons = np.linspace(r.lon_min + lon_margin, r.lon_max - lon_margin, n_cols)

    pts = []
    tid = 0
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            pts.append({"turbine_id": tid, "row": i, "col": j, "lat": float(lat), "lon": float(lon)})
            tid += 1
    return pd.DataFrame(pts)

def compute_speed_dir_from(u: xr.DataArray, v: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    speed = np.sqrt(u**2 + v**2)
    dir_from = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0 # Meteorological direction
    return speed, dir_from

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract per-turbine inflow time series from ERA5 u10/v10.")
    p.add_argument("--region", required=True, choices=sorted(REGIONS.keys()))
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--rows", type=int, default=3, help="layout rows")
    p.add_argument("--cols", type=int, default=4, help="layout cols")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    region_key = args.region

    nc_path = Path(f"data_lake/bronze/era5/{region_key}/era5_single_levels_u10v10_{args.start}_to_{args.end}.nc")
    if not nc_path.exists():
        raise FileNotFoundError(f"Missing ERA5 file: {nc_path}")

    ds = xr.open_dataset(nc_path)
    tname = get_time_name(ds)

    u10 = ds["u10"]
    v10 = ds["v10"]
    speed10, dir_from10 = compute_speed_dir_from(u10, v10)

    layout = make_grid_layout(region_key, args.rows, args.cols)
    print(f"Layout turbines: {len(layout)}  (rows={args.rows}, cols={args.cols})")
    print("Layout bounds lat:", layout["lat"].min(), "->", layout["lat"].max())
    print("Layout bounds lon:", layout["lon"].min(), "->", layout["lon"].max())

    # Extract inflow per turbine with interpolation
    records = []
    times = ds[tname].values

    for _, trb in layout.iterrows():
        lat = trb["lat"]
        lon = trb["lon"]

        u_ts = u10.interp(latitude=lat, longitude=lon)
        v_ts = v10.interp(latitude=lat, longitude=lon)
        s_ts = speed10.interp(latitude=lat, longitude=lon)
        d_ts = dir_from10.interp(latitude=lat, longitude=lon)

        # Build turbine info table
        df = pd.DataFrame({
            "time": pd.to_datetime(times),
            "turbine_id": int(trb["turbine_id"]),
            "lat": float(lat),
            "lon": float(lon),
            "u10": u_ts.values.astype(float),
            "v10": v_ts.values.astype(float),
            "speed10": s_ts.values.astype(float),
            "dir_from10": d_ts.values.astype(float),
        })
        records.append(df)

    out = pd.concat(records, ignore_index=True)
    out = out.sort_values(["turbine_id", "time"]).reset_index(drop=True)

    out_dir = Path("data_lake/gold/inflow") / region_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"inflow_u10v10_{args.start}_to_{args.end}_r{args.rows}c{args.cols}.parquet"

    out.to_parquet(out_path, index=False)
    print("Wrote inflow parquet", out_path)

    # Check with summary
    print("Speed10 stats:", out["speed10"].min(), out["speed10"].mean(), out["speed10"].max())
    print("Example rows:\n", out.head(5).to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())