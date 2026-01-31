from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import xarray as xr

from njord.config import REGIONS, DEFAULT_START, DEFAULT_END, FARM_SITES, HEIGHTS

def get_time_name(ds: xr.Dataset) -> str:
    for cand in ("time", "valid_time"):
        if cand in ds.dims or cand in ds.coords:
            return cand
    raise KeyError(f"No time coord found. Coords={list(ds.coords)} Dims={list(ds.dims)}")

# Conversions for offshore centroid farm localising
def km_to_deg_lat(km: float) -> float:
    return km / 111.0

def km_to_deg_lon(km: float, lat_deg: float) -> float:
    return km / (111.0 * np.cos(np.deg2rad(lat_deg)))


def make_compact_farm_layout(region_key: str, n_rows: int, n_cols: int, spacing_km: float) -> pd.DataFrame:
    """
    --------------------------------------------------------------------------------------------------------
    Compact farm layout around a region-specific offshore centroid (more realistic
    and beneficial for wake models)

    Produces a n_rows x n_cols grid with spacing,spacing_km.
    --------------------------------------------------------------------------------------------------------
    """
    if region_key not in FARM_SITES:
        raise KeyError(f"No FARM_SITES entry for region {region_key}")

    site = FARM_SITES[region_key]
    r = REGIONS[region_key]

    dlat = km_to_deg_lat(spacing_km)
    dlon = km_to_deg_lon(spacing_km, site.lat0)

    # Center grid on (lat0, lon0)
    row_offsets = (np.arange(n_rows) - (n_rows - 1) / 2.0) * dlat
    col_offsets = (np.arange(n_cols) - (n_cols - 1) / 2.0) * dlon

    pts = []
    tid = 0
    for i, roff in enumerate(row_offsets):
        for j, coff in enumerate(col_offsets):
            lat = float(site.lat0 + roff)
            lon = float(site.lon0 + coff)

            # Must be inside the region box boundaries
            if not (r.lat_min <= lat <= r.lat_max and r.lon_min <= lon <= r.lon_max):
                raise ValueError(
                    f"Generated turbine outside region box for {region_key}. "
                    f"(lat,lon)=({lat:.4f},{lon:.4f}) not in "
                    f"[{r.lat_min},{r.lat_max}]x[{r.lon_min},{r.lon_max}]. "
                    f"Try smaller spacing_km."
                )

            pts.append({"turbine_id": tid, "row": i, "col": j, "lat": lat, "lon": lon})
            tid += 1

    return pd.DataFrame(pts)

def compute_speed_dir_from(u: xr.DataArray, v: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    speed = np.sqrt(u**2 + v**2)
    dir_from = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return speed, dir_from

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract per-turbine inflow time series from ERA5 wind components u/v.")
    p.add_argument("--region", required=True, choices=sorted(REGIONS.keys()))
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--spacing-km", type=float, default=1.2, help="turbine spacing in km (approx)")
    p.add_argument("--height", type=int, default=10, choices=HEIGHTS, help="wind component height (m): 10 or 100")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    region_key = args.region

    u_name = f"u{args.height}"
    v_name = f"v{args.height}"
    s_name = f"speed{args.height}"
    d_name = f"dir_from{args.height}"

    nc_path = Path(f"data_lake/bronze/era5/{region_key}/era5_single_levels_{u_name}{v_name}_{args.start}_to_{args.end}.nc")
    if not nc_path.exists():
        raise FileNotFoundError(f"Missing ERA5 file: {nc_path}")

    ds = xr.open_dataset(nc_path)
    tname = get_time_name(ds)

    # Fail fast if names are missing
    missing = [name for name in (u_name, v_name) if name not in ds.data_vars]
    if missing:
        raise KeyError(f"Expected variables {missing} not found in {nc_path}. Present={list(ds.data_vars)}")

    u = ds[u_name]
    v = ds[v_name]
    speed, dir_from = compute_speed_dir_from(u, v)

    layout = make_compact_farm_layout(region_key, args.rows, args.cols, args.spacing_km)
    print(f"Layout turbines: {len(layout)}  (rows={args.rows}, cols={args.cols}) spacing_km={args.spacing_km}")
    print("Using height:", args.height, "m")
    print("ERA5 file:", nc_path.name)

    records = []
    times = ds[tname].values

    for _, trb in layout.iterrows():
        lat = float(trb["lat"])
        lon = float(trb["lon"])

        u_ts = u.interp(latitude=lat, longitude=lon)
        v_ts = v.interp(latitude=lat, longitude=lon)
        s_ts = speed.interp(latitude=lat, longitude=lon)
        d_ts = dir_from.interp(latitude=lat, longitude=lon)

        df = pd.DataFrame({
            "time": pd.to_datetime(times),
            "turbine_id": int(trb["turbine_id"]),
            "lat": lat,
            "lon": lon,
            u_name: u_ts.values.astype(float),
            v_name: v_ts.values.astype(float),
            s_name: s_ts.values.astype(float),
            d_name: d_ts.values.astype(float),
        })
        records.append(df)

    out = pd.concat(records, ignore_index=True).sort_values(["turbine_id", "time"]).reset_index(drop=True)

    out_dir = Path("data_lake/gold/inflow") / region_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"inflow_{u_name}{v_name}_{args.start}_to_{args.end}_r{args.rows}c{args.cols}_sp{args.spacing_km:.1f}km.parquet"
    out.to_parquet(out_path, index=False)

    print("Wrote inflow parquet", out_path)
    print(f"{s_name} stats:", out[s_name].min(), out[s_name].mean(), out[s_name].max())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())