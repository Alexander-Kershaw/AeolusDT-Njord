from __future__ import annotations

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from njord.config import ACTIVE_REGION_KEY, DEFAULT_START, DEFAULT_END

def get_time_name(ds: xr.Dataset) -> str:
    """
    ------------------------------------------------------------------------------------------------
    ERA5 CDS files may use 'valid_time' instead of 'time' on some occations
    -> Return the name of the time coordinate/dimension.
    ------------------------------------------------------------------------------------------------
    """
    for cand in ("time", "valid_time"):
        if cand in ds.dims or cand in ds.coords:
            return cand
    raise KeyError(f"No time referencing coord found. Coords={list(ds.coords)} Dims={list(ds.dims)}")


def wind_speed_dir_from(u: xr.DataArray, v: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    ------------------------------------------------------------------------------------------------
    Compute wind speed and meteorological wind direction:
    u, v are wind components (m/s) where u is eastward, v is northward (direction wind is going TOWARDS
    this is the general convention).
    ------------------------------------------------------------------------------------------------
    """
    speed = np.sqrt(u**2 + v**2)

    # meteorological direction FROM:
    # dir_from = atan2(-u, -v) in degrees, wrapped to [0, 360)
    dir_from = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0

    speed.name = "wind_speed_10m"
    speed.attrs["units"] = "m s-1"

    dir_from.name = "wind_dir_from_10m"
    dir_from.attrs["units"] = "degrees"
    dir_from.attrs["convention"] = "meteorological_from"

    return speed, dir_from

def main() -> int:
    region_key = ACTIVE_REGION_KEY
    nc_path = Path(f"data_lake/bronze/era5/{region_key}/era5_single_levels_u10v10_{DEFAULT_START}_to_{DEFAULT_END}.nc")
    if not nc_path.exists():
        raise FileNotFoundError(f"Missing file: {nc_path}")

    out_dir = Path(f"viz/quicklooks/{region_key}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading:", nc_path)
    ds = xr.open_dataset(nc_path)

    if "u10" not in ds or "v10" not in ds:
        raise KeyError(f"Expected variables u10 and v10. Found data_vars={list(ds.data_vars)}")

    tname = get_time_name(ds)
    print("Using time dimension:", tname)

    u10 = ds["u10"]
    v10 = ds["v10"]

    speed, dir_from = wind_speed_dir_from(u10, v10)

    # Some stats
    smin = float(speed.min().values)
    smax = float(speed.max().values)
    nt = int(speed.sizes.get(tname, -1))
    print(f"Speed range (m/s): min={smin:.2f}, max={smax:.2f}")
    print("Time steps:", nt)

    # Pull timestamps
    times = ds[tname].values
    idxs = [0, len(times)//2, len(times)-1]

    for i in idxs:
        t = np.datetime_as_string(times[i], unit="h")

        sp = speed.isel({tname: i})
        dr = dir_from.isel({tname: i})

        # Plot speed
        plt.figure()
        sp.plot()
        plt.title(f"10m Wind Speed (m/s) | {region_key} | {t}")
        f1 = out_dir / f"speed_{t.replace(':','')}.png"
        plt.savefig(f1, dpi=160, bbox_inches="tight")
        plt.close()

        # Plot direction
        plt.figure()
        dr.plot(vmin=0, vmax=360)
        plt.title(f"10m Wind Direction FROM (deg) | {region_key} | {t}")
        f2 = out_dir / f"dir_from_{t.replace(':','')}.png"
        plt.savefig(f2, dpi=160, bbox_inches="tight")
        plt.close()

        print("Saved:", f1)
        print("Saved:", f2)

    # Save a silver derived dataset
    silver_dir = Path("data_lake/silver")
    silver_dir.mkdir(parents=True, exist_ok=True)
    silver_path = silver_dir / f"{region_key}_u10v10_speed_dir_{DEFAULT_START}_to_{DEFAULT_END}.nc"

    derived = xr.Dataset(
        {
            "u10": u10,
            "v10": v10,
            "speed10": speed,
            "dir_from10": dir_from,
        }
    )
    derived.to_netcdf(silver_path)
    print("Wrote silver:", silver_path)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())