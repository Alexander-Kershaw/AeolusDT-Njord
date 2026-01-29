from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import folium

from njord.config import REGIONS, DEFAULT_START, DEFAULT_END

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create an interactive physical map (Leaflet/Folium) of turbine locations.")
    p.add_argument("--region", required=True, choices=sorted(REGIONS.keys()))
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=4)
    return p.parse_args()

def main() -> int:
    args = parse_args()
    region_key = args.region
    r = REGIONS[region_key]

    inflow_path = Path(f"data_lake/gold/inflow/{region_key}/inflow_u10v10_{args.start}_to_{args.end}_r{args.rows}c{args.cols}.parquet")
    if not inflow_path.exists():
        raise FileNotFoundError(f"Missing inflow parquet: {inflow_path}")

    df = pd.read_parquet(inflow_path)
    layout = df[["turbine_id", "lat", "lon"]].drop_duplicates().sort_values("turbine_id")

    # Center map approx on the region mid-point
    center_lat = (r.lat_min + r.lat_max) / 2
    center_lon = (r.lon_min + r.lon_max) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="OpenStreetMap")

    # Region bounding box
    bounds = [[r.lat_min, r.lon_min], [r.lat_max, r.lon_max]]
    folium.Rectangle(
        bounds=bounds,
        color="blue",
        fill=False,
        weight=2,
        tooltip=f"{r.name} ({region_key})",
    ).add_to(m)

    # Turbine location markers
    for _, row in layout.iterrows():
        tid = int(row["turbine_id"])
        lat = float(row["lat"])
        lon = float(row["lon"])
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            popup=f"Turbine {tid}<br>lat={lat:.4f}, lon={lon:.4f}",
            tooltip=f"T{tid}",
            color="red",
            fill=True,
            fill_opacity=0.8,
        ).add_to(m)

    out_path = Path("viz/maps") / f"{region_key}_layout_r{args.rows}c{args.cols}.html"
    m.save(str(out_path))
    print("Saved interactive map", out_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())