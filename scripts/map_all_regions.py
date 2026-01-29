from __future__ import annotations

from pathlib import Path
import pandas as pd
import folium

from njord.config import REGIONS, DEFAULT_START, DEFAULT_END

REGION_LABELS = {
    "tampen_box_a": "Hywind Tampen (Box A)",
    "utsira_nord_box_b": "Utsira Nord (Box B)",
    "sn2_box_c": "Sørlige Nordsjø II (Box C)",
}

def inflow_path(region_key: str, start: str, end: str, rows: int, cols: int) -> Path:
    return Path(f"data_lake/gold/inflow/{region_key}/inflow_u10v10_{start}_to_{end}_r{rows}c{cols}.parquet")

def load_layout(region_key: str, start: str, end: str, rows: int, cols: int) -> pd.DataFrame:
    p = inflow_path(region_key, start, end, rows, cols)
    if not p.exists():
        raise FileNotFoundError(f"Missing inflow parquet for {region_key}: {p}")
    df = pd.read_parquet(p)
    return df[["turbine_id", "lat", "lon"]].drop_duplicates().sort_values("turbine_id")

def centroid(r) -> tuple[float, float]:
    return ( (r.lat_min + r.lat_max) / 2.0, (r.lon_min + r.lon_max) / 2.0 )

def main() -> int:
    start, end = DEFAULT_START, DEFAULT_END
    rows, cols = 3, 4

    cents = [centroid(r) for r in REGIONS.values()]
    center_lat = sum(c[0] for c in cents) / len(cents)
    center_lon = sum(c[1] for c in cents) / len(cents)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        control_scale=True,
        tiles=None,
    )

    # Some more asethetically pleasing basemaps
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        name="Pretty (CartoDB Positron)",
        attr="© OpenStreetMap contributors © CARTO",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        name="OpenStreetMap",
        attr="© OpenStreetMap contributors",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg",
        name="Terrain",
        attr="Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors",
    ).add_to(m)

    # All regions on this map
    for key, r in REGIONS.items():
        label = REGION_LABELS.get(key, r.name)
        fg = folium.FeatureGroup(name=label, show=True)

        # Bounding box
        folium.Rectangle(
            bounds=[[r.lat_min, r.lon_min], [r.lat_max, r.lon_max]],
            color="dodgerblue",
            weight=2,
            fill=False,
            tooltip=label,
        ).add_to(fg)

        # Label marker
        clat, clon = centroid(r)
        folium.Marker(
            location=[clat, clon],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size:12px;
                    font-weight:700;
                    background:rgba(255,255,255,0.85);
                    padding:4px 6px;
                    border-radius:6px;
                    border:1px solid rgba(0,0,0,0.25);
                    box-shadow:0 1px 4px rgba(0,0,0,0.3);
                ">
                    {label}
                </div>
                """
            )
        ).add_to(fg)

        # Turbines markers
        layout = load_layout(key, start, end, rows, cols)
        for _, row in layout.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=5,
                tooltip=f"{label} | T{int(row['turbine_id'])}",
                popup=f"{label}<br>Turbine {int(row['turbine_id'])}",
                color="#e11d48",
                fill=True,
                fill_opacity=0.85,
                weight=2,
            ).add_to(fg)

        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    out = Path("viz/maps") / "njord_all_regions.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print("Saved combined regional turbine layout map", out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())