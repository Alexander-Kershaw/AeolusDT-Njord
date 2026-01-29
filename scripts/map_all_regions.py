from __future__ import annotations

from pathlib import Path
import pandas as pd
import folium

from njord.config import REGIONS

REGION_LABELS = {
    "tampen_box_a": "Hywind Tampen",
    "utsira_nord_box_b": "Utsira Nord",
    "sn2_box_c": "Sørlige Nordsjø II",
}

def latest_inflow_parquet(region_key: str) -> Path:
    d = Path("data_lake/gold/inflow") / region_key
    files = sorted(d.glob("inflow_u10v10_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No inflow parquet files found in {d}")
    return files[0]

def centroid(r) -> tuple[float, float]:
    return ((r.lat_min + r.lat_max) / 2.0, (r.lon_min + r.lon_max) / 2.0)

def main() -> int:
    cents = [centroid(r) for r in REGIONS.values()]
    center_lat = sum(c[0] for c in cents) / len(cents)
    center_lon = sum(c[1] for c in cents) / len(cents)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, control_scale=True, tiles=None)

    # Some more aesthetically pleasing basemaps
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

    # Region Labels
    label_html = lambda txt: f"""
    <div style="
        font-size: 14px;
        font-weight: 800;
        color: #111;
        background: transparent;
        padding: 0;
        margin: 0;
        white-space: nowrap;
        text-shadow:
            -1px -1px 0 rgba(255,255,255,0.85),
             1px -1px 0 rgba(255,255,255,0.85),
            -1px  1px 0 rgba(255,255,255,0.85),
             1px  1px 0 rgba(255,255,255,0.85);
    ">
        {txt}
    </div>
    """

    for key, r in REGIONS.items():
        label = REGION_LABELS.get(key, r.name)
        fg = folium.FeatureGroup(name=label, show=True)

        # Region box
        folium.Rectangle(
            bounds=[[r.lat_min, r.lon_min], [r.lat_max, r.lon_max]],
            color="dodgerblue",
            weight=2,
            fill=False,
            tooltip=label,
        ).add_to(fg)

        # Label at centroid
        clat, clon = centroid(r)
        folium.Marker(
            location=[clat, clon],
            icon=folium.DivIcon(html=label_html(label)),
            tooltip=label
        ).add_to(fg)

        # Turbines from latest inflow parquet
        p = latest_inflow_parquet(key)
        df = pd.read_parquet(p)
        layout = df[["turbine_id", "lat", "lon"]].drop_duplicates().sort_values("turbine_id")

        for _, row in layout.iterrows():
            tid = int(row["turbine_id"])
            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lon"])],
                radius=5,
                tooltip=f"{label} | T{tid}",
                popup=f"{label}<br>Turbine {tid}<br><small>{p.name}</small>",
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
    print("Saved combined regional turbine map", out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())