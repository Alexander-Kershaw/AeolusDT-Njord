***
# AeolusDT-Njord
***
AeolusDT-Njord is an offshore wind farm digital twin for testing wind turbine arrangements with historical ERA5 data: ERA5 wind-field snapshots -> turbine inflow extraction -> farm power (baseline + wake + air-density correction) -> energy & capacity factor reporting -> interactive map dashboard with wind overlays and vectors.

**What this is**
- A *replay* digital twin: pick a historical window and Njord reconstructs the wind field and power response for a wind farm configuration of N turbines in a grid arrangement.
- A pragmatic pipeline: netCDF -> derived fields -> parquet -> plots -> dashboard.

**What this is NOT**
- CFD / LES microscale simulation
- full aeroelastic turbine physics (OpenFAST was tested but intentionally skipped for now)
- forecasting system (no ML forecast here, purely replay/analysis)
- SCADA twin (unless added later)

---

## Features
### Wind field
- Downloads ERA5 single-level winds via CDS API
- Computes speed and wind direction (from-direction convention)
- Produces quicklook plots for selected timesteps
- Saves derived wind fields to a “silver layer” netCDF

### Farm twin
- Creates a synthetic turbine layout within each offshore region box
- Interpolates gridded wind to turbine points (hourly)
- Writes turbine inflow time series to parquet

### Power models
- Baseline power curve model (cut-in / rated / cut-out)
- Simple wake loss approximation (direction-aware ordering and deficit)
- Air density correction using ERA5 surface pressure and 2m temperature
- Energy and capacity factor reporting (baseline vs wake vs density)

### Dashboard
- Interactive Folium map:
  - All region boxes and turbine markers
  - Wind-speed raster overlay from ERA5
  - Optional sparse direction arrows (scaled and capped)
  - Colorbar legend
  - Time scrubber (Prev/Next and index slider)
- Time series plots, KPI cards, and data provenance panel

---

## Regions
Configured in `src/njord/config.py` as three offshore boxes:
- **Hywind Tampen** (Box A)  
- **Utsira Nord** (Box B)  
- **Sørlige Nordsjø II** (Box C)

These are *analysis boxes* (not exact lease polygons). The goal is a stable, reproducible pipeline.

---

## Repo layout

```text
├── README.md
├── data_lake
│   ├── bronze
│   │   └── era5
│   ├── gold
│   │   ├── inflow
│   │   ├── power
│   │   └── reports
│   └── silver
├── njord_dashboard.py
├── pyproject.toml
├── requirements.txt
├── scripts
│   ├── compute_density_power.py
│   ├── compute_power.py
│   ├── compute_power_boxA.py
│   ├── compute_wake_power.py
│   ├── download_era5.py
│   ├── extract_inflow.py
│   ├── map_all_regions.py
│   ├── plot_layout_and_inflow.py
│   ├── plot_wake_direction.py
│   ├── report_energy_cf.py
│   ├── turbines_map.py
│   └── wind_snapshots.py
├── src
│   └── njord
│       ├── config.py
│       ├── farm
│       ├── field
│       ├── io
│       ├── openfast
│       └── power
└── viz
    ├── dashboard
    ├── inflow
    ├── layout
    ├── maps
    ├── power
    ├── quicklooks
    ├── reports
    └── wake



```
---

## Setup

### Create/activate environment


```bash
conda activate aeolusDT_njord_env
pip install -e .
```

### CDS API Credentials

ERA5 downloads require Copernicus CDS access:

- Place credentials in `~/.cdsapirc`
- Conform it works by running a download command (see next section)

Ensure CDS API configuration is correct or everything downstream is irrelevant.

---

## Commands

### Download ERA5 Wind Data (u/v at 100m elevation)

Run for each region for a given time interval (example: January 2020):

```bash
python scripts/download_era5.py --region tampen_box_a --start 2020-01-01 --end 2020-01-31 --height 100
python scripts/download_era5.py --region utsira_nord_box_b --start 2020-01-01 --end 2020-01-31 --height 100
python scripts/download_era5.py --region sn2_box_c --start 2020-01-01 --end 2020-01-31 --height 100
```

***

### Wind Snapshots and Silver Data

```bash
python scripts/wind_snapshots.py --region tampen_box_a --start 2020-01-01 --end 2020-01-31 --height 100
python scripts/wind_snapshots.py --region utsira_nord_box_b --start 2020-01-01 --end 2020-01-31 --height 100
python scripts/wind_snapshots.py --region sn2_box_c --start 2020-01-01 --end 2020-01-31 --height 100
```

***

### Turbine Layout and Inflow Extraction

```bash
python scripts/extract_inflow.py --region tampen_box_a --start 2020-01-01 --end 2020-01-31 --rows 3 --cols 4 --spacing-km 1.2 --height 100
python scripts/extract_inflow.py --region utsira_nord_box_b --start 2020-01-01 --end 2020-01-31 --rows 3 --cols 4 --spacing-km 1.2 --height 100
python scripts/extract_inflow.py --region sn2_box_c --start 2020-01-01 --end 2020-01-31 --rows 3 --cols 4 --spacing-km 1.2 --height 100
```

***

### Baseline Power

```bash
python scripts/compute_power.py --region tampen_box_a --start 2020-01-01 --end 2020-01-31 --height 100
python scripts/compute_power.py --region utsira_nord_box_b --start 2020-01-01 --end 2020-01-31 --height 100
python scripts/compute_power.py --region sn2_box_c --start 2020-01-01 --end 2020-01-31 --height 100
```

***

### Wake Power Approximation

```bash
python scripts/compute_wake_power.py --region tampen_box_a --start 2020-01-01 --end 2020-01-31 --height 100
python scripts/compute_wake_power.py --region utsira_nord_box_b --start 2020-01-01 --end 2020-01-31 --height 100
python scripts/compute_wake_power.py --region sn2_box_c --start 2020-01-01 --end 2020-01-31 --height 100
```

***

### Optional Wake Loss VS Wind Direction Bins

```bash
python scripts/plot_wake_direction.py --region tampen_box_a --start 2020-01-01 --end 2020-01-31 --height 100 --bin-deg 15
python scripts/plot_wake_direction.py --region utsira_nord_box_b --start 2020-01-01 --end 2020-01-31 --height 100 --bin-deg 15
python scripts/plot_wake_direction.py --region sn2_box_c --start 2020-01-01 --end 2020-01-31 --height 100 --bin-deg 15
```

***

### Density-corrected Power

```bash
python scripts/compute_density_power.py --region tampen_box_a --start 2020-01-01 --end 2020-01-31 --height 100
python scripts/compute_density_power.py --region utsira_nord_box_b --start 2020-01-01 --end 2020-01-31 --height 100
python scripts/compute_density_power.py --region sn2_box_c --start 2020-01-01 --end 2020-01-31 --height 100
```

***

### Energy and Capacity Factor Report (For All Regions)

```bash
python scripts/report_energy_cf.py --start 2020-01-01 --end 2020-01-31 --height 100
```

***

### Static Region Map and Dashboard

**Static region map:**

```bash
python scripts/map_all_regions.py
```

**Dashboard:**

```bash
streamlit run njord_dashboard.py
```

---

## Outputs

- `viz/quicklooks/<region>/` wind snapshots
- `viz/layout/` turbine layout plots
- `viz/inflow/` inflow diagnostics
- `viz/power/` baseline and density plots
- `viz/wake/` wake comparisons and direction plots
- `viz/reports/` CF comparisons
- `viz/dashboard/` cached legend assets

---

## Notes

- ERA5 is coarse. This is a mesoscale replay twin, not microscale turbine-level truth.

- Wake model is deliberately lightweight. It’s there to show structure and plausible effects, not to compete with engineering-grade wake software.

- Density correction is a real effect and improves credibility with very little complexity.

---

## Future Additions

- Per-turbine wake deficit reporting (spatial heatmap of wake losses)

- Longer windows (3–12 months) with caching & storage tuning

- OpenFAST turbine-level simulation

---

## License
MIT License. See `LICENSE` for details.
