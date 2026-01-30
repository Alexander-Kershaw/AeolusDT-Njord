from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REGION = "tampen_box_a"

P_RATED_MW = 5.0
V_CI = 3.0
V_RATED = 12.0
V_CO = 25.0

def latest_inflow_parquet(region_key: str) -> Path:
    d = Path("data_lake/gold/inflow") / region_key
    files = sorted(d.glob("inflow_u10v10_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]

def power_curve_mw(v):
    v = np.asarray(v, dtype=float)
    p = np.zeros_like(v)
    m1 = (v >= V_CI) & (v < V_RATED)
    m2 = (v >= V_RATED) & (v < V_CO)
    p[m1] = P_RATED_MW * ((v[m1]**3 - V_CI**3) / (V_RATED**3 - V_CI**3))
    p[m2] = P_RATED_MW
    return p

inflow_path = latest_inflow_parquet(REGION)
df = pd.read_parquet(inflow_path)
df["time"] = pd.to_datetime(df["time"])

df["p_turbine_mw"] = power_curve_mw(df["speed10"].values)
farm = df.groupby("time", as_index=False)["p_turbine_mw"].sum().rename(columns={"p_turbine_mw":"p_farm_mw"})

out_dir = Path("data_lake/gold/power") / REGION
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"baseline_power_{inflow_path.stem}.parquet"
farm.to_parquet(out_path, index=False)

Path("viz/power").mkdir(parents=True, exist_ok=True)
plot_path = Path("viz/power") / f"{REGION}_baseline_farm_power_30d.png"

plt.figure()
plt.plot(farm["time"], farm["p_farm_mw"])
plt.title(f"Baseline Farm Power (no wakes) | {REGION} | 30 days")
plt.xlabel("Time")
plt.ylabel("Farm power (MW)")
plt.savefig(plot_path, dpi=180, bbox_inches="tight")
plt.close()

print("Used inflow:", inflow_path.name)
print("Wrote power:", out_path)
print("Plot:", plot_path)
print("Farm power stats (MW):", float(farm["p_farm_mw"].min()), float(farm["p_farm_mw"].mean()), float(farm["p_farm_mw"].max()))