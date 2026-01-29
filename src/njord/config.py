from dataclasses import dataclass

@dataclass(frozen=True)
class StudyRegion:
    key: str
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

#  Time window for ERA5 data retreival
DEFAULT_START = "2020-01-01"
DEFAULT_END   = "2020-01-03"

# ERA5 regions bounds (WGS84 lat/lon)
REGIONS = {
    # Box A: Hywind Tampen (a notable Norwegian offshore windfarm location)
    "tampen_box_a": StudyRegion(
        key="tampen_box_a",
        name="Hywind Tampen (Box A)",
        lat_min=60.8,
        lat_max=61.9,
        lon_min=2.0,
        lon_max=3.6,
    ),

    # Box B: Utsira Nord area (This is a broader area with a larger DT layout)
    "utsira_nord_box_b": StudyRegion(
        key="utsira_nord_box_b",
        name="Utsira Nord (Box B)",
        lat_min=58.3,
        lat_max=59.7,
        lon_min=4.2,
        lon_max=6.8,
    ),

    # Box C: Sørlige Nordsjø II area (southern North Sea vs Norwegian Sea contract)
    "sn2_box_c": StudyRegion(
        key="sn2_box_c",
        name="Sørlige Nordsjø II (Box C)",
        lat_min=56.2,
        lat_max=57.4,
        lon_min=4.2,
        lon_max=5.8,
    ),
}


ACTIVE_REGION_KEY = "tampen_box_a"
ACTIVE_REGION = REGIONS[ACTIVE_REGION_KEY]