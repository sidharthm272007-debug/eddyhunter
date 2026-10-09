"""End-to-end real-data pipeline: detect (vorticity-primary) -> track -> forecast.

Run from the eddyhunter project root:
    python scripts/run_real_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import xarray as xr
import pandas as pd

from eddyhunter.detection.pipeline import DetectionConfig, detect_eddies_single_step, eddies_to_dataframe
from eddyhunter.tracking.associate import track_eddies, tracks_to_dataframe, AssociationConfig
from eddyhunter.forecast.predict import build_track_b_output, track_b_to_dataframe

DS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "real_scene_fine.nc"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def main():
    ds = xr.open_dataset(DS_PATH)
    lats = ds["lat"].values
    lons = ds["lon"].values

    config = DetectionConfig(
        source_field="vorticity",
        core_neighborhood_cells=15,
        ring_box_radius_cells=25,
        ring_band_low_frac=0.15,
        ring_band_high_frac=0.75,
        ring_min_points=10,
        min_persistence_m=3000.0,
        min_area_km2=20.0,
    )

    eddies_by_time = {}
    all_eddies = []
    for t_idx in range(ds.sizes["time"]):
        u = ds["u"].isel(time=t_idx).values
        v = ds["v"].isel(time=t_idx).values
        ssha = ds["ssha"].isel(time=t_idx).values
        eddies = detect_eddies_single_step(u, v, ssha, lats, lons, t_idx, config)
        validated = [e for e in eddies if e.validated]
        eddies_by_time[t_idx] = validated
        all_eddies.extend(validated)
        print(f"t{t_idx} ({str(ds['time'].values[t_idx])[:10]}): {len(validated)} validated eddies")

    eddies_to_dataframe(all_eddies).to_csv(OUT_DIR / "real_detected_eddies.csv", index=False)

    # real-world time gap between SWOT/OSCAR-paired snapshots varies (not a
    # fixed 24h cadence -- see diagnostics in the dataset attrs), so use each
    # step's actual elapsed time rather than assuming a constant dt.
    times = pd.to_datetime(ds["time"].values)
    assoc_cfg = AssociationConfig(max_gate_km=150.0)  # wider gate: real steps are 1-4 days apart, not 1
    tracks = track_eddies(eddies_by_time, assoc_cfg)
    print(f"\n{len(tracks)} tracks total, {sum(1 for t in tracks if t.length > 1)} with length > 1")

    tracks_to_dataframe(tracks).to_csv(OUT_DIR / "real_tracks.csv", index=False)

    multi_point_tracks = [t for t in tracks if t.length > 1]
    if multi_point_tracks:
        avg_dt_hours = float(pd.Series(times).diff().dropna().dt.total_seconds().mean() / 3600.0)
        records = build_track_b_output(multi_point_tracks, dt_hours=avg_dt_hours)
        track_b_to_dataframe(records).to_csv(OUT_DIR / "real_track_b_output.csv", index=False)
        print(f"Forecast records written for {len(multi_point_tracks)} multi-point tracks (avg dt={avg_dt_hours:.1f}h)")
    else:
        print("No multi-point tracks -- nothing to forecast yet.")

    for t in sorted(tracks, key=lambda t: -t.length)[:10]:
        pts = [(p.time_index, round(p.eddy.centroid_lat, 2), round(p.eddy.centroid_lon, 2)) for p in t.points]
        print(f"{t.track_id} ({t.rotation}, len={t.length}): {pts}")


if __name__ == "__main__":
    main()
