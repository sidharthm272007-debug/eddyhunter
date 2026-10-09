"""End-to-end run: synthetic data -> detection -> tracking -> forecasting
-> evaluation against ground truth -> PPO vs baselines, on a richer
synthetic scene than the quick smoke tests used during development.
Saves all intermediate tables + a JSON report to outputs/.
"""
from __future__ import annotations

import json
import warnings
import sys
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from stable_baselines3 import PPO

from eddyhunter.ingest.synthetic import generate_synthetic_dataset, SyntheticEddySpec
from eddyhunter.detection.pipeline import detect_eddies_all_steps, eddies_to_dataframe, DetectionConfig
from eddyhunter.tracking.associate import track_eddies, tracks_to_dataframe, AssociationConfig
from eddyhunter.forecast.predict import build_track_b_output, track_b_to_dataframe
from eddyhunter.eval.metrics import (
    evaluate_detection_against_truth, evaluate_track_continuity, evaluate_forecast_error,
)
from eddyhunter.rl.adapters import track_b_records_to_replay_steps
from eddyhunter.rl.env import EddyChaseConfig
from eddyhunter.rl.train import evaluate_all

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)


def richer_demo_eddies():
    return [
        SyntheticEddySpec(name="cyclone_A", lat0=25.0, lon0=-54.0, radius_km=60.0,
                           v_max=0.55, sign=+1, drift_u=-0.03, drift_v=0.01),
        SyntheticEddySpec(name="anticyclone_B", lat0=23.5, lon0=-57.0, radius_km=75.0,
                           v_max=0.65, sign=-1, drift_u=-0.025, drift_v=-0.005),
        SyntheticEddySpec(name="cyclone_D", lat0=27.5, lon0=-58.5, radius_km=50.0,
                           v_max=0.45, sign=+1, drift_u=-0.02, drift_v=-0.015, birth_step=2),
        SyntheticEddySpec(name="anticyclone_E", lat0=21.5, lon0=-53.0, radius_km=65.0,
                           v_max=0.5, sign=-1, drift_u=0.02, drift_v=0.015, birth_step=0),
        SyntheticEddySpec(name="weak_transient_C", lat0=27.0, lon0=-52.0, radius_km=25.0,
                           v_max=0.12, sign=+1, drift_u=-0.01, drift_v=0.0,
                           lifetime_steps=4, birth_step=3),
    ]


def main():
    print("Generating synthetic scene...")
    ds, truth = generate_synthetic_dataset(
        richer_demo_eddies(), n_steps=30, lat_range=(18, 32), lon_range=(-62, -48),
        grid_step_deg=0.1,
    )
    ds.to_netcdf(OUT / "synthetic_scene.nc")
    truth.to_csv(OUT / "synthetic_truth.csv", index=False)

    print("Running detection across all time steps...")
    dcfg = DetectionConfig(min_persistence_m=8000.0)
    all_eddies = detect_eddies_all_steps(ds, dcfg)
    eddies_df = eddies_to_dataframe(all_eddies)
    eddies_df.to_csv(OUT / "detected_eddies.csv", index=False)
    print(f"  {len(all_eddies)} candidates, {eddies_df.validated.sum()} validated")

    validated = [e for e in all_eddies if e.validated]
    by_time = {}
    for e in validated:
        by_time.setdefault(e.time_index, []).append(e)

    print("Tracking...")
    tracks = track_eddies(by_time, AssociationConfig())
    tracks_df = tracks_to_dataframe(tracks)
    tracks_df.to_csv(OUT / "tracks.csv", index=False)
    print(f"  {len(tracks)} tracks, lengths={sorted([t.length for t in tracks], reverse=True)}")

    print("Forecasting (Track B output)...")
    records = build_track_b_output(tracks, dt_hours=24.0)
    track_b_df = track_b_to_dataframe(records)
    track_b_df.to_csv(OUT / "track_b_output.csv", index=False)

    print("Scoring against ground truth...")
    detection_metrics = evaluate_detection_against_truth(eddies_df, truth, max_match_km=60.0)
    continuity_metrics = evaluate_track_continuity(tracks, n_time_steps=ds.sizes["time"])
    forecast_metrics = evaluate_forecast_error(records)

    print("Training/evaluating PPO navigation...")
    long_tracks = [t for t in tracks if t.length >= 6]
    replay_sequences = []
    for tr in long_tracks:
        recs = build_track_b_output([tr], dt_hours=24.0)
        replay_sequences.append(track_b_records_to_replay_steps(recs, dt_hours=24.0))

    rl_cfg = EddyChaseConfig(max_steps=40)
    model_path = OUT / "rl" / "ppo_eddychase.zip"
    if model_path.exists():
        model = PPO.load(str(model_path).replace(".zip", ""))
        rl_results = evaluate_all(model, rl_cfg, n_episodes=200, replay_sequences=replay_sequences)
    else:
        rl_results = {"note": "no trained PPO model found at outputs/rl/ppo_eddychase.zip"}

    report = {
        "scene": {"n_time_steps": int(ds.sizes["time"]), "n_eddy_specs": len(richer_demo_eddies())},
        "detection": detection_metrics,
        "track_continuity": continuity_metrics,
        "forecast": forecast_metrics,
        "rl": rl_results,
    }
    with open(OUT / "full_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
