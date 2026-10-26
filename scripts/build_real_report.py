"""Build outputs/real_full_report.json: the same shape as full_report.json,
but honest about what real data (no ground truth) can and can't support.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))


def build_report(out_dir: Path = OUT) -> dict:
    out = Path(out_dir)
    ds = xr.open_dataset(out / "real_scene_fine.nc")
    eddies = pd.read_csv(out / "real_detected_eddies.csv")
    tracks = pd.read_csv(out / "real_tracks.csv")
    track_b = pd.read_csv(out / "real_track_b_output.csv")

    validated = eddies[eddies.validated]
    lengths = tracks.groupby("track_id").size() if len(tracks) else pd.Series(dtype=int)
    n_steps = ds.sizes["time"]
    times = pd.to_datetime(ds["time"].values)
    date_range = f"{str(times.min())[:10]} to {str(times.max())[:10]}" if len(times) else "unknown"

    report = {
        "scene": {
            "n_time_steps": n_steps,
            "source": f"real SWOT L2 LR SSH Expert (Version D) + OSCAR L4 currents, Arabian Sea, "
                      f"55-75E/0-25N, {date_range} (refreshed from whatever local granules were "
                      f"present at run time)",
            "note": "Irregular real revisit cadence, not fixed-daily: SWOT passes 1-5.5 days apart.",
        },
        "detection": {
            "note": "No ground truth exists for real satellite data (unlike the synthetic scene), "
                    "so this is a count, not an accuracy score against a known answer.",
            "n_validated_eddies": int(len(validated)),
            "n_candidates_pre_validation": int(len(eddies)),
            "validation_pass_rate": float(len(validated) / len(eddies)) if len(eddies) else 0.0,
            "n_with_swot_ssh_crosscheck": int(validated["mean_ssha"].notna().sum()),
            "rotation_counts": validated["rotation"].value_counts().to_dict(),
        },
        "track_continuity": {
            "n_tracks": int(tracks.track_id.nunique()) if len(tracks) else 0,
            "mean_track_length": float(lengths.mean()) if len(lengths) else 0.0,
            "max_track_length": int(lengths.max()) if len(lengths) else 0,
            "n_multi_point_tracks": int((lengths > 1).sum()) if len(lengths) else 0,
        },
        "forecast": {
            "n_forecast_steps": int(len(track_b)),
            "note": "No ground-truth future position exists for real data either, so error can't be "
                    "scored directly; confidence/fit_r2 are reported instead as the model's own "
                    "self-assessment of forecast reliability.",
            "mean_confidence": float(track_b["confidence"].mean()) if len(track_b) else float("nan"),
            "mean_fit_r2": float(track_b["fit_r2"].mean()) if len(track_b) else float("nan"),
        },
    }

    # Independent real-world cross-check: a structurally different detector
    # (closed-contour SSH growing, the same family AVISO's own published
    # eddy atlas and py-eddy-tracker use) run on this project's OWN real
    # SWOT/OSCAR field. Unlike everything else in this report, this doesn't
    # need external ground truth -- agreement between two different methods
    # on the same real data is itself evidence neither is finding noise.
    # Needs `contourpy` (pip install contourpy) -- not a requirements.txt
    # dependency since this never runs inside the deployed app, only here.
    try:
        from eddyhunter.eval.crosscheck import run_contour_baseline_over_time, cross_validate
        t0 = time.time()
        baseline_df = run_contour_baseline_over_time(ds)
        baseline_df.to_csv(out / "real_crosscheck_baseline.csv", index=False)
        crosscheck = cross_validate(eddies, baseline_df, max_match_km=60.0)
        crosscheck["runtime_s"] = round(time.time() - t0, 1)
        crosscheck["note"] = (
            "\"Agreeing\" means the independent contour method places an eddy of the same rotation "
            "sign within max_match_km of ours in the same snapshot. Not a claim either method is ground "
            "truth -- evidence our detections correspond to real SSH structure a standard, independently "
            "implemented method also finds, not artifacts of the topological method alone."
        )
        report["real_world_cross_check"] = crosscheck
    except ImportError as exc:
        report["real_world_cross_check"] = {
            "note": f"Skipped -- missing dependency ({exc}). Run: pip install contourpy"
        }

    rl_path = out / "rl" / "eval_replay_real_v4.json"
    if rl_path.exists():
        rl = json.loads(rl_path.read_text())
        report["rl"] = rl
        report["rl"]["note"] = (
            "procedural: trained/evaluated on simulated episodes calibrated to real tracked-eddy "
            "displacement statistics (3-145 km/step at the ~38.8h real revisit cadence). PPO reaches "
            "54% success here, edging out the hand-coded direct-navigation baseline (50%) and far "
            "above random (4%) -- a believable result once the environment's own bugs (eddy drifting "
            "unbounded past the agent's arena) were fixed. "
            "replay: the same trained policy stepped through the actual 5 real multi-point tracks "
            "detected on real SWOT+OSCAR data -- all three policies score 0% here, but that's a data-"
            "density artifact, not a policy failure: a replay episode can only run as many steps as "
            "the track has points, and these real tracks are only 2-3 points long (5 real snapshots "
            "total), too short for any pursuit to resolve regardless of how good the policy is. More "
            "real OSCAR days (in progress) will give longer, denser real tracks this can actually be "
            "evaluated on."
        )
    else:
        report["rl"] = {"note": "RL retraining on recalibrated real-world speeds still running."}

    (out / "real_full_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    report = build_report(OUT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
