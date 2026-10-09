# EddyHunter

SWOT sea-surface-height + OSCAR surface-current data → alpha-complex / persistent-H1
mesoscale eddy detection → physical validation → tracking → short-term forecasting →
PPO navigation agent, with a live Streamlit dashboard.

## Quickstart

```bash
pip install -e .
python scripts/run_full_pipeline.py   # synthetic scene -> detection -> tracking -> forecast -> PPO eval -> outputs/
streamlit run app.py                  # interactive dashboard
```

## Layout

- `src/eddyhunter/ingest/` — synthetic rotating-vortex generator (`synthetic.py`), real
  SWOT/OSCAR ingestion via NASA Earthdata (`earthdata.py`), cleaning/regridding (`clean.py`)
- `src/eddyhunter/detection/` — derived fields (vorticity, Okubo-Weiss), ring point-cloud
  construction, alpha-complex + persistent H1 (gudhi), eddy polygons, physical validation
- `src/eddyhunter/tracking/` — Hungarian-algorithm track association across time steps
- `src/eddyhunter/forecast/` — motion estimation + next-position forecasting ("Track B")
- `src/eddyhunter/rl/` — PPO navigation env (procedural + real-track replay modes),
  random/direct-navigation baselines, training/evaluation
- `src/eddyhunter/eval/` — detection/tracking/forecast metrics vs synthetic ground truth
- `app.py` — Streamlit dashboard
- `scripts/run_full_pipeline.py` — end-to-end run + evaluation report

## Real data

Set `EARTHDATA_USERNAME` and `EARTHDATA_TOKEN` (from urs.earthdata.nasa.gov) and use
`eddyhunter.ingest.earthdata` to pull real SWOT L2 SSH anomaly and OSCAR velocity granules
for your region/time window; everything downstream is unchanged.
