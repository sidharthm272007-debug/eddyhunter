"""EddyHunter dashboard: SWOT/OSCAR-style eddy detection (persistent H1
homology), tracking, forecasting, and a PPO navigation agent -- all driven
by the real pipeline output, not mocked data.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
from shapely import wkt as shapely_wkt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Web-deployment mode: on Streamlit Community Cloud (or anywhere else secrets
# are configured), EARTHDATA_USERNAME/PASSWORD live in st.secrets rather than
# a local folder the user downloaded into. Copy them into the environment
# (where earthaccess/earthdata_live.py expect to find them) and switch the
# sidebar from "point me at your local folders" to "auto-synced from NASA".
# Never typed in chat, never logged, never hardcoded -- set once by the app
# owner in Streamlit Cloud's own Secrets UI. Locally, with no secrets.toml,
# st.secrets is simply empty and this is a no-op.
try:
    if st.secrets.get("EARTHDATA_USERNAME") and st.secrets.get("EARTHDATA_PASSWORD"):
        os.environ["EARTHDATA_USERNAME"] = st.secrets["EARTHDATA_USERNAME"]
        os.environ["EARTHDATA_PASSWORD"] = st.secrets["EARTHDATA_PASSWORD"]
except Exception:
    pass
AUTO_SYNC = bool(os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"))
SYNC_INTERVAL_HOURS = 20  # roughly matches OSCAR's own once-a-day update cadence

from eddyhunter.rl.env import EddyChaseEnv, EddyChaseConfig, ReplayStep  # noqa: E402
from eddyhunter.rl.baselines import random_policy, direct_navigation_policy  # noqa: E402
from eddyhunter.rl.adapters import track_b_records_to_replay_steps  # noqa: E402
from eddyhunter.forecast.predict import TrackBRecord  # noqa: E402

OUT = ROOT / "outputs"

st.set_page_config(page_title="EddyHunter", layout="wide", page_icon="🌊")

# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap" rel="stylesheet">
    <style>
    /* Modern dark operational dashboard: dark navy surfaces, a teal/cyan
       accent (reads as "ocean data product"), rounded cards with a soft
       glow border instead of flat academic rules -- built to look like a
       tool someone would actually run day to day, not a research figure. */
    :root {
        --bg: #0b0f14;
        --surface: #121821;
        --surface-2: #171f2a;
        --border: #232d3a;
        --accent: #2dd4bf;
        --accent-soft: rgba(45, 212, 191, 0.12);
        --text: #e6edf3;
        --text-dim: #8b98a5;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp { background-color: var(--bg); }
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1360px;}
    h1, h2, h3, h4 {
        font-family: 'Manrope', 'Inter', -apple-system, sans-serif;
        color: var(--text);
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    h1 { font-size: 2.2rem; margin-bottom: 0.3rem; }
    h3 { font-size: 1.1rem; margin-top: 1.6rem; color: var(--text); }
    p, li, label, span, div { font-family: 'Inter', -apple-system, sans-serif; }
    p, li { color: var(--text-dim); line-height: 1.6; }
    [data-testid="stCaptionContainer"] { color: var(--text-dim) !important; }
    /* KPI tiles: dark card, rounded corners, accent left-border + glow --
       the "product dashboard" stat tile instead of a bare number. */
    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 0 0 1px rgba(45, 212, 191, 0.03), 0 4px 14px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetricValue"] {
        color: var(--text);
        font-family: 'Manrope', sans-serif;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--text-dim);
        font-weight: 600;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Tabs: filled pill for the active tab in the accent color -- a clear
       "you are here" state, like a real product's nav rather than a
       journal section list. */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid var(--border);
        padding-bottom: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        color: var(--text-dim);
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-soft) !important;
        color: var(--accent) !important;
    }
    .stTabs [aria-selected="true"] p { color: var(--accent) !important; font-weight: 700; }
    section[data-testid="stSidebar"] { background-color: var(--surface); border-right: 1px solid var(--border); }
    section[data-testid="stSidebar"] * { color: var(--text-dim); }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: var(--text); font-size: 1rem; }
    div[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }
    div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
    hr { border-color: var(--border); }
    .stButton button[kind="primary"] {
        background-color: var(--accent);
        color: #06201c;
        border: none;
        font-weight: 700;
        border-radius: 8px;
    }
    .stButton button[kind="primary"]:hover { background-color: #5eead4; color: #06201c; }
    div[data-testid="stAlert"] { border-radius: 10px; border: 1px solid var(--border); }
    </style>
    """,
    unsafe_allow_html=True,
)

# The map view used to be built on go.Scattermap tile basemaps
# (OpenStreetMap / CARTO). Dropped in favor of go.Scattergeo below: tile
# basemaps render place-name labels in whatever language the tile server
# defaults to for that region (Arabic for the Arabian Sea coastline), and
# the zoom-to-bbox math was also wrong, leaving the view zoomed out to
# roughly a quarter of the globe instead of the actual study area.
# go.Scattergeo draws coastlines/country borders with no city/place labels
# at all, fetches no tiles, and is scoped exactly to the data's own bbox.


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

@st.cache_data
def load_outputs(source: str):
    import xarray as xr
    if source == "real":
        ds = xr.load_dataset(OUT / "real_scene_fine.nc")  # load fully into memory & close the file, so a later live refresh can overwrite it
        eddies_df = pd.read_csv(OUT / "real_detected_eddies.csv")
        tracks_df = pd.read_csv(OUT / "real_tracks.csv")
        track_b_df = pd.read_csv(OUT / "real_track_b_output.csv")
        report = json.load(open(OUT / "real_full_report.json"))
    else:
        ds = xr.load_dataset(OUT / "synthetic_scene.nc")
        eddies_df = pd.read_csv(OUT / "detected_eddies.csv")
        tracks_df = pd.read_csv(OUT / "tracks.csv")
        track_b_df = pd.read_csv(OUT / "track_b_output.csv")
        report = json.load(open(OUT / "full_report.json"))
    return ds, eddies_df, tracks_df, track_b_df, report


_LAST_SYNC_FILE = OUT / ".last_nasa_sync"


def _last_sync_time():
    if _LAST_SYNC_FILE.exists():
        try:
            return pd.Timestamp(_LAST_SYNC_FILE.read_text().strip())
        except Exception:
            return None
    return None


def run_nasa_sync(force: bool = False):
    """Web-deployment auto-refresh: if the last NASA Earthdata sync is older
    than SYNC_INTERVAL_HOURS (or there's never been one, or `force` is set),
    pull whatever new OSCAR/SWOT granules exist and merge them into the
    existing real scene. Returns (ran: bool, result_or_None) -- `ran=False`
    means the cache was still fresh and nothing happened, which is the
    common case on most page loads, not an error.
    """
    last = _last_sync_time()
    now = pd.Timestamp.now("UTC")
    if not force and last is not None and (now - last) < pd.Timedelta(hours=SYNC_INTERVAL_HOURS):
        return False, None
    from eddyhunter.live import sync_from_nasa_and_refresh
    result = sync_from_nasa_and_refresh(out_dir=str(OUT), days_back=25)
    if result.ok:
        _LAST_SYNC_FILE.write_text(now.isoformat())
        load_outputs.clear()
    return True, result


@st.cache_resource
def load_ppo_model(source: str):
    from stable_baselines3 import PPO
    # The real-data-calibrated policy (v4) was trained on episode speeds
    # matched to real tracked-eddy displacement (see rl/env.py's
    # EddyChaseConfig docstring); the original policy was trained on an
    # idealized 0-3 km/step assumption and is kept only for the synthetic
    # tab, where that assumption is what the synthetic scene itself uses.
    name = "ppo_eddychase_v5" if source == "real" else "ppo_eddychase"
    path = OUT / "rl" / f"{name}.zip"
    if not path.exists():
        path = OUT / "rl" / "ppo_eddychase.zip"
    return PPO.load(str(path))


with st.sidebar:
    st.header("Data source")
    has_real = (OUT / "real_full_report.json").exists()
    source_label = st.radio(
        "Run this dashboard on:",
        ["Real SWOT + OSCAR (Arabian Sea)", "Synthetic (ground-truth validated)"] if has_real
        else ["Synthetic (ground-truth validated)"],
    )
    source = "real" if source_label.startswith("Real") else "synthetic"
    if source == "real":
        st.caption(
            "Detection runs on OSCAR-derived vorticity (full areal coverage); SWOT SSH is kept as a "
            "sparse independent cross-check, not the primary signal -- a single SWOT swath pass is "
            "too narrow (~120km) to ever close a full SSH ring. See the Evaluation tab for why."
        )

        if AUTO_SYNC:
            with st.expander("🛰️ NASA Earthdata auto-sync", expanded=False):
                st.caption(
                    "This deployment pulls new OSCAR/SWOT granules directly from NASA Earthdata itself -- "
                    "no local folders needed. SWOT/OSCAR aren't a continuous stream (SWOT revisits a spot "
                    "every 1-21 days, OSCAR updates once a day), so 'live' here means this checks for "
                    f"anything new about every {SYNC_INTERVAL_HOURS}h, merges it into the existing real "
                    "history (nothing already detected is thrown away), and reruns detection/tracking/"
                    "forecasting on the combined series."
                )
                last = _last_sync_time()
                st.caption(f"Last synced: {last.strftime('%Y-%m-%d %H:%M UTC') if last is not None else 'never yet'}")
                if st.button("🔄 Force sync now", type="primary"):
                    with st.spinner("Checking NASA Earthdata for new granules and re-running the pipeline..."):
                        _, result = run_nasa_sync(force=True)
                    if result.ok:
                        st.success(result.message)
                    else:
                        st.warning(result.message)
        else:
            with st.expander("🔄 Live refresh (local folders)", expanded=False):
                st.caption(
                    "No NASA Earthdata credentials are configured for this deployment, so refresh reads "
                    "local SWOT/OSCAR folders instead -- the same mechanism used when running this app on "
                    "your own machine. SWOT/OSCAR aren't a continuous stream -- SWOT revisits a spot every "
                    "1-21 days, OSCAR updates once a day -- so there's nothing to show between revisits. "
                    "This button re-scans the folders below for whatever .nc files are actually sitting "
                    "there right now and reruns detection/tracking/forecasting on them fresh."
                )
                def _default_dir(*candidates: str) -> str:
                    for c in candidates:
                        if Path(c).is_dir():
                            return c
                    return candidates[0]

                swot_dir = st.text_input("SWOT folder", value=_default_dir(
                    r"C:\Users\Lenovo\Downloads\swot", "/mnt/user-data/uploads/swot", str(ROOT / "data" / "swot"),
                ))
                oscar_dir = st.text_input("OSCAR folder", value=_default_dir(
                    r"C:\Users\Lenovo\Downloads\oscar", "/mnt/user-data/uploads/oscar", str(ROOT / "data" / "oscar"),
                ))
                if st.button("🔄 Scan folders & refresh now", type="primary"):
                    from eddyhunter.live import refresh_real_pipeline
                    with st.spinner("Re-ingesting real SWOT+OSCAR files and rerunning detection/tracking/forecast..."):
                        result = refresh_real_pipeline(swot_dir=swot_dir, oscar_dir=oscar_dir, out_dir=str(OUT))
                    if result.ok:
                        st.success(
                            f"{result.message} {result.n_validated_eddies} validated eddies, "
                            f"{result.n_multi_point_tracks} multi-point tracks across {result.n_time_steps} snapshots."
                        )
                        # No st.rerun() here: Streamlit already reruns this whole script
                        # top-to-bottom on every interaction, and load_outputs(source)
                        # below hasn't executed yet in THIS run -- clearing the cache now
                        # is enough for it to pick up the fresh files further down in the
                        # same pass. Calling st.rerun() would restart the script
                        # immediately and wipe this success message before it ever
                        # rendered.
                        load_outputs.clear()
                    else:
                        st.error(result.message)
    else:
        st.info("Synthetic rotating-vortex scene: every number here is checked against a known answer.")

ok = (OUT / "full_report.json").exists()
if not ok:
    st.error("No pipeline outputs found yet. Run `python scripts/run_full_pipeline.py` first.")
    st.stop()

if source == "real" and AUTO_SYNC:
    # NOT auto-run on page load anymore. Every page load used to check (and,
    # once past SYNC_INTERVAL_HOURS, actually perform) a full NASA sync +
    # detect/track/forecast rerun before the page could render at all -- on
    # a cold container (e.g. right after this free-tier app wakes from sleep
    # or gets redeployed, when outputs/.last_nasa_sync doesn't exist yet
    # either) that stacked a full sync+pipeline run on top of an already
    # heavy cold import of torch/stable-baselines3/gudhi, well within reach
    # of the free tier's ~1GB RAM limit. Syncing is now strictly opt-in via
    # the sidebar's "Force sync now" button -- the page itself only ever
    # reads whatever's already cached, which is cheap and can't OOM on load.
    last = _last_sync_time()
    if last is None:
        st.caption("🛰️ NASA auto-sync hasn't run yet this deployment -- click **Force sync now** in the sidebar to pull real data.")
    else:
        st.caption(f"🛰️ Showing cached real data (last NASA sync: {last.strftime('%Y-%m-%d %H:%M UTC')}). "
                   "Click **Force sync now** in the sidebar to check for anything newer.")

if source == "real" and not (OUT / "real_scene_fine.nc").exists():
    # The regridded real scene (hundreds of MB) isn't shipped in the project
    # download -- on your own machine it's rebuilt from your SWOT/OSCAR
    # folders (live-refresh button); on a deployment with NASA credentials
    # configured, the auto-sync above just attempted to build it directly.
    if AUTO_SYNC:
        st.warning(
            "No real data is cached yet and the initial NASA Earthdata sync hasn't produced a scene "
            "(see the message above). Click **🔄 Force sync now** in the sidebar, or switch to the "
            "Synthetic source in the meantime."
        )
    else:
        st.warning(
            "The real-data scene hasn't been built on this machine yet. Open the sidebar, set your "
            "SWOT/OSCAR folder paths, and click **🔄 Scan folders & refresh now** once -- after that "
            "this tab will have real data to show, same as every refresh after."
        )
    st.stop()

ds, eddies_df, tracks_df, track_b_df, report = load_outputs(source)
model = load_ppo_model(source)

st.title("EddyHunter")
st.caption(
    "SWOT sea-surface-height + OSCAR surface currents → alpha-complex / persistent-H1 "
    "eddy detection → tracking & forecasting → PPO navigation agent"
)

tab_overview, tab_detect, tab_rl, tab_eval = st.tabs(
    ["Overview", "Detection & Tracking", "PPO Navigation", "Evaluation Report"]
)

# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
with tab_overview:
    metric_box = st.container(border=True)
    c1, c2, c3, c4 = metric_box.columns(4)
    if source == "real":
        c1.metric("Validated eddies", report["detection"]["n_validated_eddies"])
        c2.metric("SWOT SSH cross-checks", report["detection"]["n_with_swot_ssh_crosscheck"])
        c3.metric("Mean forecast confidence", f"{report['forecast']['mean_confidence']*100:.0f}%")
        ppo_sr = report["rl"].get("procedural", {}).get("ppo", {}).get("success_rate")
        c4.metric("PPO success (procedural)", f"{ppo_sr*100:.0f}%" if ppo_sr is not None else "running…")
        st.markdown(
            f"""
Running on **real SWOT L2 LR SSH Expert + OSCAR L4 currents** — {report['scene']['n_time_steps']} real
time steps over the Arabian Sea. Real satellite data has no ground-truth answer key, so these are
counts and the model's own confidence, not accuracy scores against a known truth — that honest
distinction (and what *can* still be validated: physical plausibility checks, track continuity, a
sparse SWOT SSH cross-check) is in the Evaluation tab.
            """
        )
        st.caption(report["scene"]["source"])
    else:
        c1.metric("Detection recall", f"{report['detection']['recall']*100:.1f}%")
        c2.metric("Centroid error", f"{report['detection']['mean_centroid_error_km']:.1f} km")
        c3.metric("Forecast error", f"{report['forecast']['mean_forecast_error_km']:.1f} km")
        ppo_sr = report["rl"]["procedural"]["ppo"]["success_rate"]
        c4.metric("PPO success rate", f"{ppo_sr*100:.0f}%")
        st.markdown(
            """
Running on a **synthetic rotating-vortex scene** (5 eddies, 30 time steps) so every number above can
be checked against ground truth. Switch to the real SWOT+OSCAR source in the sidebar to see this same
pipeline running on actual satellite data.
            """
        )

    st.subheader("Pipeline")
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            bgcolor="transparent";
            node [shape=box, style="rounded,filled", fillcolor="#171f2a", color="#2dd4bf", fontcolor="#e6edf3", fontname="Helvetica"];
            edge [color="#8b98a5"];
            SWOT [label="SWOT SSH"]; OSCAR [label="OSCAR u,v"];
            clean [label="Clean + regrid"];
            topo [label="Alpha complex\\n+ persistent H1"];
            poly [label="Eddy polygons\\n+ validation"];
            track [label="Tracking\\n(Hungarian)"];
            forecast [label="Forecast\\n(Track B)"];
            ppo [label="PPO agent"];
            SWOT -> clean; OSCAR -> clean; clean -> topo -> poly -> track -> forecast -> ppo;
        }
        """
    )

with st.sidebar:
    st.divider()
    st.caption(
        "Real data is loaded from local SWOT/OSCAR files via "
        "eddyhunter.ingest.real_local (manually downloaded through NASA Earthdata Search, since "
        "this deployment's network can't reach podaac.jpl.nasa.gov directly). Drop newer files in "
        "and re-run scripts/run_real_pipeline.py to refresh."
    )


# ----------------------------------------------------------------------------
# Detection & tracking map
# ----------------------------------------------------------------------------
with tab_detect:
    n_t = ds.sizes["time"]
    ctrl1, ctrl2 = st.columns([3, 1])
    with ctrl1:
        t = st.slider("Time step", 0, n_t - 1, 0, key="detect_t")
    with ctrl2:
        view = st.radio("View", ["Map", "Grid"], horizontal=True, key="detect_view")

    ssha_t = ds["ssha"].isel(time=t).values
    u_t = ds["u"].isel(time=t).values
    v_t = ds["v"].isel(time=t).values
    lats = ds["lat"].values
    lons = ds["lon"].values

    eddies_t = eddies_df[(eddies_df.time_index == t) & (eddies_df.validated)]
    colors = {"cyclonic": "#1f77b4", "anticyclonic": "#d62728"}

    fig = go.Figure()
    is_map = view.startswith("Map")

    if is_map:
        # Background field as a colored point grid -- subsampled so the
        # browser renders smoothly even for the full-resolution 93-day real
        # series (501x401 cells).
        bg_step = max(1, int(np.sqrt((len(lats) * len(lons)) / 6000)))
        lon_grid, lat_grid = np.meshgrid(lons[::bg_step], lats[::bg_step])
        z_sub = ssha_t[::bg_step, ::bg_step]
        finite_bg = np.isfinite(z_sub)
        fig.add_trace(go.Scattergeo(
            lat=lat_grid[finite_bg], lon=lon_grid[finite_bg], mode="markers",
            marker=dict(size=7, color=z_sub[finite_bg], colorscale="RdBu_r", cmid=0,
                        colorbar=dict(title="SSHA (m)"), opacity=0.6),
            name="SSHA", hoverinfo="skip",
        ))

        # velocity field as short directional line segments (one trace, NaN-
        # separated -- the quiver-plot equivalent for a geo subplot)
        step = max(1, len(lons) // 18)
        lon_q, lat_q = np.meshgrid(lons[::step], lats[::step])
        u_sub, v_sub = u_t[::step, ::step], v_t[::step, ::step]
        finite = np.isfinite(u_sub) & np.isfinite(v_sub)
        if finite.any():
            spacing_deg = float(np.mean(np.diff(lons[::step]))) if len(lons[::step]) > 1 else 0.3
            mag = np.sqrt(u_sub**2 + v_sub**2)
            max_mag = np.nanmax(mag[finite]) or 1.0
            arrow_len = 0.8 * spacing_deg * (mag / max_mag)
            lats_l, lons_l = [], []
            for yi in range(u_sub.shape[0]):
                for xi in range(u_sub.shape[1]):
                    if not finite[yi, xi]:
                        continue
                    dlon = (u_sub[yi, xi] / mag[yi, xi]) * arrow_len[yi, xi]
                    dlat = (v_sub[yi, xi] / mag[yi, xi]) * arrow_len[yi, xi]
                    lons_l += [lon_q[yi, xi], lon_q[yi, xi] + dlon, None]
                    lats_l += [lat_q[yi, xi], lat_q[yi, xi] + dlat, None]
            fig.add_trace(go.Scattergeo(
                lat=lats_l, lon=lons_l, mode="lines",
                line=dict(color="rgba(20,20,20,0.55)", width=2),
                name="velocity", hoverinfo="skip",
            ))

        for _, row in eddies_t.iterrows():
            poly = shapely_wkt.loads(row.polygon_wkt)
            xs, ys = poly.exterior.coords.xy
            fig.add_trace(go.Scattergeo(
                lat=list(ys), lon=list(xs), mode="lines", fill="toself",
                line=dict(color=colors.get(row.rotation, "black"), width=2.5),
                fillcolor="rgba(0,0,0,0.10)",
                name=row.rotation, showlegend=False,
                hovertext=f"{row.eddy_id}<br>P={row.persistence_m:.0f}m<br>area={row.area_km2:.0f}km2",
                hoverinfo="text",
            ))

        for track_id, g in tracks_df[tracks_df.time_index <= t].groupby("track_id"):
            g = g.sort_values("time_index")
            if len(g) < 2:
                continue
            rot = g.rotation.iloc[0]
            fig.add_trace(go.Scattergeo(
                lat=g.centroid_lat, lon=g.centroid_lon, mode="lines+markers",
                line=dict(color=colors.get(rot, "gray"), width=1.5),
                marker=dict(size=6), showlegend=False, hoverinfo="skip", opacity=0.8,
            ))

        # Scope the geo subplot tightly to the data's own bbox (with a small
        # padding margin) instead of a tile-map zoom level -- the earlier
        # zoom formula only looked at latitude span and left the view
        # zoomed out across roughly a quarter of the globe. go.Scattergeo
        # also draws coastlines/country borders with no city/place-name
        # labels at all, which is what actually fixes "the map shouldn't be
        # in Arabic": there is no label layer to be in any language.
        lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
        lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
        lat_pad = max(0.5, (lat_max - lat_min) * 0.08)
        lon_pad = max(0.5, (lon_max - lon_min) * 0.08)
        fig.update_layout(
            geo=dict(
                projection_type="mercator",
                lataxis=dict(range=[lat_min - lat_pad, lat_max + lat_pad]),
                lonaxis=dict(range=[lon_min - lon_pad, lon_max + lon_pad]),
                showland=True, landcolor="#eef2f0",
                showocean=True, oceancolor="#dcedf7",
                showcountries=True, countrycolor="#a9c2d4",
                showcoastlines=True, coastlinecolor="#7b9cb3",
                showlakes=False, showrivers=False,
                bgcolor="#ffffff",
            ),
            height=650, margin=dict(l=0, r=0, t=30, b=0),
            title=f"t={t}  ({str(ds['time'].values[t])[:10]})  —  blue=cyclonic, red=anticyclonic",
        )
        st.caption(
            "Coastlines and country borders only — no place-name tiles, so there's no label layer to "
            "render in any language. Dots: SSH anomaly (subsampled). Lines: surface velocity direction. "
            "Filled outlines: validated eddy polygons. Solid trails: accumulated tracks."
        )
    else:
        fig.add_trace(go.Heatmap(
            z=ssha_t, x=lons, y=lats, colorscale="RdBu_r", zmid=0,
            colorbar=dict(title="SSHA (m)"), opacity=0.9,
        ))
        step = max(1, len(lons) // 20)
        lon_grid, lat_grid = np.meshgrid(lons[::step], lats[::step])
        u_sub, v_sub = u_t[::step, ::step], v_t[::step, ::step]
        finite = np.isfinite(u_sub) & np.isfinite(v_sub)
        if finite.any():
            quiver_fig = ff.create_quiver(
                lon_grid[finite], lat_grid[finite], u_sub[finite], v_sub[finite],
                scale=0.6, arrow_scale=0.3, line=dict(color="rgba(20,20,20,0.45)", width=1),
                name="velocity",
            )
            fig.add_traces(quiver_fig.data)

        for _, row in eddies_t.iterrows():
            poly = shapely_wkt.loads(row.polygon_wkt)
            xs, ys = poly.exterior.coords.xy
            fig.add_trace(go.Scatter(
                x=list(xs), y=list(ys), mode="lines", fill="toself",
                line=dict(color=colors.get(row.rotation, "black"), width=2),
                fillcolor="rgba(0,0,0,0)", name=row.rotation,
                showlegend=False,
                hovertext=f"{row.eddy_id}<br>P={row.persistence_m:.0f}m<br>area={row.area_km2:.0f}km2",
            ))

        for track_id, g in tracks_df[tracks_df.time_index <= t].groupby("track_id"):
            g = g.sort_values("time_index")
            if len(g) < 2:
                continue
            rot = g.rotation.iloc[0]
            fig.add_trace(go.Scatter(
                x=g.centroid_lon, y=g.centroid_lat, mode="lines+markers",
                line=dict(color=colors.get(rot, "gray"), width=1, dash="dot"),
                marker=dict(size=4), showlegend=False, hoverinfo="skip", opacity=0.7,
            ))

        fig.update_layout(
            height=650, xaxis_title="Longitude", yaxis_title="Latitude",
            yaxis=dict(scaleanchor="x"), margin=dict(l=10, r=10, t=30, b=10),
            title=f"t={t}  ({str(ds['time'].values[t])[:10]})  —  blue=cyclonic, red=anticyclonic",
        )
        st.caption(
            "Pixel-accurate grid view (no basemap). Shaded background: SSH anomaly. Arrows: surface "
            "velocity (subsampled). Filled outlines: validated eddy polygons. Dotted trails: tracks."
        )

    st.plotly_chart(fig, width='stretch')

    st.subheader("Detected eddies at this time step")
    st.dataframe(
        eddies_t[["eddy_id", "rotation", "centroid_lat", "centroid_lon", "area_km2",
                  "persistence_m", "mean_speed", "mean_vorticity"]].round(4),
        width='stretch',
    )


# ----------------------------------------------------------------------------
# PPO navigation
# ----------------------------------------------------------------------------
with tab_rl:
    st.subheader("Watch the PPO agent chase a forecasted eddy")

    long_tracks = sorted(tracks_df.track_id.unique().tolist())
    track_lengths = tracks_df.groupby("track_id").size()
    # Real tracks only run 5 time steps total (max observed length 3), vs.
    # the synthetic scene's 30 -- a fixed ">=6" threshold would silently
    # hide every real track, so the bar scales with what's actually there.
    min_len = 2 if source == "real" else 6
    usable = [tid for tid in long_tracks if track_lengths[tid] >= min_len]

    mode = st.radio("Episode source", ["Real detected track (replay)", "Procedural (random eddy)"], horizontal=True)

    colA, colB = st.columns([1, 1])
    with colA:
        policy_name = st.selectbox("Policy", ["PPO (trained)", "Direct navigation (baseline)", "Random (baseline)"])
    with colB:
        run_btn = st.button("▶ Run episode", type="primary")

    def get_policy(name):
        if name.startswith("PPO"):
            return lambda obs: model.predict(obs, deterministic=True)[0]
        if name.startswith("Direct"):
            return lambda obs: direct_navigation_policy(obs)
        rng = np.random.default_rng()
        return lambda obs: random_policy(obs, rng)

    def build_replay_sequence(track_id):
        sub = track_b_df[track_b_df.track_id == track_id].sort_values("time_index")
        recs = []
        for _, r in sub.iterrows():
            recs.append(TrackBRecord(
                track_id=r.track_id, time_index=int(r.time_index), eddy_id=r.eddy_id,
                centroid_lat=r.centroid_lat, centroid_lon=r.centroid_lon, polygon_lonlat=None,
                persistence_m=r.persistence_m, mean_speed=r.mean_speed, rotation=r.rotation,
                track_length=int(r.track_length), velocity_u_ms=r.velocity_u_ms, velocity_v_ms=r.velocity_v_ms,
                displacement_km_last_step=r.displacement_km_last_step, predicted_lat=r.predicted_lat,
                predicted_lon=r.predicted_lon, confidence=r.confidence, fit_r2=r.fit_r2,
            ))
        dt_hours = 38.8 if source == "real" else 24.0  # real: avg actual SWOT revisit gap, not an assumed daily cadence
        return track_b_records_to_replay_steps(recs, dt_hours=dt_hours)

    if run_btn:
        # The synthetic-tab policy (ppo_eddychase.zip) was trained on the
        # original idealized scale (0-3 km/step, 6km agent step, 150km
        # arena); the real-data-calibrated policy (ppo_eddychase_v5.zip,
        # loaded when source=="real") was trained on EddyChaseConfig's
        # current real-world-calibrated defaults. Running either policy
        # through the other's scale would be a meaningless comparison, so
        # the episode config has to track which policy is actually loaded.
        cfg = EddyChaseConfig(max_steps=40) if source == "real" else EddyChaseConfig(
            arena_half_km=150.0, step_km=6.0, max_steps=40, success_radius_km=10.0,
            eddy_speed_kmstep_range=(0.0, 3.0), eddy_wobble_kmstep=0.4,
        )
        if mode.startswith("Real") and usable:
            track_id = usable[0] if "track_choice" not in st.session_state else st.session_state["track_choice"]
            seq = build_replay_sequence(track_id)
            env = EddyChaseEnv(config=cfg, mode="replay", replay_sequence=seq)
        else:
            env = EddyChaseEnv(config=cfg, mode="procedural", seed=np.random.randint(0, 1_000_000))

        policy = get_policy(policy_name)
        obs, info = env.reset()
        agent_path = [env.agent_xy.copy()]
        eddy_path = [env.eddy_xy.copy()]
        pred_path = [env._predicted_xy().copy()]
        done = False
        total_reward = 0.0
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            agent_path.append(env.agent_xy.copy())
            eddy_path.append(env.eddy_xy.copy())
            pred_path.append(env._predicted_xy().copy())
            done = terminated or truncated

        agent_path = np.array(agent_path)
        eddy_path = np.array(eddy_path)
        pred_path = np.array(pred_path)

        frames = []
        for k in range(len(agent_path)):
            frames.append(go.Frame(
                data=[
                    go.Scatter(x=agent_path[:k+1, 0], y=agent_path[:k+1, 1], mode="lines+markers",
                               line=dict(color="#2ca02c", width=2), marker=dict(size=6), name="Agent"),
                    go.Scatter(x=eddy_path[:k+1, 0], y=eddy_path[:k+1, 1], mode="lines+markers",
                               line=dict(color="#d62728", width=2), marker=dict(size=6), name="Eddy"),
                    go.Scatter(x=[pred_path[k, 0]], y=[pred_path[k, 1]], mode="markers",
                               marker=dict(size=14, symbol="star", color="#ff7f0e"), name="Predicted target"),
                ],
                name=str(k),
            ))

        fig = go.Figure(
            data=frames[0].data,
            frames=frames,
            layout=go.Layout(
                height=600, xaxis_title="x (km)", yaxis_title="y (km)",
                yaxis=dict(scaleanchor="x"),
                updatemenus=[dict(type="buttons", showactive=False, buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 180, "redraw": True}, "fromcurrent": True}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                ])],
                title=f"{policy_name} — success={info['success']}  steps={info['t']}  "
                      f"travel={info['travel_km']:.0f}km  reward={total_reward:.1f}",
            ),
        )
        half = cfg.arena_half_km
        fig.update_xaxes(range=[-half, half])
        fig.update_yaxes(range=[-half, half])
        st.plotly_chart(fig, width='stretch')

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Success", "Yes" if info["success"] else "No")
        m2.metric("Steps to resolve", info["t"])
        m3.metric("Distance travelled", f"{info['travel_km']:.0f} km")
        m4.metric("Episode reward", f"{total_reward:.1f}")

    st.divider()
    st.subheader("PPO vs. baselines")
    rl = report["rl"]
    for scope_name, scope_key in [("Procedural (training distribution)", "procedural"), ("Real detected tracks (replay)", "replay")]:
        if scope_key not in rl:
            continue
        rows = []
        for policy_key, label in [("ppo", "PPO"), ("direct", "Direct navigation"), ("random", "Random")]:
            d = rl[scope_key][policy_key]
            rows.append({"Policy": label, "Success rate": d["success_rate"], "Mean reward": d["mean_reward"],
                         "Mean travel (km)": d["mean_travel_km"], "Mean steps": d["mean_steps"]})
        rdf = pd.DataFrame(rows)
        st.markdown(f"**{scope_name}**")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(rdf.round(2), width='stretch', hide_index=True)
        with c2:
            bar = go.Figure(go.Bar(x=rdf.Policy, y=rdf["Success rate"], marker_color=["#2ca02c", "#ff7f0e", "#7f7f7f"]))
            bar.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Success rate")
            st.plotly_chart(bar, width='stretch')


# ----------------------------------------------------------------------------
# Evaluation report
# ----------------------------------------------------------------------------
with tab_eval:
    if source == "real":
        st.subheader("Full evaluation report (real data — no ground truth available)")
        st.markdown(
            """
Real satellite data has no answer key, so nothing here is an accuracy score against a known truth.
What *is* checked, honestly: physical plausibility (does velocity-derived vorticity sign agree with
the rotation label; is circulation strong enough to not be noise; is the eddy's own core actually
enclosed by its ring polygon), track continuity across real SWOT revisits, and — wherever the sparse
SWOT swath happens to cross a detected eddy — an independent cross-check against real SWOT SSH sign,
which is non-gating since coverage is only ~5% of the domain by construction.
            """
        )
    else:
        st.subheader("Full evaluation report (synthetic ground truth)")

    def _fmt(key, val):
        label = key.replace("_", " ").replace("n ", "# ").strip().capitalize()
        if isinstance(val, bool):
            return label, "Yes" if val else "No"
        if isinstance(val, float):
            return label, f"{val:,.3f}" if abs(val) < 10 else f"{val:,.1f}"
        return label, val

    def render_section(title, section):
        if not isinstance(section, dict) or not section:
            return
        note = section.get("note")
        scalars = {k: v for k, v in section.items()
                   if k != "note" and not isinstance(v, (dict, list))}
        nested = {k: v for k, v in section.items()
                  if k != "note" and isinstance(v, (dict, list))}
        st.markdown(f"#### {title}")
        if scalars:
            cols = st.columns(min(4, len(scalars)))
            for i, (k, v) in enumerate(scalars.items()):
                label, val = _fmt(k, v)
                cols[i % len(cols)].metric(label, val)
        for k, v in nested.items():
            if isinstance(v, dict) and all(isinstance(x, dict) for x in v.values()):
                # two-level nesting, e.g. rl.procedural.{ppo,direct,random}
                rows = []
                for policy_key, d in v.items():
                    row = {"Policy": policy_key.replace("_", " ").title()}
                    row.update({_fmt(kk, vv)[0]: _fmt(kk, vv)[1] for kk, vv in d.items()})
                    rows.append(row)
                st.markdown(f"**{k.replace('_', ' ').title()}**")
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            elif isinstance(v, dict):
                st.markdown(f"**{k.replace('_', ' ').title()}**")
                st.dataframe(
                    pd.DataFrame([{"": kk, " ": vv} for kk, vv in v.items()]),
                    width='stretch', hide_index=True,
                )
            elif isinstance(v, list) and v:
                st.markdown(f"**{k.replace('_', ' ').title()}**")
                st.dataframe(pd.DataFrame(v), width='stretch', hide_index=True)
        if note:
            st.caption(f"ℹ️ {note}")
        st.divider()

    section_titles = {
        "scene": "Scene", "detection": "Detection", "track_continuity": "Track continuity",
        "forecast": "Forecast", "rl": "PPO navigation agent",
        "real_world_cross_check": "Independent cross-check (closed-contour method)",
    }
    for key, section in report.items():
        render_section(section_titles.get(key, key.replace("_", " ").title()), section)

    with st.expander("Raw report JSON"):
        st.json(report)
