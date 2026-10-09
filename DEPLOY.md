# Deploying EddyHunter to the web (Streamlit Community Cloud)

This gets the app running at a public `https://*.streamlit.app` URL, auto-
syncing new real OSCAR/SWOT data from NASA Earthdata itself instead of
reading a local folder. Everything below is a one-time setup; none of it
needs redoing for day-to-day use afterward.

**I (Claude) can't do these steps for you** — they require signing in to
your own GitHub and Streamlit Cloud accounts, which I'm not able to do on
your behalf. Follow them in order; ask me if anything errors.

## 0. What you need first
- A free [GitHub](https://github.com) account.
- A free [Streamlit Community Cloud](https://share.streamlit.io) account
  (sign in with GitHub — one click, no separate password).
- Your NASA Earthdata Login (the same one you already used to manually
  download the SWOT/OSCAR files — no new signup needed). If you don't have
  it handy, reset it at https://urs.earthdata.nasa.gov.

## 1. Push this project to a new GitHub repo
This project's real-data scene file (`outputs/real_scene_fine.nc`, ~204MB)
is bigger than GitHub's 100MB plain-file limit, so it needs **Git LFS**
(free, built into `git`, just needs a one-time install). Everything else in
the project is small and commits normally.

In a terminal, inside this unzipped project folder:
```bash
git init
git lfs install
git lfs track "outputs/real_scene_fine.nc"
git add .gitattributes
git add .
git commit -m "Initial EddyHunter web deployment"
```
Then create a new empty repo on github.com (no README/license — just the
bare repo), and push:
```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```
(If `git lfs` isn't installed yet: Windows → `winget install GitHub.GitLFS`,
or download from https://git-lfs.com — then re-run `git lfs install` once.)

**Double-check `.streamlit/secrets.toml` was NOT pushed** (only
`secrets.toml.example` should be in the repo — the real one, if you ever
make it locally, is excluded by `.gitignore`). Real credentials only ever
go into Streamlit Cloud's own Secrets box in step 3, never into a file in
the repo.

## 2. Create the Streamlit Cloud app
1. Go to https://share.streamlit.io → **New app**.
2. Pick the repo and branch you just pushed, and set **Main file path** to
   `app.py`.
3. Click **Advanced settings** → confirm Python version `3.11` (matches
   `runtime.txt`).
4. Don't click Deploy yet — do step 3 first so the app has its NASA
   credentials from the very first boot.

## 3. Add your NASA Earthdata credentials as Secrets
Still in the app's creation/settings screen: open **Secrets**, and paste:
```toml
EARTHDATA_USERNAME = "your-earthdata-username"
EARTHDATA_PASSWORD = "your-earthdata-password"
```
This box is private to your Streamlit Cloud account — it's not part of the
repo, not visible to anyone you share the app link with, and I never see or
type these values. Without this step, the app still runs, but falls back to
the local-folder refresh UI (which does nothing useful on a server with no
local folders to read).

## 4. Deploy
Click **Deploy**. First build takes a few minutes (installing `gudhi`,
`stable-baselines3`, etc. from `requirements.txt`). Once it's up:
- It loads the seeded 93-day real scene (May 17 – Aug 17 2024, 532 eddies,
  211 tracks) immediately — nothing is lost by moving to the web.
- On that first real page load it also checks NASA Earthdata for anything
  newer and merges it in automatically; after that it re-checks roughly
  every 20 hours (matching OSCAR's own once-a-day update cadence — see the
  sidebar's "NASA Earthdata auto-sync" panel for the honest explanation of
  why that's the right cadence, not a limitation).
- A "🔄 Force sync now" button in the sidebar re-checks on demand anytime.

## Known constraints, stated plainly
- **Free-tier resources are limited** (Streamlit Community Cloud's free
  tier is roughly 1 vCPU / 1GB RAM). Loading the ~204MB real scene plus
  running persistent-homology detection across it is the same real work
  described earlier in this project — it was measured at a few minutes on
  this project's own dev machine, and a constrained free host could be
  slower, not faster. If it's consistently too slow, the fix is a paid tier
  or a VM with more RAM/CPU, not a code change.
- **Git LFS free quota** is 1GB storage / 1GB bandwidth per month on a
  personal GitHub account. The 204MB seed file fits storage easily; each
  full redeploy re-pulls it and counts against the monthly bandwidth quota,
  so a handful of redeploys a month is fine, constant redeploying is not.
- **NASA Earthdata rate limits / outages**: if a sync fails (bad
  credentials, NASA maintenance, no new granules yet), the app shows a
  warning and falls back to the last good cached data rather than crashing
  — it never leaves visitors looking at a broken page.
