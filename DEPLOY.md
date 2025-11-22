# DEPLOY.md — How to run & host the Streamlit demo

This document gives short, practical steps to run the app locally and deploy it in common ways.

Prerequisites
- Python 3.10+ and `pip` available.
- Git (for Streamlit Community Cloud or other git-based deploys).
- Optional: Docker (for container-based deploy).

1) Run locally (Windows PowerShell)

  - Create and activate a venv (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

  - Install runtime deps (development or full runtime):

```powershell
python -m pip install -r requirements.txt
```

  - Run Streamlit locally:

```powershell
python -m streamlit run app.py --server.headless true
```

  - Open http://localhost:8501 in your browser.

Notes:
- `requirements.txt` includes `shap` (heavy). If you want faster CI or smaller containers, use `requirements-ci.txt` (this excludes `shap`).
- If you don't have `models/demo_model.joblib` or a `DATA_PATH` environment variable, the app will train a small demo model on first run (takes a short time).

2) Deploy on Streamlit Community Cloud (recommended for quick, free demos)

- Make sure your project is pushed to GitHub and that the repo contains `app.py` at top level (or set the path when configuring the app).
- In Streamlit Community Cloud (https://share.streamlit.io):
  1. Click "New app" and connect your GitHub account.
  2. Select your repo and branch (e.g., `main` or `enhance/ci-lint-tests`).
  3. Set `app.py` as the main file and the correct branch.
  4. (Optional) In Secrets, set `DATA_PATH` or `MODEL_PATH` if you want to use remote data or a different model path.

Good-to-know:
- If `shap` is present, deploys may take longer and require more build time. Remove `shap` from `requirements.txt` if you want a smaller, faster build (and rely on the model's saved `feature_importances_` or permutation fallback in the app).

3) Deploy with Docker (self-host or cloud provider)

- Build the image (from project root):

```powershell
docker build -t car-price-demo .
```

- Run locally:

```powershell
docker run -p 8501:8501 --name car-demo -e PORT=8501 car-price-demo
```

- Push to a container registry (Docker Hub, Azure Container Registry, etc.) and deploy to your cloud of choice (Azure App Service, AWS ECS, GKE, Heroku container registry).

4) GitHub Actions → Streamlit Community Cloud or Container registry

- You can add a GitHub Actions workflow that builds the repo and either:
  - Pushes a Docker image to a registry, or
  - Automatically merges to the branch used by Streamlit Cloud (Streamlit will rebuild on push).

5) Troubleshooting & tips

- If Diagnostics shows an informative `st.info`/`st.error` message, follow that guidance (typically about missing `feature_names`, insufficient sample rows, or missing model file).
- Use `requirements-ci.txt` for CI runs to avoid installing heavy optional packages like `shap`.
- If deployments fail due to long build times, remove or pin heavy packages and pre-build a model artifact (`models/demo_model.joblib`) committed to the repo or hosted on a fast object store.