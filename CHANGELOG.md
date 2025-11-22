# Changelog

All notable changes to this project are documented in this file.
This project follows the "Keep a Changelog" principles and is
semantically versioned where practical.

## [Unreleased]

- Add interactive Streamlit demo (`app.py`) with Predict, EDA and Diagnostics tabs.
- Diagnose model explainability using SHAP when available with safe fallbacks:
  - SHAP explainer when `shap` is installed and the model contains saved background data.
  - Fall back to model `feature_importances_` for tree models.
  - If needed, compute permutation importance as a robust fallback.
- Add a compact sklearn `Pipeline` and save a demo model artifact (`models/demo_model.joblib`).
- Add unit tests (`tests/test_pipeline.py`, `tests/test_diagnostics.py`) and CI to run `pytest` and `flake8`.
- Keep `shap` in `requirements.txt` (optional heavy dependency); use `requirements-ci.txt` (without `shap`) for CI.
- Replace deprecated Streamlit plotting parameter and harden Diagnostics messaging to give a clear reason when diagnostics can't be computed.
- Remove developer helper scripts from `scripts/` for a production-ready repo layout.

## [0.1.0] - 2025-11-22

- Initial public demo release: basic data pipeline, demo model, Streamlit demo and tests.

## Notes / Migration

- Branch `enhance/ci-lint-tests` contains CI, lint and diagnostics improvements.
- If you retrain or update the model artifact, increment the version and add a short entry here.

---

For hosting and deployment instructions, see `DEPLOY.md`.
# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- Initial repository scaffolding and analysis notebook.

