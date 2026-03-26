# MASEval Backend Notes

This directory is intended to live beside `EEG-Music-Generation-New` in the backend repository.

## Runtime role

- `MASEval/maseval_backend_adapter.py` is the stable JSON wrapper used by the EEG backend.
- `MASEval/music_eval_v4_single_cached_fixed_v2.template.py` is the actual evaluation template in use.

The frontend never calls this directory directly.

## Environment variables

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`

Both are read on the backend only.

## Files that should not be committed

- demo output directories
- report JSON files generated during testing
- evaluation logs
- caches under `MASEval/.cache_music_eval_v4*`
- local virtual environments

## Deployment note

Install this project on the same host as the EEG backend or set `EMMAQ_MASEVAL_ADAPTER` and `EMMAQ_MASEVAL_PYTHON` explicitly in the EEG backend environment.
