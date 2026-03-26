# EMMA-Q Bridge Notes

This file keeps the short integration summary for the EMMA-Q bridge layer.

## Scope

- adapts frontend parameter payloads to the EEG generation scripts
- stores job artifacts as JSON plus MIDI files
- calls the MASEval backend adapter when evaluation is requested
- never reads DeepSeek credentials from the browser

## Public routes

- `GET /api/health`
- `POST /api/preferences`
- `POST /api/generate`
- `POST /api/pipeline/run`
- `POST /api/evaluate`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/artifacts/{artifact_key}`

## Relative path behavior

- EEG job data lives under `.api_data/`
- the MASEval adapter is discovered from `EMMAQ_MASEVAL_ADAPTER` first
- if not configured, the backend looks for a sibling `../MASEval/MASEval/maseval_backend_adapter.py`

## Deployment note

Use this bridge behind a normal Python host. Keep the frontend static and let it call this backend over HTTPS.
