# EEG-Music-Generation-New

This directory is the public API entrypoint for the backend stack.

It exposes the JSON API used by the EMMA-Q frontend, generates MIDI from EEG CSV input, stores job artifacts under `.api_data/`, and delegates evaluation to the sibling `MASEval` project.

## Recommended backend repository layout

For GitHub and Linux deployment, keep these directories as siblings:

```text
backend-repo/
  EEG-Music-Generation-New/
  MASEval/
```

The runtime entrypoint is:

- `api_server.py`

The main internal services are:

- `emmaq_generation_service.py`
- `emmaq_generation_adapter.py`
- `maseval_pipeline_adapter.py`

## Local installation

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment variables

Copy `.env.example` to `.env` or configure the variables directly in your process manager.

- `DEEPSEEK_API_KEY`
  Required if evaluation is enabled.
- `DEEPSEEK_BASE_URL`
  Optional. Defaults to `https://api.deepseek.com/v1`.
- `EMMAQ_ENABLE_MASEVAL`
  `1` to enable evaluation, `0` to skip it.
- `EMMAQ_MASEVAL_ADAPTER`
  Optional absolute path to `maseval_backend_adapter.py`.
- `EMMAQ_MASEVAL_PYTHON`
  Recommended in Linux production. Point it to the Python interpreter that has the MASEval dependencies installed.
- `EMMAQ_MASEVAL_TIMEOUT_SECONDS`
  Evaluation subprocess timeout.
- `EMMAQ_ALLOW_ORIGINS`
  Comma-separated frontend origins for CORS.

## Linux deployment notes

Recommended source layout on the host:

```text
/opt/brainwave-stack/
  EEG-Music-Generation-New/
  MASEval/
```

Recommended runtime storage:

- Keep `.api_data/` on persistent disk.
- Do not publish `.api_data/` through Git.

Recommended interpreter paths:

- EEG backend: `/opt/brainwave-stack/EEG-Music-Generation-New/.venv/bin/python`
- MASEval backend: `/opt/brainwave-stack/MASEval/MASEval/.venv/bin/python` or another explicit path exported through `EMMAQ_MASEVAL_PYTHON`

## Start command

Development:

```bash
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Linux production:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## Health check

`GET /api/health`

The health payload includes:

- backend status
- configured CORS origins
- `.api_data` path status
- whether the evaluation adapter was found
- whether `DEEPSEEK_API_KEY` is present

## Main API routes

- `GET /api/health`
- `POST /api/preferences`
- `POST /api/generate`
- `POST /api/pipeline/run`
- `POST /api/evaluate`
- `POST /api/jobs/{job_id}/evaluate`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/artifacts/{artifact_key}`

## Files that should not be committed

- `.env`
- `.venv/`
- `.api_data/`
- `__pycache__/`
- generated MIDI, logs, and JSON output files

## Deployment boundary

This project is suitable for a normal Python host or VM.

It is not a good direct fit for Cloudflare Pages or Workers in its current form because it:

- writes job artifacts to the local filesystem
- runs evaluation as a subprocess
- depends on Python packages and longer-running requests
