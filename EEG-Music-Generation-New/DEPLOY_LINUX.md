# Linux Deployment Checklist

## Recommended directories

```text
/opt/brainwave-stack/
  EEG-Music-Generation-New/
  MASEval/
```

## Backend virtual environments

```bash
cd /opt/brainwave-stack/EEG-Music-Generation-New
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install MASEval dependencies in its own environment and then export `EMMAQ_MASEVAL_PYTHON` to that interpreter.

## Required environment variables

```bash
export DEEPSEEK_API_KEY='replace-me'
export DEEPSEEK_BASE_URL='https://api.deepseek.com/v1'
export EMMAQ_ENABLE_MASEVAL='1'
export EMMAQ_MASEVAL_ADAPTER='/opt/brainwave-stack/MASEval/MASEval/maseval_backend_adapter.py'
export EMMAQ_MASEVAL_PYTHON='/opt/brainwave-stack/MASEval/MASEval/.venv/bin/python'
export EMMAQ_MASEVAL_TIMEOUT_SECONDS='600'
export EMMAQ_ALLOW_ORIGINS='https://app.your-domain.com'
```

## Start command

```bash
cd /opt/brainwave-stack/EEG-Music-Generation-New
source .venv/bin/activate
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## First checks

1. `curl http://127.0.0.1:8000/api/health`
2. verify `evaluation_backend.adapter_found` is `true`
3. verify `evaluation_backend.deepseek_api_key_present` matches your environment
4. run one local `POST /api/pipeline/run` request before pointing Cloudflare Pages at the backend
