# Prompt Enhancer

A small, self-hosted web app that uses an LLM (via OpenRouter) to craft image-generation prompts for SDXL, Z-Image-Turbo, and any other model family you configure.

## Features

- **Generar** — turn a natural-language idea into a ready-to-copy positive/negative prompt pair for the selected model family, or refine an existing prompt (generated here or pasted from elsewhere) by leaving the "existing prompt" field filled in.
- **Imagen** — upload an image and generate a prompt from it using an OpenRouter vision model.
- **Personajes** — reusable text snippets you can insert into the idea/changes field with one click.
- **Admin panel** — edit the three independent system prompts (Generar/Iterar/Imagen), manage model families (their rules and whether they use negative prompts), and manage personajes, all from the browser.
- **Historial** — read-only log of every prompt generated.

## Setup

1. Copy `.env.example` to `.env` and fill in:
   - `OPENROUTER_API_KEY` — from https://openrouter.ai/keys
   - `ADMIN_USERNAME` / `ADMIN_PASSWORD` — credentials for the HTTP Basic Auth prompt protecting the whole app
2. `docker compose up --build`
3. Open `http://localhost:8000` and log in with the credentials above.

On first run, the app seeds `data/families.json` with SDXL and Z-Image-Turbo. Add more families (Krea 2, Pony, Flux, ...) from the Admin panel.

## Development

```
pip install -e ".[dev]"
pytest
uvicorn app.main:app --reload
```

Data (families, characters, the three per-mode system prompts, and history) lives under `data/`, which is gitignored and mounted as a Docker volume so it survives container rebuilds.
