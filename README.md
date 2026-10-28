# Prompt Enhancer

A small, self-hosted web app that uses an LLM (via OpenRouter) to craft image-generation prompts for SDXL, Z-Image-Turbo, and any other model family you configure.

## Features

**Crear** is a two-column workspace: the composer on the left, and on the right every result of the session, newest first, kept across mode switches and page reloads (until the tab is closed).

- **Generar** — turn an idea into a ready-to-copy positive/negative prompt pair for the selected model family. Paste an existing prompt as the starting prompt to refine it instead ("Iterar"). Ask for 1–4 variants to get independent alternatives side by side.
- **Imagen** — generate a prompt from a reference image (pick, drag or paste it) with an OpenRouter vision model, also in 1–4 variants.
- **Historia** — tell a story as 2–12 sequential prompts that keep characters and style consistent.
- **Variaciones** — keep characters, aesthetic and base idea constant and choose, per aspect (outfit, setting, lighting, pose, camera, expression), whether it stays fixed (optionally to a given value) or varies across 2–12 prompts (optionally within hints) — a photo shoot of the same subject. Optionally start from up to 12 prompts you already like, usually of the same character: the LLM reads them together, keeps what they share as the constant subject and takes the best of each. Paste them in, or send any result there with "Añadir a referencias" (also from Historial).
- **Personajes** — reusable character descriptions, selected as chips and inlined verbatim into the prompt. Create, edit and delete them from the workspace, or extract one from any generated prompt or reference image.
- **Historial** — every generation, searchable and filterable by mode, family and favorites, with per-entry cost; reuse any prompt in Generar or as a Variaciones reference, reload a past set of references, star an entry or delete it.
- **Ajustes** — edit the per-mode system prompts (with a one-click reload of the original text), the model families (their rules and whether they use negative prompts) and the characters.

## Setup

1. Copy `.env.example` to `.env` and fill in:
   - `OPENROUTER_API_KEY` — from https://openrouter.ai/keys
   - `ADMIN_USERNAME` / `ADMIN_PASSWORD` — credentials for the HTTP Basic Auth prompt protecting the whole app
   - optionally `DEFAULT_LLM_MODEL` / `DEFAULT_VISION_MODEL` — the models prefilled until you pick others in the UI
2. `docker compose up --build`
3. Open `http://localhost:8000` and log in with the credentials above.

On first run, the app seeds `data/families.json` with SDXL and Z-Image-Turbo. Add more families (Krea 2, Pony, Flux, ...) from Ajustes. System prompts are seeded per mode, so upgrading adds the prompts for new modes without touching the ones you have edited.

## Development

```
pip install -e ".[dev]"
pytest
uvicorn app.main:app --reload
```

Data (families, characters, the per-mode system prompts, and history) lives under `data/`, which is gitignored and mounted as a Docker volume so it survives container rebuilds.
