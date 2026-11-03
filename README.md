# Prompt Enhancer

A small, self-hosted web app that uses an LLM (via OpenRouter) to craft image-generation prompts for SDXL, Z-Image-Turbo, and any other model family you configure.

## Features

**Crear** is a two-column workspace: the composer on the left, and on the right every result of the session, newest first, kept across mode switches and page reloads (until the tab is closed).

- **Generar** — turn an idea into a ready-to-copy positive/negative prompt pair for the selected model family. Ask for 1–4 variants to get independent alternatives side by side.
- **Imagen** — generate a prompt from a reference image (pick, drag or paste it) with an OpenRouter vision model, also in 1–4 variants.
- **Historia** — tell a story as 2–12 sequential prompts that keep characters and style consistent.
- **Variaciones** — keep characters, aesthetic and base idea constant and choose, per aspect (outfit, setting, lighting, pose, camera, expression), whether it stays fixed (optionally to a given value) or varies across 2–12 prompts (optionally within hints) — a photo shoot of the same subject. Optionally start from up to 12 prompts you already like, usually of the same character: the LLM reads them together, keeps what they share as the constant subject and takes the best of each. Paste them in, or send any result there with "Añadir a referencias" (also from Historial).
- **Conversar** — a chat with a prompt-crafting partner whose personality you write. It proposes prompts (shown as numbered, copyable cards, so you can say "the 3, but at night"), comments on them, and takes your feedback and the images you generated (paste or drop them) to refine them. Replies stream in as they are written; conversations are saved and can be resumed. The bot keeps short notes about your tastes and what works with each model, carries them into every conversation, and adds, rewrites or forgets them as it learns — each change shows in the chat, with a way to undo it. "Iterar" on any prompt elsewhere starts a conversation from it.
  - **Images and their metadata** — every image in a conversation is numbered. An image you pick or drop keeps the metadata it was generated with (ComfyUI's embedded workflow, or A1111/Forge/Civitai parameters: prompt, models, LoRAs, seed, sampler…); the bot reads it and you can see it under the image.
  - **Running ComfyUI** (with `COMFY_GATEWAY_URL` set) — the bot launches workflows on your ComfyUI through comfy-gateway: from the library, from an image's embedded workflow or from an earlier run, changing the prompt, seed, LoRAs, sampler, size or any node. The images arrive in the chat as a contact sheet and the bot comments on them by itself. React to each one (thumbs up, thumbs down, a comment): reactions go to the bot with your next message, or on their own. It also knows the models and LoRAs you have installed, what you wrote about each, and how the images made with them went.
- **Personajes** — reusable character descriptions, selected as chips and inlined verbatim into the prompt. Create, edit and delete them from the workspace, or extract one from any generated prompt or reference image.
- **Historial** — every generation, searchable and filterable by mode, family and favorites, with per-entry cost (prompts proposed in Conversar included, with a link back to their conversation); take any prompt into a new conversation or use it as a Variaciones reference, reload a past set of references, star an entry or delete it.
- **Peticiones** (from Ajustes) — every request sent to OpenRouter exactly as it went out, with what came back: messages, images, parameters, tokens, cost, timing and errors. Each Conversar reply and history entry links to its own.
- **Ajustes** — edit the per-mode system prompts (with a one-click reload of the original text) including the Conversar bot's personality, the bot's memory, the model families (their rules and whether they use negative prompts), the characters, the ComfyUI workflow library (import from an API-format JSON or from an image ComfyUI generated) and the installed models and LoRAs, with a description of what each one does.

## Setup

1. Copy `.env.example` to `.env` and fill in:
   - `OPENROUTER_API_KEY` — from https://openrouter.ai/keys
   - `ADMIN_USERNAME` / `ADMIN_PASSWORD` — credentials for the HTTP Basic Auth prompt protecting the whole app
   - optionally `DEFAULT_LLM_MODEL` / `DEFAULT_VISION_MODEL` — the models prefilled until you pick others in the UI
   - optionally `COMFY_GATEWAY_URL` (and `COMFY_DIRECT_URL`, ComfyUI itself, used only when the gateway doesn't answer) — lets Conversar's bot run workflows on your ComfyUI
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
