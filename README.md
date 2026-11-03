# Prompt Enhancer

A small, self-hosted web app that uses an LLM (via OpenRouter) to craft image-generation prompts for SDXL, Z-Image-Turbo, and any other model family you configure.

## Features

**Crear** is a two-column workspace: the composer on the left, and on the right every result of the session, newest first, kept across mode switches and page reloads (until the tab is closed).

- **Generar** — turn an idea into a ready-to-copy positive/negative prompt pair for the selected model family. Ask for 1–4 variants to get independent alternatives side by side.
- **Imagen** — generate a prompt from a reference image (pick, drag or paste it) with an OpenRouter vision model, also in 1–4 variants.
- **Historia** — tell a story as 2–12 sequential prompts that keep characters and style consistent.
- **Variaciones** — keep characters, aesthetic and base idea constant and choose, per aspect (outfit, setting, lighting, pose, camera, expression), whether it stays fixed (optionally to a given value) or varies across 2–12 prompts (optionally within hints) — a photo shoot of the same subject. Optionally start from up to 12 prompts you already like, usually of the same character: the LLM reads them together, keeps what they share as the constant subject and takes the best of each. Paste them in, or send any result there with "Añadir a referencias" (also from Historial).
- **Conversar** — a chat with a prompt-crafting partner whose personality you write. It proposes prompts (numbered, folded to their first line with a copy button, so you can say "the 3, but at night" without the chat filling up), comments on them, and takes your feedback and the images you generated (paste or drop them) to refine them. Replies stream in as they are written; conversations are saved and can be resumed, renamed and deleted from the list. Several can be going at once: move to another conversation (or reload, or close the tab) while the bot is answering and the reply goes on, with the list showing which ones are busy and which have a reply you haven't seen. The bot keeps short notes about your tastes and what works with each model, carries them into every conversation, and adds, rewrites or forgets them as it learns — each change shows in the chat, with a way to undo it. "Iterar" on any prompt elsewhere starts a conversation from it.
  - **Civitai** — paste a link to a Civitai image and it joins the conversation with what Civitai knows about it: prompt, settings, and the checkpoint and LoRAs it was made with (weights, base model, trigger words). The bot can reproduce it with what you have installed, saying what's missing, or borrow its style for what you are making.
  - **Images and their metadata** — every image in a conversation is numbered; click one to see it full size (arrow keys move between them). Mention one as `#4` (typing `#` offers them): the bot gets that image again, however long ago it was shared, and `#4` in either side's messages links to it. An image you pick or drop keeps the metadata it was generated with (ComfyUI's embedded workflow, or A1111/Forge/Civitai parameters: prompt, models, LoRAs, seed, sampler…); the bot reads it and you can see it under the image.
  - **Running ComfyUI** (with `COMFY_GATEWAY_URL` set) — the bot launches workflows on your ComfyUI through comfy-gateway: from the library, from an image's embedded workflow or from an earlier run, changing the prompt, seed, LoRAs, sampler, size or any node. It picks the image's proportions for what it shows (portrait, square, landscape, panorama) at the model's own pixel count, and keeps them while refining an image. Ask it to iterate on its own ("hasta que salga bien, máximo 12 imágenes y 2 $") and it launches, reviews and launches again by itself until the goal is met or the budget, in images and in dollars spent on its replies, runs out; a bar over the composer shows what it has used and stops it. The images arrive in the chat as a contact sheet and the bot comments on them by itself (or, with that turned off in the conversation's settings to save a request per run, when you next write or ask it to); in ComfyUI they are saved under `output/prompt-enhancer/`, one folder per conversation (`2026-11-03_witch-at-dawn_a8c1a153/run-01_seed-…`). React to each one (thumbs up, thumbs down, a comment): reactions go to the bot with your next message, or on their own. It also knows the models and LoRAs you have installed, what you wrote about each, how much and at what weight you have used them (aggregates from comfy-gateway's history, never prompts or images), and how the images made with them went.
- **Personajes** — reusable character descriptions, selected as chips and inlined verbatim into the prompt. Create, edit and delete them from the workspace, or extract one from any generated prompt or reference image.
- **Historial** — every generation, searchable and filterable by mode, family and favorites, with per-entry cost (prompts proposed in Conversar included, with a link back to their conversation); take any prompt into a new conversation or use it as a Variaciones reference, reload a past set of references, star an entry or delete it.
- **Peticiones** (from Ajustes) — every request sent to OpenRouter exactly as it went out, with what came back: messages, images, parameters, tokens, cost, timing and errors. Each Conversar reply and history entry links to its own.
- **MCP for agents** — everything above, for AI agents that work without the web page: the app is an MCP server at `/mcp` (see [Connecting an agent](#connecting-an-agent)). An agent can write prompts in every mode, hold conversations with the bot, launch runs, wait for the images, look at them and react, and manage families, characters, notes, system prompts, workflows and models.
- **Ajustes** — edit the per-mode system prompts (with a one-click reload of the original text) including the Conversar bot's personality, the bot's memory, the model families (their rules and whether they use negative prompts), the characters, the ComfyUI workflow library (import from an API-format JSON or from an image ComfyUI generated) and the installed models and LoRAs, with a description of what each one does.

## Privacy

- Every request to OpenRouter asks for providers that neither store nor train on the data (`data_collection: deny`).
- The Conversar bot works in isolation: in ComfyUI it can only use the library workflows, the workflows in images you share with it, its own runs and the images of the conversation. Workflows that would read anything else from ComfyUI (its output folder, files by path, URLs, other projects' history) are refused before they are sent, and the app only downloads the images it made.
- Everything sent to OpenRouter is also logged locally, under `data/requests/`.

## Setup

1. Copy `.env.example` to `.env` and fill in:
   - `OPENROUTER_API_KEY` — from https://openrouter.ai/keys
   - `ADMIN_USERNAME` / `ADMIN_PASSWORD` — credentials for the HTTP Basic Auth prompt protecting the whole app
   - optionally `DEFAULT_LLM_MODEL` / `DEFAULT_VISION_MODEL` — the models prefilled until you pick others in the UI
   - optionally `COMFY_GATEWAY_URL` (and `COMFY_DIRECT_URL`, ComfyUI itself, used only when the gateway doesn't answer) — lets Conversar's bot run workflows on your ComfyUI
2. `docker compose up --build`
3. Open `http://localhost:8000` and log in with the credentials above.

**On Windows, without Docker** (how it runs next to ComfyUI, using the least memory): double-click `Start.bat`. The first time it prepares a Python environment in `.venv` (and again whenever `requirements.txt` changes); then it starts the server in a minimized window and opens `http://127.0.0.1:8090/conversar`. `Stop.bat` stops it (so does closing that window). It also listens on the local network, so a phone on the same Wi-Fi reaches it at `http://<the PC's IP>:8090/conversar` (with the same login; plain HTTP, so only on a network you trust). It keeps its data in `data/` next to it.

On first run, the app seeds `data/families.json` with SDXL and Z-Image-Turbo. Add more families (Krea 2, Pony, Flux, ...) from Ajustes. System prompts are seeded per mode, so upgrading adds the prompts for new modes without touching the ones you have edited.

## Connecting an agent

The app speaks MCP (Model Context Protocol) over streamable HTTP at `http://<host>:<port>/mcp` (on the Windows setup, `http://<the PC's IP>:8090/mcp`, or `http://127.0.0.1:8090/mcp` on the PC itself), behind the same login as the web page: the agent sends an `Authorization: Basic …` header with your `ADMIN_USERNAME:ADMIN_PASSWORD` in base64. To get that value in PowerShell:

```
[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("user:password"))
```

With Claude Code, for example:

```
claude mcp add --transport http prompt-enhancer http://127.0.0.1:8090/mcp --header "Authorization: Basic <that value>"
```

Any MCP client that does streamable HTTP and lets you set a header works the same way. The tools that call an LLM spend your OpenRouter credit, and runs use your GPU, just as from the web page.

## Development

```
pip install -e ".[dev]"
pytest
uvicorn app.main:app --reload
```

Data (families, characters, the per-mode system prompts, and history) lives under `data/`, which is gitignored and mounted as a Docker volume so it survives container rebuilds.
