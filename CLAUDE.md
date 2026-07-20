# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A ComfyUI custom node package (`comfyui-prompt-enhancer`). It adds one node, **Prompt Enhancer (LLM)**, which sends a user's image idea to an LLM via OpenRouter and returns an engineered `positive_prompt`/`negative_prompt` pair tailored to either the SDXL or Z-Image-Turbo architecture. This is a ComfyUI plugin, not a standalone app — it only runs inside a ComfyUI installation, loaded from `ComfyUI/custom_nodes/`.

## Architecture

Four files, each with a single responsibility:

- `__init__.py` — ComfyUI entry point. Exposes `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`, and `WEB_DIRECTORY`. This is what ComfyUI imports on startup.
- `prompt_enhancer_node.py` — the `PromptEnhancer` node class. Defines `INPUT_TYPES`, drives the request/response flow, and parses the LLM's freeform text response into separate positive/negative strings (`_parse_response`). It looks for `POSITIVE:`/`NEGATIVE:` markers and falls back to stripping conversational preamble/postamble if the LLM doesn't follow the format exactly.
- `openrouter_client.py` — thin `requests`-based wrapper around the OpenRouter chat completions endpoint (`call_openrouter`). Maps HTTP status codes (401/402/429/other) to human-readable `RuntimeError` messages; the node surfaces these directly as the output text instead of raising ComfyUI-level errors.
- `system_prompts.py` — the prompt-engineering knowledge base. Two large system prompts (`SDXL_SYSTEM_PROMPT`, `ZIT_SYSTEM_PROMPT`) encode very different rules per target architecture (see below), plus an `ITERATION_ADDENDUM` appended when refining an existing prompt. `build_system_prompt(target_model, is_iteration)` assembles the final system prompt.
- `web/prompt_enhancer.js` — frontend extension registered via `WEB_DIRECTORY`. Adds two read-only multiline widgets (`positive_output`, `negative_output`) to the node so generated prompts render directly on the node UI; populated from the `ui.text` payload returned by `enhance_prompt` on `onExecuted`.

### Data flow

`enhance_prompt()` in `prompt_enhancer_node.py` is the node's execution entrypoint:
1. Validates `user_input` and `openrouter_api_key` are non-empty, returning an inline error string as the output otherwise (no exceptions raised to ComfyUI).
2. Determines iteration mode from whether `previous_prompt` is wired in and non-empty.
3. Builds the system prompt (`system_prompts.build_system_prompt`) and user message (`_build_user_message`, which formats either `## Image Idea` or `## Previous Prompt` + `## Requested Changes`).
4. Calls `openrouter_client.call_openrouter`.
5. Parses the response into positive/negative via `_parse_response`.
6. Returns both a `result` tuple (for downstream nodes) and a `ui` dict (for the frontend widgets) — this dual return is required for `OUTPUT_NODE = True` nodes that need to both output values and display them.

`IS_CHANGED` always returns `NaN` to force re-execution on every queue, since LLM output is non-deterministic and ComfyUI would otherwise cache based on unchanged inputs.

### SDXL vs Z-Image-Turbo — why the code branches this way

These two architectures need fundamentally different prompt shapes, which is why `system_prompts.py` has two near-fully-separate prompt templates rather than one parameterized template:

| | SDXL | Z-Image-Turbo |
|---|---|---|
| Style | Natural language + light `(word:1.1–1.4)` weighting | Pure prose, no weighting/brackets |
| Length | 40-70 tokens | 80-250 words (sweet spot 120-180) |
| Quality tags | Avoided | Forbidden (cause artifacts) |
| Negative prompt | Yes, 15-30 tokens | Never (guidance_scale=0.0 ignores it) |
| Anti-hallucination | Via negative prompt | Inline end-of-prompt language |

`_parse_response` always forces `negative = ""` for Z-Image-Turbo regardless of what the LLM returns, since negative prompts are meaningless for that architecture.

## Development

There is no build step, package manager, linter, or test suite configured in this repo — it's plain Python + one vanilla JS file, installed by copying/symlinking into `ComfyUI/custom_nodes/`.

- Install the single dependency: `pip install -r requirements.txt` (just `requests`).
- To sanity-check prompt assembly without a full ComfyUI environment, exercise `system_prompts.build_system_prompt` directly, e.g.:
  ```
  python -c "from system_prompts import build_system_prompt; print(build_system_prompt('SDXL'))"
  ```
- To actually run the node, install/link this directory under `ComfyUI/custom_nodes/` and restart ComfyUI; there's no standalone runner.
- When changing `_parse_response`, test against realistic LLM outputs that both do and don't follow the `POSITIVE:`/`NEGATIVE:` marker format, since the fallback preamble-stripping logic is regex-based and heuristic.
