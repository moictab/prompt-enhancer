# Prompt Enhancer (LLM) for ComfyUI

A custom ComfyUI node that uses an LLM (via OpenRouter) to craft optimized image generation prompts. Intelligently adapts to **SDXL** and **Z-Image-Turbo** architectures, which require fundamentally different prompting strategies.

## Features

- **Architecture-aware prompting** — SDXL gets concise weighted prompts; Z-Image-Turbo gets flowing natural language prose
- **Fresh generation** — Describe an idea and get a fully engineered prompt
- **Iterative refinement** — Wire the output back in and give modification instructions
- **Configurable LLM** — Use any model available on OpenRouter (Claude, GPT, Llama, etc.)

## Installation

### Manual Install

1. Navigate to your ComfyUI custom nodes directory:
   ```
   cd ComfyUI/custom_nodes/
   ```
2. Clone or copy this repository:
   ```
   git clone https://github.com/moictab/prompt-enhancer.git
   ```
3. Install dependencies:
   ```
   pip install -r prompt-enhancer/requirements.txt
   ```
4. Restart ComfyUI.

### ComfyUI Manager

Search for "Prompt Enhancer" in the ComfyUI Manager and install.

## Setup

1. Get an API key from [OpenRouter](https://openrouter.ai/keys).
2. Add the **Prompt Enhancer (LLM)** node to your workflow (found under `prompt/enhance`).
3. Paste your API key into the `openrouter_api_key` input.

## Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `user_input` | STRING (multiline) | — | Your image idea or iteration instructions |
| `target_model` | COMBO | `SDXL` | Target architecture: `SDXL` or `Z-Image-Turbo` |
| `openrouter_api_key` | STRING | — | Your OpenRouter API key |
| `llm_model` | STRING | `anthropic/claude-sonnet-4` | Any OpenRouter model ID |
| `creativity` | FLOAT (slider) | `0.7` | LLM temperature (0.0 = deterministic, 1.0 = creative) |
| `previous_prompt` | STRING (optional) | — | Wire-only input for iterative refinement |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `positive_prompt` | STRING | The enhanced positive prompt |
| `negative_prompt` | STRING | Negative prompt (empty for Z-Image-Turbo) |

## Usage

### Fresh Prompt Generation

1. Add the **Prompt Enhancer (LLM)** node.
2. Type your idea into `user_input` (e.g., "a cyberpunk samurai in neon rain").
3. Select your `target_model`.
4. Queue the prompt.
5. Connect `positive_prompt` to a **CLIP Text Encode** node.
6. Connect `negative_prompt` to a second **CLIP Text Encode** node (for the negative conditioning).

### Iterative Refinement

1. Wire the `positive_prompt` output back into the `previous_prompt` input of the same (or a new) Prompt Enhancer node.
2. Type your changes into `user_input` (e.g., "add lightning, move to a rooftop").
3. Queue — the LLM will refine the previous prompt rather than starting from scratch.

## Architecture Differences

| Feature | SDXL | Z-Image-Turbo |
|---------|------|---------------|
| Prompt style | Natural language + light weighting `(word:1.2)` | Pure natural language prose |
| Length | 40-70 tokens | 80-250 words |
| Quality tags | Avoided (dilutes prompt) | Forbidden (causes artifacts) |
| Negative prompt | Yes (15-30 tokens) | No (guidance_scale=0.0) |
| Anti-hallucination | Via negative prompt | End-of-prompt language |

## Troubleshooting

- **"API key is invalid"** — Double-check your key at [openrouter.ai/keys](https://openrouter.ai/keys)
- **"Insufficient credits"** — Add credits at [openrouter.ai/credits](https://openrouter.ai/credits)
- **"Rate limit exceeded"** — Wait a moment and queue again
- **Node not appearing** — Make sure the folder is in `ComfyUI/custom_nodes/` and restart ComfyUI
