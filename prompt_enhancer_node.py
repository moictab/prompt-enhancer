"""ComfyUI node that uses an LLM to enhance image generation prompts."""

from .openrouter_client import call_openrouter
from .system_prompts import build_system_prompt


class PromptEnhancer:
    """Enhances image generation prompts using an LLM via OpenRouter.

    Supports SDXL and Z-Image-Turbo target models with architecture-specific
    prompting strategies. Can generate fresh prompts or iteratively refine
    existing ones.
    """

    CATEGORY = "prompt/enhance"
    FUNCTION = "enhance_prompt"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_input": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe your image idea, or give iteration instructions...",
                }),
                "target_model": (["SDXL", "Z-Image-Turbo"],),
                "openrouter_api_key": ("STRING", {
                    "default": "",
                    "placeholder": "sk-or-...",
                }),
                "llm_model": ("STRING", {
                    "default": "anthropic/claude-sonnet-4",
                }),
                "creativity": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                }),
            },
            "optional": {
                "previous_prompt": ("STRING", {"forceInput": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution every queue — LLM output is non-deterministic
        return float("NaN")

    def enhance_prompt(self, user_input, target_model, openrouter_api_key,
                       llm_model, creativity, previous_prompt=None):
        # Validate inputs
        if not user_input.strip():
            return ("Please enter a prompt idea or iteration instructions.", "")

        if not openrouter_api_key.strip():
            return (
                "ERROR: OpenRouter API key is required. "
                "Get your key at https://openrouter.ai/keys",
                "",
            )

        # Determine if this is an iteration
        is_iteration = previous_prompt is not None and previous_prompt.strip() != ""

        # Build prompts
        system_prompt = build_system_prompt(target_model, is_iteration=is_iteration)
        user_message = self._build_user_message(user_input, previous_prompt, is_iteration)

        # Call LLM
        try:
            response = call_openrouter(
                api_key=openrouter_api_key.strip(),
                model=llm_model.strip(),
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=creativity,
            )
        except RuntimeError as e:
            return (f"ERROR: {e}", "")

        # Parse response
        positive, negative = self._parse_response(response, target_model)
        return (positive, negative)

    def _build_user_message(self, user_input, previous_prompt, is_iteration):
        """Format the user message for the LLM."""
        if is_iteration:
            return (
                f"## Previous Prompt\n{previous_prompt}\n\n"
                f"## Requested Changes\n{user_input}"
            )
        return f"## Image Idea\n{user_input}"

    def _parse_response(self, response, target_model):
        """Extract positive and negative prompts from the LLM response.

        Looks for POSITIVE: and NEGATIVE: markers. Falls back to treating
        the entire response as the positive prompt if markers are missing.
        Always returns empty negative for Z-Image-Turbo.
        """
        positive = ""
        negative = ""

        # Try to find markers (case-insensitive)
        response_upper = response.upper()
        pos_idx = response_upper.find("POSITIVE:")
        neg_idx = response_upper.find("NEGATIVE:")

        if pos_idx != -1:
            # Extract positive prompt
            pos_start = pos_idx + len("POSITIVE:")
            if neg_idx != -1 and neg_idx > pos_idx:
                positive = response[pos_start:neg_idx].strip()
            else:
                positive = response[pos_start:].strip()

            # Extract negative prompt
            if neg_idx != -1:
                neg_start = neg_idx + len("NEGATIVE:")
                negative = response[neg_start:].strip()
        else:
            # Fallback: treat entire response as positive prompt
            positive = response.strip()

        # ZIT never uses negative prompts
        if target_model == "Z-Image-Turbo":
            negative = ""

        return (positive, negative)
