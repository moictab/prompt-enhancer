"""ComfyUI Prompt Enhancer — LLM-powered prompt engineering for SDXL and Z-Image-Turbo."""

from .prompt_enhancer_node import PromptEnhancer

NODE_CLASS_MAPPINGS = {
    "PromptEnhancer": PromptEnhancer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptEnhancer": "Prompt Enhancer (LLM)"
}

WEB_DIRECTORY = "./web"
