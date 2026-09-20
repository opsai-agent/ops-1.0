"""LLaMA model wrapper for OPS."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from .base import BaseModelWrapper


class LLaMAModel(BaseModelWrapper):
    """
    LLaMA model wrapper for OPS Agent.

    Uses Meta's LLaMA as the base model, fine-tuned with OPS training pipeline.
    """

    SUPPORTED_MODELS = {
        "llama-3-8b": {
            "path": "meta-llama/Meta-Llama-3-8B",
            "context_length": 8192,
            "description": "LLaMA 3 8B parameter model"
        },
        "llama-3-70b": {
            "path": "meta-llama/Meta-Llama-3-70B",
            "context_length": 8192,
            "description": "LLaMA 3 70B parameter model"
        },
        "ops-llama-8b": {
            "path": None,  # Custom trained model
            "context_length": 8192,
            "description": "OPS custom trained on LLaMA 3 8B"
        },
    }

    def __init__(
        self,
        model_name: str = "llama-3-8b",
        model_path: str | None = None,
        device: str = "auto",
        max_context_length: int = 8192,
    ):
        self._model_name = model_name
        self._model_path = model_path
        self._device = device
        self._max_context_length = max_context_length
        self._model = None
        self._tokenizer = None
        self._initialized = False

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def context_length(self) -> int:
        return self._max_context_length

    async def initialize(self) -> None:
        """Initialize the model (called lazily on first use)."""
        if self._initialized:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            model_path = self._model_path or self.SUPPORTED_MODELS.get(self._model_name, {}).get("path")
            if not model_path:
                raise ValueError(f"Model path not found for {self._model_name}")

            logger.info(f"Loading model: {model_path}")

            # Determine device
            if self._device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self._device

            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
            )
            self._model.eval()
            self._initialized = True

            logger.info(f"Model loaded on {device}")

        except Exception as e:
            logger.warning(f"Failed to load model {self._model_name}: {e}")
            logger.info("Falling back to mock mode for development/testing")
            self._model = None
            self._tokenizer = None
            self._initialized = True  # Allow mock operations

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stream: bool = False,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Chat completion with tool support."""
        await self.initialize()

        # Extract system message
        system_msg = next((m for m in messages if m["role"] == "system"), None)
        user_msg = next((m for m in reversed(messages) if m["role"] == "user"), None)

        if not user_msg:
            return {"content": "", "finish_reason": "stop", "tool_calls": None}

        prompt = self._build_prompt(messages)

        # If model failed to load, use mock mode
        if self._model is None or self._tokenizer is None:
            return self._mock_response(prompt, tools)

        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                )

            generated_text = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )

            return {
                "content": generated_text,
                "finish_reason": "stop",
                "tool_calls": None,
                "usage": {
                    "prompt_tokens": len(inputs["input_ids"][0]),
                    "completion_tokens": len(outputs[0]) - len(inputs["input_ids"][0]),
                }
            }
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            return {
                "content": f"Error: {str(e)}",
                "finish_reason": "stop",
                "tool_calls": None,
            }

    async def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, **kwargs: Any) -> str:
        """Generate text completion."""
        await self.initialize()

        # If model failed to load, use mock mode
        if self._model is None or self._tokenizer is None:
            return f"[Mock generation] {prompt[:100]}..."

        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )

            return self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
        except Exception as e:
            return f"Error: {str(e)}"

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        info = self.SUPPORTED_MODELS.get(self._model_name, {})
        return {
            "name": self._model_name,
            "description": info.get("description", "OPS custom model"),
            "context_length": self._max_context_length,
            "path": self._model_path or info.get("path"),
            "initialized": self._initialized,
        }

    def _build_prompt(self, messages: list[dict]) -> str:
        """Build LLaMA prompt from messages."""
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{msg['content']}<|eot_id|>\n"
            elif msg["role"] == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n{msg['content']}<|eot_id|>\n"
            elif msg["role"] == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{msg['content']}<|eot_id|>\n"
        return prompt

    def _mock_response(self, prompt: str, tools: list[dict] | None = None) -> dict:
        """Mock response for testing without model."""
        return {
            "content": f"[Mock] Response to: {prompt[:100]}...",
            "finish_reason": "stop",
            "tool_calls": None,
        }

    def _mock_response(self, prompt: str, tools: list[dict] | None = None) -> dict:
        """Mock response for testing without model."""
        return {
            "content": f"[Mock] Response to: {prompt[:100]}...",
            "finish_reason": "stop",
            "tool_calls": None,
        }
