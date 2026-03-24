import json as _json
import os
from dataclasses import dataclass, field
from typing import Generator

import requests


@dataclass
class OpenRouterClient:
    api_key: str
    model_name: str = "meta-llama/llama-3-8b-instruct"
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    timeout_s: int = 120
    default_options: dict = field(default_factory=dict)

    def _call(self, prompt: str, options: dict | None = None) -> str:
        merged = {**self.default_options, **(options or {})}

        messages = [{"role": "user", "content": prompt}]

        payload: dict = {
            "model": self.model_name,
            "messages": messages,
        }
        if "temperature" in merged:
            payload["temperature"] = float(merged["temperature"])
        if "top_p" in merged:
            payload["top_p"] = float(merged["top_p"])
        if "frequency_penalty" in merged:
            payload["frequency_penalty"] = float(merged["frequency_penalty"])
        if "presence_penalty" in merged:
            payload["presence_penalty"] = float(merged["presence_penalty"])
        if "seed" in merged:
            payload["seed"] = int(merged["seed"])

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/draft-to-ready-writing-agent",
            "X-Title": "Draft-to-Ready Writing Agent",
        }

        resp = requests.post(
            self.base_url, json=payload, headers=headers, timeout=self.timeout_s
        )
        resp.raise_for_status()
        data = resp.json()

        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

    def generate(self, prompt: str, *, model_name: str | None = None) -> str:
        return self._call(prompt)

    def generate_with_options(
        self, prompt: str, *, options: dict, model_name: str | None = None
    ) -> str:
        return self._call(prompt, options=options)

    def generate_stream(
        self, prompt: str, *, options: dict | None = None
    ) -> Generator[str, None, None]:
        """Yield text chunks as they arrive via SSE streaming."""
        merged = {**self.default_options, **(options or {})}
        payload: dict = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        if "temperature" in merged:
            payload["temperature"] = float(merged["temperature"])
        if "top_p" in merged:
            payload["top_p"] = float(merged["top_p"])
        if "frequency_penalty" in merged:
            payload["frequency_penalty"] = float(merged["frequency_penalty"])
        if "presence_penalty" in merged:
            payload["presence_penalty"] = float(merged["presence_penalty"])
        if "seed" in merged:
            payload["seed"] = int(merged["seed"])

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/draft-to-ready-writing-agent",
            "X-Title": "Draft-to-Ready Writing Agent",
        }
        resp = requests.post(
            self.base_url, json=payload, headers=headers, timeout=self.timeout_s, stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = _json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    yield delta
            except (_json.JSONDecodeError, IndexError, KeyError):
                continue


def get_openrouter_client(
    model_name: str | None = None,
    api_key: str | None = None,
) -> OpenRouterClient:
    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    model = model_name or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")
    if not key:
        raise ValueError("OPENROUTER_API_KEY not set")
    return OpenRouterClient(api_key=key, model_name=model)
