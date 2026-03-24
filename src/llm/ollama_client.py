import os
from dataclasses import dataclass

import requests


@dataclass
class OllamaClient:
    model_name: str
    base_url: str = "http://localhost:11434"
    timeout_s: int = 120

    def generate(self, prompt: str, *, model_name: str = None) -> str:
        model = model_name or self.model_name

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        resp = requests.post(url, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()

    def generate_with_options(
        self,
        prompt: str,
        *,
        options: dict,
        model_name: str = None,
    ) -> str:
        """
        Ollama supports sampling controls via the `options` field.
        See: https://docs.ollama.com/api/generate
        """
        model = model_name or self.model_name
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        resp = requests.post(url, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()


def get_ollama_client(model_name: str) -> OllamaClient:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return OllamaClient(model_name=model_name, base_url=base_url)

