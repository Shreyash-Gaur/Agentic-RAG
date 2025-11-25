# backend/tools/ollama_client.py
import os
import requests
from requests.adapters import HTTPAdapter, Retry
from typing import Optional

class OllamaClient:
    def __init__(self, base_url: Optional[str] = None, timeout: int = 120):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = timeout
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def generate(self, model: str, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """
        Try POST /api/models/{model}/generate. Be defensive about response shapes.
        Returns plain text (string) if successful, else raises.
        """
        # prefer models endpoint
        url = f"{self.base_url}/api/models/{model}/generate"
        payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        try:
            r = self.session.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            # typical shapes: {"generated": "..."} or {"text": "..."} or {"data":[{"content":"..."}]}
            if isinstance(data, dict):
                if "generated" in data:
                    return data["generated"]
                if "text" in data:
                    return data["text"]
                if "data" in data and isinstance(data["data"], list):
                    # drill down for content/text
                    for item in data["data"]:
                        if isinstance(item, dict):
                            for k in ("content", "text", "generated"):
                                if k in item:
                                    return item[k]
                    return str(data)
            return str(data)
        except requests.RequestException as e:
            # fallback: try older endpoint or bubble up informative error
            # try generic /api/generate (some versions)
            try:
                alt_url = f"{self.base_url}/api/generate"
                r2 = self.session.post(alt_url, json={"model": model, "prompt": prompt}, timeout=self.timeout)
                r2.raise_for_status()
                d2 = r2.json()
                if isinstance(d2, dict) and "text" in d2:
                    return d2["text"]
                return str(d2)
            except Exception:
                raise RuntimeError(f"Ollama generate failed: {e}") from e
