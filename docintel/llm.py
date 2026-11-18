"""Optional LLM backend: Google Gemini via REST (free tier works, no SDK needed).
Without GEMINI_API_KEY the system runs in deterministic extractive mode."""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LLM:
    def __init__(self):
        self.key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = os.getenv("LLM_MODEL", "gemini-3.8-flash")
        # free tier is rate limited (~10 requests/min): space calls out
        self.min_interval = float(os.getenv("LLM_MIN_INTERVAL", "6"))
        self.timeout = float(os.getenv("LLM_TIMEOUT", "180"))
        # less thinking = faster, fewer timeouts. Set LLM_THINKING_LEVEL="" to use the model default.
        self.thinking = os.getenv("LLM_THINKING_LEVEL", "low")
        self._last = 0.0

    @property
    def available(self) -> bool:
        return bool(self.key)

    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> str:
        use_thinking = bool(self.thinking)
        for attempt in range(5):
            gen = {"temperature": 0.2, "maxOutputTokens": max_tokens + 1500}
            if use_thinking:
                gen["thinkingConfig"] = {"thinkingLevel": self.thinking}
            body = json.dumps({
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": gen,
            }).encode()
            req = urllib.request.Request(
                URL.format(model=self.model), data=body,
                headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                data = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
                parts = data["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts).strip()
            except urllib.error.HTTPError as e:
                detail = e.read()[:400].decode(errors="replace")
                if e.code == 400 and use_thinking and "thinking" in detail.lower():
                    use_thinking = False  # model rejects this thinking level: retry without it
                    continue
                if e.code in (429, 500, 503) and attempt < 4:
                    time.sleep(10 * 2 ** min(attempt, 3))  # backoff on rate limit / overload
                    continue
                raise RuntimeError(f"Gemini API error {e.code}: {detail}") from e
            except (TimeoutError, socket.timeout, urllib.error.URLError) as e:
                if attempt < 4:  # timeouts / dropped connections: retry
                    time.sleep(5)
                    continue
                raise RuntimeError(f"Gemini request failed: {e}") from e
        raise RuntimeError("Gemini API: retries exhausted")
