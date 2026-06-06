from __future__ import annotations

import json
import os
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/api")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

class LLMError(RuntimeError):
    pass

def ollama_chat(messages : list[dict], *, schema : dict | None = None, temperature : float, timeout : int = 120) -> str:
    payload = {
        "model" : OLLAMA_MODEL,
        "messages" : messages,
        "stream" : False,
        "options" : {
            "temperature" : temperature,
        }
    }

    if schema is not None:
        payload["format"] = schema

    resp = requests.post(f"{OLLAMA_BASE_URL}/chat", json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise LLMError(f"Ollama error {resp.status_code} : {resp.text[:666]}")

    data = resp.json()
    return data["message"]["content"]

def ollama_json(messages : list[dict], schema : dict, *, temperature : float, timeout : int = 120) -> dict:
    raw = ollama_chat(messages, schema=schema, temperature=temperature, timeout=timeout)
    return json.loads(raw)
