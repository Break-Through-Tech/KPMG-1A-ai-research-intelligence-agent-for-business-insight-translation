"""Thin wrapper around the local Ollama model so the LLM is swappable in one place."""
import ollama

LLM_MODEL = "qwen2.5:3b"


def generate(prompt: str, system: str = "", model: str = LLM_MODEL, temperature: float = 0.1) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = ollama.chat(
        model=model, messages=messages, options={"temperature": temperature, "num_ctx": 4096}
    )
    return resp["message"]["content"].strip()
