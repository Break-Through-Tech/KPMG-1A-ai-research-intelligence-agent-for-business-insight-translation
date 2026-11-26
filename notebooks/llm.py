"""Thin wrapper around the local Ollama model so the LLM is swappable in one place."""
import ollama

LLM_MODEL = "qwen2.5:3b"


def generate(prompt: str, system: str = "", model: str = None, temperature: float = 0.1) -> str:
    model = model or LLM_MODEL  # read at call time so scripts can override llm.LLM_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = ollama.chat(
        model=model, messages=messages, options={"temperature": temperature, "num_ctx": 4096}
    )
    return resp["message"]["content"].strip()
