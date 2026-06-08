"""
model.py — Text generation.
Uses Modal GPU (Qwen3-4B) when MODAL_TOKEN_ID/SECRET are set,
falls back to local llama-cpp GGUF otherwise.
"""
import os
from functools import lru_cache

MODEL_REPO = os.getenv("MODEL_REPO", "Qwen/Qwen3-1.7B-GGUF")
MODEL_FILE = os.getenv("MODEL_FILE", "Qwen3-1.7B-Q8_0.gguf")
N_CTX = int(os.getenv("N_CTX", "4096"))
N_THREADS = int(os.getenv("N_THREADS", str(os.cpu_count() or 4)))
MODAL_APP = os.getenv("MODAL_APP_NAME", "storyforge")


def _modal_ready() -> bool:
    return bool(os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"))


@lru_cache(maxsize=1)
def _local_llm():
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return Llama(model_path=path, n_ctx=N_CTX, n_threads=N_THREADS, verbose=False)


def generate(system: str, user: str, max_tokens: int = 512) -> str:
    if _modal_ready():
        try:
            import modal

            TextModel = modal.Cls.from_name(MODAL_APP, "TextModel")
            return TextModel().generate.remote(system, user, max_tokens)
        except Exception as e:
            import traceback
            print(f"[model] Modal call failed, falling back to local: {e}")
            traceback.print_exc()
    # Local fallback
    llm = _local_llm()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.8,
        top_p=0.9,
    )
    return out["choices"][0]["message"]["content"]
