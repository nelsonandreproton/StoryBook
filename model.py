"""
Model layer. Small GGUF model via llama.cpp, running IN-PROCESS on CPU.
No cloud APIs -> "Off the Grid". Swap MODEL_REPO/MODEL_FILE for whatever you
benchmark best on day 1.
"""

import os
from functools import lru_cache

MODEL_REPO = os.getenv("MODEL_REPO", "Qwen/Qwen3-1.7B-GGUF")
MODEL_FILE = os.getenv("MODEL_FILE", "Qwen3-1.7B-Q8_0.gguf")
N_CTX = int(os.getenv("N_CTX", "4096"))
N_THREADS = int(os.getenv("N_THREADS", str(os.cpu_count() or 4)))


@lru_cache(maxsize=1)
def _load():
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return Llama(model_path=path, n_ctx=N_CTX, n_threads=N_THREADS, verbose=False)


def generate(system: str, user: str, max_tokens: int = 512) -> str:
    llm = _load()
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
