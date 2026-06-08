"""
Upload StoryBook to your HF Space.

Usage (Windows PowerShell):
    $env:HF_TOKEN="hf_..."
    python3 upload_space.py
"""
import os
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
REPO  = "nelsonandreproton/storybook"

IGNORE = [
    "upload_space.py",
    ".git",
    ".git*",
    "__pycache__",
    "**/__pycache__",
    "*.gguf",
    "*.bin",
    "*.safetensors",
    "*.pt",
    "*.pth",
    ".env",
    "*.pyc",
    "**/*.pyc",
    "test_image*",
    "*.egg-info",
    "**/*.egg-info",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "*.log",
    ".DS_Store",
    "Thumbs.db",
]

import huggingface_hub
huggingface_hub.login(token=TOKEN)

api = HfApi()

print(f"Uploading to {REPO} ...")
api.upload_large_folder(
    folder_path=".",
    repo_id=REPO,
    repo_type="space",
    ignore_patterns=IGNORE,
)
print("Done! Visit: https://huggingface.co/spaces/nelsonandreproton/storybook")
