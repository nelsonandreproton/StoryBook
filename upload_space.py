"""
Upload StoryBook to your HF Space.

Usage:
    HF_TOKEN=hf_... python3 upload_space.py
"""
import os
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
REPO  = "nelsonandreproton/storybook"

api = HfApi()

url = api.upload_folder(
    folder_path=".",
    repo_id=REPO,
    repo_type="space",
    token=TOKEN,
    ignore_patterns=[
        "upload_space.py",
        ".git*",
        "__pycache__",
        "*.gguf",
        "*.bin",
        "*.safetensors",
        ".env",
        "*.pyc",
        "test_image*",
        "*.egg-info",
    ],
)
print("Done:", url)
