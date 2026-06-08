"""
Push StoryBook to HF Space via git.

Usage (Windows PowerShell):
    $env:HF_TOKEN="hf_..."
    python3 upload_space.py
"""
import os
import subprocess

TOKEN = os.environ["HF_TOKEN"]
REPO  = "nelsonandreproton/storybook"
REMOTE_URL = f"https://user:{TOKEN}@huggingface.co/spaces/{REPO}"

# Add / update the hf-space remote
subprocess.run(["git", "remote", "remove", "space"], capture_output=True)
subprocess.run(["git", "remote", "add", "space", REMOTE_URL], check=True)

print(f"Pushing to {REPO} ...")
subprocess.run(["git", "push", "space", "master", "--force"], check=True)
print(f"Done! https://huggingface.co/spaces/{REPO}")
