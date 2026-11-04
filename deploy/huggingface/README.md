---
title: RAGLens
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Banking-policy Q&A that cites its evidence and knows when to refuse
---

# RAGLens live demo

This folder holds the files for a free [Hugging Face Space](https://huggingface.co/spaces).
The Space runs the Streamlit UI with the API embedded in the same process, on the banking
corpus, with the offline hash embedder and the deterministic FakeLLM, so it needs no GPU and
no API keys.

## Deploy (about 5 minutes)

1. Create a free account at huggingface.co, then **New Space** → name `raglens` → SDK **Docker**
   → template **Blank** → hardware **CPU basic (free)** → Create.
2. Clone the empty Space and copy the project into it:

   ```powershell
   git clone https://huggingface.co/spaces/<your-username>/raglens hf-raglens
   robocopy raglens hf-raglens /E /XD .git .venv data\index data\logs
   copy raglens\deploy\huggingface\Dockerfile hf-raglens\Dockerfile
   copy raglens\deploy\huggingface\README.md hf-raglens\README.md
   cd hf-raglens
   git add .
   git commit -m "Deploy RAGLens demo"
   git push
   ```

   Git asks for your Hugging Face username and an access token (Settings → Access Tokens →
   create one with *write* permission) as the password.
3. Wait for the build log to show "Running", then open the Space URL and put it in your CV and
   the GitHub README.

The Space's README must start with the YAML block above; that is why this file replaces the
project README in the Space repository.
