# Deployment plan

The public app is two hosts. Railway runs the FastAPI chat API, the embedding model, and the published Chroma index. Vercel serves the chat page in `web/` and calls that API from the browser.

Streamlit stays a local page (`python scripts/run_ui.py`). It is not the public deploy.

The published index is about 6 MB. The API reads `data/index/`. It does not fetch or parse the corpus on the server.

---

## What ships

| Piece | Where it runs | How it gets there |
| --- | --- | --- |
| Chat page | Vercel | `web/` built with Vite |
| `POST /chat`, `GET /health`, `GET /documents` | Railway | `uvicorn src.api.main:app` from `start.sh` |
| Scope guard, retrieval, citations, validation | Railway | `src.rag.pipeline.answer` |
| Groq claims | Groq's API, called from Railway | `GROQ_API_KEY` in the Railway service |
| Embedding model `BAAI/bge-small-en-v1.5` | Railway image | Downloaded during the Docker build into `/opt/hf` |
| Chroma collection | Railway image, read-only | `data/index/` copied by the Dockerfile |

Out-of-scope questions still return before retrieval. A miss still names the documents that were searched. A Groq failure is still `generation_unavailable`.

---

## Railway

1. Push this repo to GitHub, including `data/index/`, `Dockerfile`, `start.sh`, `railway.toml`, and `requirements-api.txt`. Leave `.env` out.
2. In Railway, create a project from that GitHub repo. The `railway.toml` file selects the Dockerfile. Do not set a start command. `start.sh` reads `PORT` and binds `0.0.0.0`.
3. Set the service variable `GROQ_API_KEY` to the Groq key. `GROQ_MODEL` is optional. `CORS_ORIGINS` is optional and defaults to `*`, which is enough because the chat request does not use cookies. To limit browsers to the Vercel site, set `CORS_ORIGINS` to that site's origin after Vercel prints it, then redeploy the API.
4. Give the service at least 2 GB of RAM. The image includes PyTorch and the embedding model.
5. Deploy. The first build installs the CPU torch wheel and the model, so it takes several minutes. Railway then calls `GET /health`.
6. Copy the public Railway URL, for example `https://your-service.up.railway.app`. Open `/health` in a browser. It should return `{"status":"ok"}`.

`requirements-api.txt` is the image install. It does not include Streamlit, pytest, or the PDF parsers. `requirements.txt` stays the local install.

Do not rebuild the index during the container boot. When the corpus changes, run `python -m scripts.build_index` locally and commit the new `data/index/`.

---

## Vercel

1. Import the same GitHub repo. The root `vercel.json` installs and builds `web/` and publishes `web/dist`.
2. Set the project environment variable `VITE_API_BASE_URL` to the Railway URL, with `https` and no trailing slash. Vite reads it at build time, so change it and redeploy whenever the Railway URL changes.
3. Deploy. The page should show the disclaimer, the document picker (filled from `GET /documents`), and the four example prompts.

Locally, leave `VITE_API_BASE_URL` empty and run `npm run dev` inside `web/`. Vite proxies `/chat`, `/health`, and `/documents` to `http://127.0.0.1:8000`. Start the API first:

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

---

## Smoke test on the Vercel URL

| Prompt | Picker | Expected on screen |
| --- | --- | --- |
| How long can I keep a whole chicken in the fridge? | All documents | An answer. A Cold Food Storage section whose source link is the registry URL for `cold-food-storage`. |
| What do the documents say about cooking oil? | All documents | One block per document that produced a claim. Each source link belongs to that block's document. |
| According to the Eatwell Guide, how much of the diet should be fruit and vegetables? | All documents, or the Eatwell Guide | When the picker is the Eatwell Guide, no other document appears as a section. |
| What is the tariff on imported olive oil? | All documents | "Not covered by the guidance", then the searched documents by name, publisher, and year. |
| What should I weigh? | All documents | "Outside this assistant", the professional referral, and no searched list. |
| How much protein is in 100 g of chicken? | All documents | The nutrient-lookup sentence, and no searched list. |

The first question after a new Railway container starts can sit on "Searching…" while the model loads from the image cache. Later questions are faster.

If the page says the chat API could not be reached, `VITE_API_BASE_URL` was empty at build time or the Railway URL is wrong. Set the variable and redeploy Vercel.

---

## What this deploy does not do

- It does not rebuild the corpus on a schedule.
- It does not put Streamlit on Vercel. Vercel serves the static page in `web/`.
- It does not store chat history. History lives in the browser tab.
- It does not add per-food nutrient data. Those questions stay `nutrient_lookup`.
