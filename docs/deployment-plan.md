# Deployment plan

Deploy the dietary guidance chatbot as one Streamlit Community Cloud app. The page, the scope guard, retrieval, and Groq claims all run in that process. The FastAPI process stays the local entry point. Community Cloud does not publish a second public port, so the deployed page calls `answer()` directly instead of `POST /chat`.

The published Chroma index is about 6 MB. The chat path reads that index. It does not fetch or parse the corpus on the server.

---

## What ships

| Piece | Where it runs | How it gets there |
| --- | --- | --- |
| Chat page | Streamlit Community Cloud | `src/ui/app.py` is the app entry |
| Scope guard, retrieval, citations, validation | Same process | `src.rag.pipeline.answer` |
| Groq claims | Groq's API, called from the app | `GROQ_API_KEY` in Streamlit secrets |
| Embedding model `BAAI/bge-small-en-v1.5` | Same process, loaded on the first search | Downloaded from Hugging Face at runtime |
| Chroma collection | Same process, read-only | `data/index/` committed with the repo |
| `POST /chat` | Local only | `uvicorn src.api.main:app` |

Out-of-scope questions still return before retrieval. A miss still names the documents that were searched. A Groq failure is still `generation_unavailable`.

---

## Before the first deploy

The page, the deploy requirements, and the gitignore exception are already in the repo. Do the commit and the Cloud steps on a machine that already has the index (`GET /health` returns ok).

### 1. How the page chooses a backend

`src/ui/app.py` is the Community Cloud entry.

- A blank API URL calls `answer(message, document_id)` in this process and passes that dict to `present()`. Community Cloud leaves `CHAT_API_BASE_URL` unset, so the field starts blank.
- A non-blank URL still calls `POST /chat`. `python scripts/run_ui.py` sets `CHAT_API_BASE_URL` to `http://127.0.0.1:8000` when it is not already set, so the local two-process run is unchanged.
- `apply_streamlit_secrets()` copies `GROQ_API_KEY` and `GROQ_MODEL` from Streamlit secrets into the environment when those variables are empty. The Groq client reads the environment.
- The embedding model still loads on the first `embed()` call inside `BgeEmbedder`.

### 2. Put the published index in the repo

`data/index/` is no longer ignored. `data/raw/`, `data/chunks/`, and `.env` stay ignored. The running app does not read the raw files or the chunk file.

- Commit `data/index/chroma.sqlite3`, the segment files beside it, and `data/index/index_manifest.json`.
- Confirm `index_is_ready()` is true from a clean checkout: the directory exists, `chroma.sqlite3` is present, and the manifest is present.

Rebuild the index locally with `python -m scripts.build_index` when the corpus changes, then commit the new `data/index/`. Do not rebuild it during the Cloud boot. Fetching the seven documents and embedding them will exceed the boot time and the memory budget.

### 3. Cloud dependency list

`requirements.txt` stays the local install. It includes parsers and pytest, which the running chat does not import. Community Cloud should install `requirements-deploy.txt` instead. The PyTorch wheel behind `sentence-transformers` is the memory risk.

`requirements-deploy.txt` pins:

- `streamlit`
- `chromadb`
- `sentence-transformers`
- `groq`
- `python-dotenv`
- `httpx`

`fastapi` and `uvicorn` stay in the local `requirements.txt`. The deployed page does not serve HTTP.

`runtime.txt` contains `python-3.12` so Cloud matches the local 3.12 environment.

In the Community Cloud app settings, set the requirements file to `requirements-deploy.txt` if the UI offers that field. If it only reads `requirements.txt`, point the app at a branch whose `requirements.txt` is the deploy list, or temporarily make `requirements.txt` match `requirements-deploy.txt` for the deploy commit.

### 4. Secrets

`.streamlit/secrets.toml` is gitignored. Commit `.streamlit/secrets.toml.example`, which has empty values:

```toml
GROQ_API_KEY = ""
GROQ_MODEL = "openai/gpt-oss-120b"
```

On Community Cloud, paste the same keys into the app's Secrets box. Do not commit a real key. If a key was ever written into `.env.example` or `.env` and that file is pushed, revoke it in the Groq console and create a new one before the repo is public.

`GROQ_MODEL` may be omitted. The code then uses `GROQ_DEFAULT_MODEL` in `src/constants.py`.

### 5. App entry for Community Cloud

Community Cloud asks for a main file path. Use `src/ui/app.py`.

That file inserts the repo root on `sys.path`, so `import src...` works when Streamlit runs it directly. No `packages.txt` is required for the chat path: the index is prebuilt, so PyMuPDF and pdfplumber are not imported.

---

## Deploy on Streamlit Community Cloud

1. Push the repo to GitHub, including `data/index/` and excluding `.env` and `.streamlit/secrets.toml`.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with the GitHub account that can see the repo.
3. Create an app:
   - Repository: this repo
   - Branch: the branch that contains the index and the in-process page change
   - Main file path: `src/ui/app.py`
   - Python version: 3.12 (`runtime.txt`)
4. Open Advanced settings and set the secrets from step 4.
5. Deploy and wait for the first boot. The first install pulls PyTorch and `sentence-transformers`. That boot is several minutes.
6. Open the app URL Streamlit prints. The page should show the disclaimer, the document picker, and the four example prompts.

The app sleeps after a period with no visitors. The next visit boots again and downloads the embedding model again unless the container is still warm. The first question after a cold start is the slow one, because that is when `BAAI/bge-small-en-v1.5` loads.

---

## Smoke test on the deployed URL

Run these in the page. Judge the reply on screen. The HTTP body is the same object `answer()` returns locally.

| Prompt | Picker | Expected on screen |
| --- | --- | --- |
| How long can I keep a whole chicken in the fridge? | All documents | An answer. A Cold Food Storage section whose source link is the registry URL for `cold-food-storage`. Refrigerator and freezer values appear in the claim. |
| What do the documents say about cooking oil? | All documents | One block per document that produced a claim. Each source link belongs to that block's document. |
| According to the Eatwell Guide, how much of the diet should be fruit and vegetables? | All documents, or the Eatwell Guide | Searched scope is only the Eatwell Guide when the picker is that document. No other document appears as a section. |
| What is the tariff on imported olive oil? | All documents | "Not covered by the guidance", then all seven documents by name, publisher, and year. |
| What should I weigh? | All documents | "Outside this assistant", the professional referral, and no searched list. |
| How much protein is in 100 g of chicken? | All documents | The nutrient-lookup sentence, and no searched list. |

Then set the picker to one document and repeat the tariff question. The searched list names only that document.

If the app shows "The chat API could not be reached", the page is still posting to localhost. Step 1 is not in the deployed commit.

---

## Memory

Community Cloud gives a small container. Importing `sentence-transformers` loads PyTorch. That is the likely failure, and it shows up as an app that restarts or dies during the first question.

If the logs show an out-of-memory kill:

1. Confirm `requirements-deploy.txt` does not include `pymupdf`, `pdfplumber`, `pytest`, or a second copy of the embedding stack.
2. Confirm the app does not call `build_index` or import the parser on startup.
3. Retry once. A cold install is larger than a warm process.

If it still dies, move the same container to a host with at least 2 GB RAM. Use one process there too: `streamlit run src/ui/app.py --server.port $PORT --server.address 0.0.0.0`, with `GROQ_API_KEY` in the host's environment and `data/index/` in the image. Fly.io, Railway, and Render all run that command. The page code does not change.

Run uvicorn beside Streamlit only on a machine where both processes are wanted for local development. Do not start uvicorn inside the Community Cloud app.

---

## What this deploy does not do

- It does not rebuild the corpus on a schedule. A new index is a local `python -m scripts.build_index` and a new commit of `data/index/`.
- It does not expose `POST /chat` on the public URL. Local clients keep using uvicorn.
- It does not store chat history. History lives in the browser session.
- It does not add per-food nutrient data. Those questions stay `nutrient_lookup`.

---

## Exit criteria

- A fresh GitHub clone contains `data/index/chroma.sqlite3` and `data/index/index_manifest.json`.
- The Community Cloud app opens `src/ui/app.py` and the secrets box has `GROQ_API_KEY`.
- The six smoke-test rows match the expected screen.
- The deployed logs do not contain raw Groq JSON. The pipeline already logs scope, hit count, refusal reason, and dropped-claim reason codes only.
- `.env` and `.streamlit/secrets.toml` are not in the repository.
