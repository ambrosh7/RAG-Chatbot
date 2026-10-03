const DISCLAIMER =
  "Answers come from the seven official guidance documents. This page is a reading of those documents.";

const EXAMPLES = [
  "How long can I keep a whole chicken in the fridge?",
  "What do the documents say about cooking oil?",
  "According to the Eatwell Guide, how much of the diet should be fruit and vegetables?",
  "What should I weigh?",
];

const FOREST_SPINES = new Set(["cold-food-storage", "kitchen-companion"]);

const API_BASE = String(import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

const disclaimer = document.querySelector("#disclaimer");
const documentSelect = document.querySelector("#document");
const indexList = document.querySelector("#index");
const examples = document.querySelector("#examples");
const stamp = document.querySelector("#stamp");
const banner = document.querySelector("#banner");
const thread = document.querySelector("#thread");
const empty = document.querySelector("#empty");
const composer = document.querySelector("#composer");
const messageBox = document.querySelector("#message");
const sendButton = document.querySelector("#send");
const statusLine = document.querySelector("#status");

/** @type {string[]} */
let registryOrder = [];
/** @type {Map<string, string>} */
const names = new Map();

disclaimer.textContent = DISCLAIMER;

if (import.meta.env.PROD && !API_BASE) {
  showBanner("Set VITE_API_BASE_URL to the Railway URL in the Vercel project, then redeploy this page.");
}

for (const prompt of EXAMPLES) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = prompt;
  button.addEventListener("click", () => {
    messageBox.value = prompt;
    composer.requestSubmit();
  });
  examples.append(button);
}

documentSelect.addEventListener("change", () => markIndex(documentSelect.value));

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = messageBox.value.trim();
  if (!message || sendButton.disabled) {
    return;
  }
  ask(message);
});

loadDocuments();

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

function showBanner(text) {
  banner.hidden = false;
  banner.textContent = text;
}

async function loadDocuments() {
  try {
    const response = await fetch(apiUrl("/documents"));
    if (!response.ok) {
      throw new Error(String(response.status));
    }
    const rows = await response.json();
    if (!Array.isArray(rows)) {
      throw new Error("documents");
    }
    registryOrder = [];
    for (const row of rows) {
      if (!row || typeof row.document_id !== "string" || typeof row.document_name !== "string") {
        continue;
      }
      registryOrder.push(row.document_id);
      names.set(row.document_id, row.document_name);
      const option = document.createElement("option");
      option.value = row.document_id;
      option.textContent = row.document_name;
      documentSelect.append(option);

      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = row.document_name;
      button.dataset.id = row.document_id;
      button.addEventListener("click", () => {
        documentSelect.value = row.document_id;
        markIndex(row.document_id);
      });
      item.append(button);
      indexList.append(item);
    }
    stamp.textContent = `${registryOrder.length} of 7 records indexed`;
  } catch {
    stamp.textContent = "Register unavailable";
    showBanner("The document list could not be loaded. Questions still search all documents.");
  }
}

function markIndex(documentId) {
  for (const button of indexList.querySelectorAll("button")) {
    if (button.dataset.id === documentId) {
      button.setAttribute("aria-current", "true");
    } else {
      button.removeAttribute("aria-current");
    }
  }
}

function scopeLabel(documentId) {
  if (!documentId) {
    return "All documents";
  }
  return names.get(documentId) || documentId;
}

async function ask(message) {
  const documentId = documentSelect.value || null;
  const scope = scopeLabel(documentId);
  appendQuery(message, scope);
  messageBox.value = "";
  setBusy(true);
  try {
    const response = await fetch(apiUrl("/chat"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, document_id: documentId }),
    });
    if (response.status === 422) {
      appendNotice("error", "Chat API", "That question could not be sent.");
      return;
    }
    if (!response.ok) {
      appendNotice("error", "Chat API", "The chat API could not be reached.");
      return;
    }
    appendReply(await response.json());
  } catch {
    appendNotice("error", "Chat API", "The chat API could not be reached.");
  } finally {
    setBusy(false);
    messageBox.focus();
  }
}

function setBusy(busy) {
  sendButton.disabled = busy;
  sendButton.textContent = busy ? "Searching…" : "Ask";
  statusLine.hidden = !busy;
  for (const button of examples.querySelectorAll("button")) {
    button.disabled = busy;
  }
}

function appendQuery(message, scope) {
  empty.hidden = true;
  const item = document.createElement("li");
  item.className = "query";
  const label = document.createElement("p");
  label.className = "slug";
  label.textContent = scope;
  const heading = document.createElement("h2");
  heading.textContent = `“${message}”`;
  item.append(label, heading);
  thread.append(item);
  item.scrollIntoView({ block: "nearest" });
}

function appendReply(body) {
  const kind = body && body.type;
  if (kind === "answer") {
    const sections = orderedSections(body.sections);
    if (!sections) {
      appendNotice("error", "Chat API", "The chat API returned an unexpected response.");
      return;
    }
    const item = document.createElement("li");
    item.dataset.kind = "answer";
    const stack = document.createElement("div");
    stack.className = "dossier-stack";
    for (const section of sections) {
      stack.append(renderSection(section));
    }
    item.append(stack);
    thread.append(item);
    item.scrollIntoView({ block: "nearest" });
    return;
  }
  if (kind === "not_in_corpus") {
    const searched = searchedRows(body.searched);
    if (typeof body.message !== "string" || !searched) {
      appendNotice("error", "Chat API", "The chat API returned an unexpected response.");
      return;
    }
    const item = noticeItem("not_in_corpus", "Not covered by the guidance", body.message);
    const title = document.createElement("p");
    title.className = "reply-title searched-title";
    title.textContent = "Guidance searched";
    const list = document.createElement("ul");
    list.className = "searched";
    for (const row of searched) {
      const entry = document.createElement("li");
      entry.textContent = `${row.document_name} — ${row.publisher}, ${row.year}`;
      list.append(entry);
    }
    item.append(title, list);
    thread.append(item);
    item.scrollIntoView({ block: "nearest" });
    return;
  }
  if (kind === "out_of_scope" && typeof body.message === "string") {
    const item = noticeItem("out_of_scope", "Outside this assistant", body.message);
    thread.append(item);
    item.scrollIntoView({ block: "nearest" });
    return;
  }
  if (kind === "generation_unavailable" && typeof body.message === "string") {
    const item = noticeItem("generation_unavailable", "Answer unavailable", body.message);
    thread.append(item);
    item.scrollIntoView({ block: "nearest" });
    return;
  }
  appendNotice("error", "Chat API", "The chat API returned an unexpected response.");
}

function orderedSections(raw) {
  if (!Array.isArray(raw) || raw.length === 0) {
    return null;
  }
  const sections = [];
  for (const entry of raw) {
    if (!entry || typeof entry.document_name !== "string" || !Array.isArray(entry.claims) || !entry.claims.length) {
      return null;
    }
    const claims = [];
    for (const claim of entry.claims) {
      if (!claim || typeof claim.text !== "string" || typeof claim.section_heading !== "string") {
        return null;
      }
      claims.push(claim);
    }
    if (typeof entry.publisher !== "string" || typeof entry.year !== "number" || typeof entry.source_url !== "string") {
      return null;
    }
    sections.push({ ...entry, claims });
  }
  sections.sort((left, right) => rank(left.document_id) - rank(right.document_id));
  return sections;
}

function rank(documentId) {
  const index = registryOrder.indexOf(documentId);
  return index === -1 ? registryOrder.length : index;
}

function searchedRows(raw) {
  if (!Array.isArray(raw)) {
    return null;
  }
  const rows = [];
  for (const entry of raw) {
    if (!entry || typeof entry.document_name !== "string" || typeof entry.publisher !== "string" || typeof entry.year !== "number") {
      return null;
    }
    rows.push(entry);
  }
  return rows;
}

function renderSection(section) {
  const block = document.createElement("article");
  block.className = "dossier";
  const spine = document.createElement("div");
  spine.className = FOREST_SPINES.has(section.document_id) ? "spine-forest" : "spine-slate";
  const body = document.createElement("div");
  body.className = "dossier-body";
  const label = document.createElement("p");
  label.className = "slug";
  label.textContent = "Document dossier";
  const heading = document.createElement("h3");
  heading.textContent = section.document_name;
  body.append(label, heading);
  for (const claim of section.claims) {
    const text = document.createElement("p");
    text.className = "claim";
    text.textContent = claim.text;
    const citation = document.createElement("p");
    citation.className = "citation";
    citation.append(document.createTextNode(`${section.document_name} · ${section.publisher} · ${section.year}`));
    citation.append(document.createElement("br"));
    citation.append(document.createTextNode(claim.section_heading));
    citation.append(document.createElement("br"));
    const link = document.createElement("a");
    link.textContent = "Source";
    if (/^https?:\/\//.test(section.source_url)) {
      link.href = section.source_url;
      link.target = "_blank";
      link.rel = "noreferrer";
    }
    citation.append(link);
    body.append(text, citation);
  }
  block.append(spine, body);
  return block;
}

function noticeItem(kind, title, message) {
  const item = document.createElement("li");
  item.className = `notice kind-${kind}`;
  item.dataset.kind = kind;
  const heading = document.createElement("p");
  heading.className = "reply-title";
  heading.textContent = title;
  const body = document.createElement("p");
  body.textContent = message;
  item.append(heading, body);
  return item;
}

function appendNotice(kind, title, message) {
  const item = noticeItem(kind, title, message);
  thread.append(item);
  item.scrollIntoView({ block: "nearest" });
}
