# Edge Cases

This catalogue is the corner behaviour for [implementation-plan.md](./implementation-plan.md). Each row is a case the phase must get right. The expected result is the behaviour to build and to test. Cases are numbered so a test can cite them.

A case belongs to the phase that owns the decision. Later phases inherit it. Cross-phase rows at the end are the failures that only show up when two phases meet.

---

## How a case is decided

When two rules could apply, use this order:

1. An empty or whitespace-only message is rejected before the scope guard.
2. The scope guard runs before retrieval and before Groq. A match returns `out_of_scope` and an empty `searched` list.
3. If the guard matches more than one reason, the reason is the first hit in this order: `medical`, `weight_target`, `calorie_target`, `nutrient_lookup`.
4. An explicit `document_id` overrides alias detection. An id outside the registry is rejected before retrieval.
5. Alias detection that matches two documents leaves the search unfiltered.
6. Retrieval with no surviving hit returns `not_in_corpus` and does not call Groq.
7. A Groq failure returns `generation_unavailable`. That type is never rewritten into `not_in_corpus`.
8. The validator drops a bad claim. If every claim is dropped, the response becomes `not_in_corpus` for the scope that was searched.

Similarity at the threshold is kept. Hits strictly below `MIN_SIMILARITY` are dropped.

---

## Phase 0: Project setup

| ID | Case | Expected |
| --- | --- | --- |
| P0-01 | The registry has a document that is not one of the seven in the problem statement | Setup fails the exit check. Fetch, filter, and citation read only the registry. |
| P0-02 | A registry row is missing `document_id`, publisher, year, or `source_url` | The row is invalid. The process does not start a fetch. |
| P0-03 | `landing_page` is true for a document other than `fao-who-healthy-diets` and `who-five-keys` | The exit check fails. Only those two publication pages are followed to a PDF. |
| P0-04 | `kind` disagrees with the source: an HTML chart marked `pdf`, or a direct PDF marked `html` | The fetcher would store the wrong extension. `cold-food-storage` and `fsa-chill` are `html`. The other five are `pdf`. |
| P0-05 | Two registry rows share a `document_id` | The allowlist is ambiguous. Setup fails. |
| P0-06 | The same alias is listed on two documents | A question naming that alias must not pick one document silently. Leave the search unfiltered when the alias is shared. |
| P0-07 | `MIN_SIMILARITY` is still the phase 0 placeholder `0.45` when evaluation runs | Phase 9 does not pass. Phase 4 replaces the placeholder from probes and records the bounding questions. |
| P0-08 | `GROQ_MODEL` is set in the environment | That value is the model used. When it is unset, `GROQ_DEFAULT_MODEL` is used. Temperature stays 0. |
| P0-09 | `scripts/build_index.py` is run before phase 3 wires the pipeline | The script exits with a clear not-implemented message. It does not write a partial index. |
| P0-10 | `.env` is present in the work tree | It stays out of version control. `.env.example` contains the key names and no secret. |

---

## Phase 1: Corpus fetch

| ID | Case | Expected |
| --- | --- | --- |
| P1-01 | The host returns 403 to a client with no browser User-Agent | The fetcher sends a browser User-Agent and stores the document. A 403 body is not saved as a successful source. |
| P1-02 | The response is HTTP 200 and the body is a bot-challenge or error page | The row is not `ok`. The bytes are not treated as the guidance document. |
| P1-03 | The first download of a document fails and no previous file exists | The error names that `document_id`. No empty file is marked `ok`. The indexer will not publish. |
| P1-04 | A later download fails and a previous file exists | The previous bytes stay on disk. The manifest row becomes `stale`. The hash of the kept file is unchanged. |
| P1-05 | A re-run finds the same sha256 | That file is not rewritten. `retrieval_date` stays the date of the download that produced those bytes. |
| P1-06 | A re-run finds a new sha256 | The new bytes replace the file. `retrieval_date` is the UTC date of this download. The index fingerprint will no longer match, so phase 3 re-embeds. |
| P1-07 | Six documents succeed and one fails | The run stops with that `document_id`. The pipeline does not publish an index of a partial corpus. |
| P1-08 | `fao-who-healthy-diets` or `who-five-keys` returns a publication page with several links | The fetcher stores the English PDF. `source_url` remains the publication page. `downloaded_url` is the PDF that was saved. |
| P1-09 | A landing page offers a non-English PDF and an English PDF | The English PDF is the file on disk. |
| P1-10 | A landing-page follow receives HTML instead of a PDF | The row is not `ok`. HTML is not saved as `source.pdf`. |
| P1-11 | `cold-food-storage` or `fsa-chill` is an HTML guidance page that also contains incidental links | The saved file is that HTML page. The fetcher does not follow a link away from the page. `landing_page` is false for both. |
| P1-12 | Redirects change the final host | The stored bytes are the body after redirects. `downloaded_url` is the final URL. `source_url` stays the registry URL used in citations. |
| P1-13 | The `Content-Type` header disagrees with the file bytes | The row records the header. The extension follows `kind` and the sniffed body. A mismatch that means the body is not the expected PDF or HTML leaves the row not `ok`. |
| P1-14 | The body is empty | The row is not `ok`. |
| P1-15 | The download times out or the connection drops mid-body | The partial write is discarded. A previous good file, if any, is kept and the row is `stale`. |
| P1-16 | The machine clock is not UTC | `retrieval_date` is the UTC date, not the local date. |
| P1-17 | Two fetches write the manifest at the same time | The manifest is one complete JSON document. A crash mid-write does not leave a file that parses as seven successful rows when some are missing. |

---

## Phase 2: Parse and chunk

| ID | Case | Expected |
| --- | --- | --- |
| P2-01 | A heading has no body under it | The heading updates `heading_path` and is not emitted as a chunk. |
| P2-02 | Body text appears before any heading | The chunk still has a non-empty `section_heading`. Use the document name when no heading has been seen. |
| P2-03 | Headings nest three or four levels deep | `section_heading` is the path joined with ` > `, from the outermost heading to the nearest one. |
| P2-04 | The PDF text layer contains the same grid the table extractor already returned | That reading-order text is dropped. The grid is indexed once, as rows. |
| P2-05 | An HTML page contains navigation, a cookie banner, or a feedback widget | That text is stripped before chunking and does not appear in `chunks.jsonl`. |
| P2-06 | A Cold Food Storage data row has a refrigerator value and a freezer value | Both values are in that row’s chunk. A neighbouring food’s values are not. |
| P2-07 | A table uses a merged category cell, so the next data row leaves the food column blank | The row chunk still carries the category that applies to it, together with the column headers. |
| P2-08 | A table cell is empty | The labeled row omits that empty cell and keeps the cells that have values. The row is still one chunk. |
| P2-09 | A data row is longer than `CHUNK_TARGET_TOKENS` (200) | The row stays one chunk. It is not split on a sentence boundary. On this corpus the longest labeled row is 61 estimated tokens. Phase 3 then refuses the build if `embed_text` exceeds 512 model tokens. |
| P2-10 | A small table has only a few rows | Each data row is still its own chunk, with the header repeated. The whole grid is not also indexed. |
| P2-11 | The same storage times appear in Cold Food Storage and in Kitchen Companion | Both rows are kept. Each chunk carries its own `document_id`, publisher, and `source_url`. |
| P2-12 | A numbered recommendation is `1.`, `1)`, or `(1)` | The chunk text includes that number. The item is one chunk. |
| P2-13 | A list item has no number | It is still one `list_item` chunk. It is not merged with the next item. |
| P2-14 | A list item is only a few words | It stays its own chunk. Short length is the cost recorded for the README, not a reason to join items. |
| P2-15 | Consecutive paragraphs sit under one heading and together exceed 200 tokens | The pack breaks between paragraphs. No paragraph is cut to hit the target. No pack exceeds 200 estimated tokens unless it is a single paragraph. |
| P2-16 | A paragraph is at or under 200 estimated tokens | It is not sentence-split. The longest paragraph in the parsed corpus is 162 tokens, so this is the path the seven documents take. |
| P2-17 | A single paragraph exceeds 200 estimated tokens | It splits on sentence boundaries. The pieces share `section_heading` and do not overlap. The second chunk does not repeat the tail of the first. |
| P2-18 | An oversized paragraph has no sentence boundary | The build stops with that block identified. The chunker does not cut the paragraph at an arbitrary token. |
| P2-19 | A list item or a table row is given overlap or is merged to reach 200 tokens | Neither happens. Overlap is not used. Packing applies only to consecutive paragraphs under one heading. |
| P2-20 | A paragraph pack reaches a new heading | The pack closes. The next chunk uses the new heading path. |
| P2-21 | A heading block is about to be written into `chunks.jsonl` | It is omitted. Ordinals count chunks, so headings do not consume a `chunk_id`. |
| P2-22 | `chunk_id` is built from the block index instead of the chunk sequence | Ids would skip or collide with headings. The id is `{document_id}:{ordinal}` over emitted chunks, starting at a stable zero-based or one-based sequence with no gaps. |
| P2-23 | A chunk’s `source_url` is the downloaded PDF asset rather than the registry URL | Replace it with the registry `source_url`. Citations use that URL, including for the two landing-page documents. |
| P2-24 | A chunk is missing document name, publisher, year, or section heading | It is not written to `chunks.jsonl`. |
| P2-25 | `embed_text` is the body alone | It is `{document_name} — {section_heading}\n{text}`, so a short storage row stays tied to its chart. |
| P2-26 | Page numbers, running headers, or footers are extracted as paragraphs | They are dropped and do not become chunks. |
| P2-27 | One of the seven documents yields zero content blocks | The pipeline does not publish an index that silently omits that document. The failure names the `document_id`. |

---

## Phase 3: Embed and index

| ID | Case | Expected |
| --- | --- | --- |
| P3-01 | Any `embed_text`, including special tokens, is over 512 tokens | The build stops before upsert and names the `chunk_id`. Nothing is written to the collection. The chunk is not truncated. |
| P3-02 | The document-name prefix is what pushes a chunk over 512 | Same as P3-01. The prefix stays. The chunker is fixed so the passage fits. |
| P3-03 | Passages are embedded with `BGE_QUERY_PREFIX` | The indexer embeds `embed_text` only. The prefix is for queries in phase 4. |
| P3-04 | `year` is stored as a string | Metadata stores `year` as an int. A round-trip returns an int. |
| P3-05 | A metadata field is null | Chroma metadata omits empty optional values rather than storing null. Required citation fields are present: `document_id`, `document_name`, `publisher`, `year`, `source_url`, `section_heading`, `block_type`, `retrieval_date`. |
| P3-06 | `embed_text` is copied into metadata | It is not. The stored document body is `text`, which is what a citation quotes. |
| P3-07 | The chunk fingerprint and the model name both match the manifest | Re-embedding is skipped. |
| P3-08 | The fingerprint matches and the model name does not | The collection is rebuilt with the current model. |
| P3-09 | The model name matches and the fingerprint does not | The collection is rebuilt. A changed chunk cannot stay in the index under the old vector. |
| P3-10 | `chunks.jsonl` line order changes while the text does not | The fingerprint covers `chunk_id + embed_text` in file order, so the index rebuilds. Keep the writer’s order stable across unchanged parses. |
| P3-11 | Two chunks share a `chunk_id` | Upsert would collapse them. The build stops. Collection count must equal the number of lines in `chunks.jsonl`. |
| P3-12 | A corpus-manifest row is missing or `stale` | The indexer does not publish. Existing index files from the last good build stay in place until a full successful run replaces them. |
| P3-13 | The manifest says `ok` and the raw file is missing | The row is treated as not publishable. The index is not updated. |
| P3-14 | The process dies after some upserts and before the manifest is written | The next run does not treat the collection as complete. A missing or older fingerprint means a rebuild. |
| P3-15 | The cosine space is configured so the code compares raw distance with `MIN_SIMILARITY` | Similarity is `1 - distance`. The threshold is applied to similarity. |
| P3-16 | A document’s chunks are absent from `chunks.jsonl` while the other six are present | The index is not published as the seven-document corpus. |

---

## Phase 4: Retrieval

| ID | Case | Expected |
| --- | --- | --- |
| P4-01 | `document_id` is not in the registry | The retriever raises before Chroma is called. |
| P4-02 | `document_id` is omitted | Search runs across the collection. The searched scope is all seven documents in registry order. |
| P4-03 | The filter is `eatwell-guide` and a nearer chunk exists in another document | Every returned hit has `document_id` `eatwell-guide`. |
| P4-04 | The filtered document has fewer chunks than `TOP_K_FILTERED` | The result contains only that document’s surviving hits. It does not borrow chunks from elsewhere. |
| P4-05 | Filtered search is at or above the threshold for six chunks | At most `TOP_K_FILTERED` (5) are kept. The per-document cap of 3 does not apply inside a filtered search. |
| P4-06 | Unfiltered search: one document occupies the raw top 8 | The retriever asks for `CANDIDATE_K` (24), then keeps at most 3 chunks from that document, so another authority can occupy a slot. The final list is at most `TOP_K_ALL` (8). |
| P4-07 | After the per-document cap, more than 8 hits remain | The kept list is the best 8 by similarity. |
| P4-08 | Two hits have the same similarity | Order is similarity, then registry order, then `chunk_id`, so the result is stable. |
| P4-09 | A hit’s similarity equals `MIN_SIMILARITY` | The hit is kept. |
| P4-10 | A hit’s similarity is just under `MIN_SIMILARITY` | The hit is dropped. |
| P4-11 | Every candidate is under the threshold | The grouped result is empty. The searched scope is still returned. This is not an exception and not a Groq call. |
| P4-12 | The query is embedded without `BGE_QUERY_PREFIX` | The retriever prefixes the query. Passages in the index stay unprefixed. |
| P4-13 | Chroma’s own embedding function also embeds the query | The collection is opened so the retriever’s vector is the one that is searched. The query is not embedded twice. |
| P4-14 | The question is “What is the tariff on imported olive oil?” | After the phase 4 threshold is set, the grouped result is empty and the searched scope lists all seven documents. |
| P4-15 | The question is “How long can raw chicken stay in the fridge?” filtered to `eatwell-guide` | The grouped result is empty. The searched scope is only the Eatwell Guide. A fridge-time row from another document is not added. |
| P4-16 | Whole chicken and chicken pieces are adjacent rows with close scores | Both may sit inside the cap of 3. The chunk text still names the food, so the generator can tell the rows apart. |
| P4-17 | Cold Food Storage and Kitchen Companion both describe leftovers | Both documents may appear, each with its own chunks. The retriever does not collapse them into one hit. |
| P4-18 | “What should I weigh?” scores above the threshold | Retrieval would return hits. That question never reaches the retriever. The phase 5 guard owns it. |
| P4-19 | The query plus the BGE prefix exceeds 512 tokens | The retriever rejects the query with a clear error. It does not truncate the question and search the prefix of it. |
| P4-20 | `MIN_SIMILARITY` is still `0.45` | The miss probes in the phase 4 table still return hits. The threshold is not left at the placeholder. |
| P4-21 | Raising `CANDIDATE_K` is required because one handbook fills all 24 slots | The new value is recorded with the probe that forced it. `TOP_K_ALL` and the cap of 3 stay unless a probe shows they hide a second authority. |

---

## Phase 5: Scope guard and refusals

The guard reads the raw message. It does not import the retriever or the Groq client.

### Reason boundaries

| ID | Case | Expected |
| --- | --- | --- |
| P5-01 | “What should I weigh?” | `out_of_scope` / `weight_target`. The message points to a qualified professional. `searched` is empty. |
| P5-02 | “What is my ideal weight?” or “How much should I weigh at 180 cm?” | `weight_target`. Same template as P5-01. |
| P5-03 | “How many calories should I eat to lose weight?” | `calorie_target`. Professional referral. `searched` is empty. |
| P5-04 | “What calorie deficit should I run?” | `calorie_target`. |
| P5-05 | “Is this chest pain from something I ate?” | `medical`. Professional referral. `searched` is empty. |
| P5-06 | “I have diabetes. What should I eat?” | `medical`. |
| P5-07 | “Which medicine should I take for food poisoning?” | `medical`. |
| P5-08 | “Can I eat this cheese if I am pregnant?” | `medical`. The question ties a food to a personal health condition. |
| P5-09 | “Is it safe to eat raw chicken?” | In scope. This is a food-safety question with no personal condition. Retrieval may answer from the food-safety documents. |
| P5-10 | “How much protein is in 100 g of chicken?” | `nutrient_lookup`. The reply says this assistant answers from dietary guidance and does not look up nutrient numbers for individual foods. It does not use the professional-referral sentence. `searched` is empty. |
| P5-11 | “How many calories are in a banana?” | `nutrient_lookup`, not `calorie_target`. The question asks for a per-food number. |
| P5-12 | “How many grams of fat are in olive oil?” | `nutrient_lookup`. |
| P5-13 | “What does WHO say about limiting free sugars?” | In scope. Population guidance is a question about the documents. |
| P5-14 | “What does the guidance say about salt?” | In scope, even though the documents state a population figure such as grams per day. |
| P5-15 | “According to the Eatwell Guide, how many calories should I eat?” | `calorie_target`. Naming a document does not make a personal calorie goal in scope. The guard wins over alias detection. |
| P5-16 | “I weigh 90 kg. Is that too much, and how much protein is in chicken?” | `weight_target`, the earlier reason in the precedence list. One response, no retrieval. |
| P5-17 | The message is empty or whitespace | Rejected as an empty message before the guard. The API returns 422. |

### Alias detection

| ID | Case | Expected |
| --- | --- | --- |
| P5-18 | “According to the Eatwell Guide, how much of the diet should be fruit and vegetables?” | Scope is `eatwell-guide`. |
| P5-19 | “eatwell”, “Eatwell Guide”, and “the Eatwell Guide booklet” | All three resolve to `eatwell-guide` when no second document is named. |
| P5-20 | “What do the Eatwell Guide and Kitchen Companion say about oil?” | Two aliases match. Scope stays unfiltered so both documents can be retrieved. |
| P5-21 | The request sends `document_id: kitchen-companion` and the text says “According to the Eatwell Guide” | The explicit id wins. Search is limited to Kitchen Companion. |
| P5-22 | “What is a healthy diet?” | Unfiltered. The words “healthy diet” are the topic of the product. They select `who-healthy-diet` only when the message names that fact sheet. |
| P5-23 | “What does WHO say about free sugars?” | Unfiltered. A bare “WHO” is not an alias. More than one corpus document is from WHO. |
| P5-24 | The alias is a substring of another word, such as matching `eat` inside an unrelated word | The match is on the alias as its own span. An accidental substring does not set `document_id`. |
| P5-25 | “Five keys to safer food” and “the five keys” | Scope is `who-five-keys`. |
| P5-26 | The UI sends a `document_id` and the message matches no alias | The explicit id is the scope. |

### Templates

| ID | Case | Expected |
| --- | --- | --- |
| P5-27 | Building any refusal template | The builder returns fixed text. It does not call Groq. |
| P5-28 | `not_in_corpus` for a filtered search | The list names that one document: name, publisher, and year. |
| P5-29 | `not_in_corpus` for an unfiltered search | The list names all seven documents in registry order. |
| P5-30 | `generation_unavailable` | The text says the answer could not be generated. It does not say the guidance was searched and found empty. `searched` is empty. |

---

## Phase 6: Answer layer

| ID | Case | Expected |
| --- | --- | --- |
| P6-01 | The guard matches | The pipeline returns `out_of_scope` and does not call the retriever or Groq. |
| P6-02 | Retrieval returns no groups | `not_in_corpus` for the searched scope. Groq is not called. |
| P6-03 | Groq returns HTTP 429, a timeout, or a transport error | `generation_unavailable`. The response is not `not_in_corpus`. |
| P6-04 | Groq returns prose, a markdown fence, or trailing-comma JSON | Strict parse fails. The response is `generation_unavailable`. |
| P6-05 | Groq returns valid JSON with an empty `documents` array, or every claims list empty | `not_in_corpus` for the searched scope. |
| P6-06 | A claim’s `chunk_id` was not in that document’s retrieved set | The claim is dropped. |
| P6-07 | A claim cites a chunk from document A inside document B’s section | The claim is dropped. A claim’s chunk ids all belong to the section’s `document_id`. |
| P6-08 | A claim lists two chunk ids, both retrieved, both from the same document | The claim can stand. Numbers in the claim must appear in those chunks, and at least half of the claim’s content words must appear in them. |
| P6-09 | A claim lists two chunk ids from two documents | The claim is dropped. |
| P6-10 | The claim text is empty or whitespace | The claim is dropped. |
| P6-11 | The chunk says “1 to 2 days” and the claim says “3 days” | The claim is dropped. Every number in the claim appears in the cited chunk text. |
| P6-12 | The chunk says “less than 10% of total energy intake” and the claim repeats that figure | The number is supported. The claim can stand when the question was in scope. |
| P6-13 | Fewer than half of the claim’s content words appear in the cited chunk | The claim is dropped. A share equal to `MIN_CLAIM_TERM_OVERLAP` (0.5) is kept. |
| P6-14 | The citation URL is anything other than that document’s registry `source_url` | The claim is dropped. The renderer fills the URL from the chunk record, so a model-supplied link is ignored. |
| P6-15 | The claim tells the person what they should weigh, or states a personal calorie goal, even though a chunk mentioned weight or energy | The claim is dropped by the same medical, calorie-target, and weight-target checks. A population figure that the chunk actually states, on an in-scope question, is not dropped for merely containing a number. |
| P6-16 | Every claim is dropped and at least one other document still has a surviving claim | The response is `answer`. The empty document is omitted. Sections stay in registry order. |
| P6-17 | Every claim in every section is dropped | `not_in_corpus`. The named search is the scope that was searched, not only the documents that had hits. |
| P6-18 | Two documents return surviving claims | Two sections. Each claim cites only its own document. Registry order, not similarity order. |
| P6-19 | The model writes “the guidelines say” into a claim | That claim is dropped. The renderer never adds a blended sentence of its own. |
| P6-20 | The model includes a publisher or a URL field | Those fields are ignored. Name, publisher, year, and link come from the chunk. |
| P6-21 | A filtered miss | `not_in_corpus` lists only that document’s name, publisher, and year. |
| P6-22 | An unfiltered answer | `searched` lists all seven documents. `sections` lists only documents with a surviving claim. |
| P6-23 | The same oil question retrieves a nutrition chunk and a storage chunk | The answer has one section per document. The two claims are not merged into one sentence. |
| P6-24 | Groq is given the user question plus chunk text from outside the retrieved set | That does not happen. The prompt contains the question and the grouped chunks only. |

---

## Phase 7: Chat API

| ID | Case | Expected |
| --- | --- | --- |
| P7-01 | The body omits `message`, or `message` is null, empty, or whitespace | HTTP 422. The pipeline is not called. |
| P7-02 | `message` is not a string | HTTP 422. |
| P7-03 | `document_id` is omitted or null | The pipeline searches all documents. |
| P7-04 | `document_id` is `""` | HTTP 422. The UI sends null for “All documents”, not an empty string. |
| P7-05 | `document_id` is not one of the seven ids | HTTP 422 before the pipeline runs. |
| P7-06 | The pipeline returns `answer`, `not_in_corpus`, `out_of_scope`, or `generation_unavailable` | HTTP 200 and the same JSON, including `type`, `sections`, `searched`, and `reason`. A generation failure is not turned into HTTP 500. |
| P7-07 | `GET /health` and the Chroma directory or the index manifest is missing | The body reports not ready. It does not report that the corpus is searchable. |
| P7-08 | `GET /health` and both the directory and the manifest exist | The body reports ok. |
| P7-09 | A request is logged after validation drops claims | The log has the scope, the hit count, and the drop reason codes. The raw model JSON is not logged. |
| P7-10 | An unfiltered `answer` | `searched` has seven entries. `sections` has one entry per document that survived validation. |

---

## Phase 8: Chat UI

| ID | Case | Expected |
| --- | --- | --- |
| P8-01 | The picker is “All documents” | The request sends `document_id: null`. |
| P8-02 | The picker is one registry document | The request sends that `document_id`, including when the typed question names a different document. |
| P8-03 | The response has two sections | The page shows two blocks, in registry order. Each citation’s link is that section’s `source_url`. |
| P8-04 | A claim has a section heading | The citation shows document name, publisher, year, section heading, and the link. |
| P8-05 | `type` is `not_in_corpus` | The page shows the message and the searched documents. The layout is different from `out_of_scope`. |
| P8-06 | `type` is `out_of_scope` and the reason is `weight_target`, `calorie_target`, or `medical` | The page shows the decline and the professional referral. It shows no searched list, even if the picker had a document selected. |
| P8-07 | The reason is `nutrient_lookup` | The page shows that sentence. It does not show the professional-referral sentence and it does not show a searched list. |
| P8-08 | `type` is `generation_unavailable` | The page shows its own state. It does not reuse the not-in-corpus wording or the searched list. |
| P8-09 | The API is unreachable | The page says the request failed. It does not present an empty chat turn as “the guidance does not cover that”. |
| P8-10 | The four example prompts | They send the fridge-time question, the cooking-oil question, the Eatwell fruit-and-vegetables question, and “What should I weigh?” unchanged. |
| P8-11 | Any response | The disclaimer stays visible: answers are a reading of the seven official documents. |

---

## Phase 9: Evaluation

| ID | Case | Expected |
| --- | --- | --- |
| P9-01 | The index is missing | The fridge, oil, and Eatwell paths are not recorded as passes. The guard paths can still be recorded, because they do not retrieve. |
| P9-02 | `GROQ_API_KEY` is missing | The out-of-scope and not-covered paths can still be judged. The fridge, oil, and Eatwell paths are not judged as `not_in_corpus` to paper over the missing key. |
| P9-03 | Groq fails during the fridge-time path | The recorded type is `generation_unavailable`. That run is not a pass of the fridge path. |
| P9-04 | The cooking-oil path returns one section because only one document survived validation | The shape still passes the per-document rule: that section cites only itself, and the text does not say what “the guidelines” say. A second section is not invented. Zero sections is a failure of the path. |
| P9-05 | The Eatwell path returns `not_in_corpus` | It passes only when `searched` is exactly `eatwell-guide` and no other `document_id` appears. |
| P9-06 | A citation URL is the downloaded asset URL rather than the registry URL | The eval fails for that claim. |
| P9-07 | A worked path fails because the threshold is too high or too low | `MIN_SIMILARITY` moves only inside the band the phase 4 probes established. The new value and the probe are written into `docs/eval.md`. |
| P9-08 | The not-covered and out-of-scope paths are judged by reading Groq’s prose | They are judged from the code path: empty retrieval, or the guard. Groq is not required for those two. |
| P9-09 | `MIN_SIMILARITY` is still `0.45` | Evaluation does not pass. |

---

## Phase 10: README

| ID | Case | Expected |
| --- | --- | --- |
| P10-01 | The README describes chunking only as “we split the documents” | The exit check fails. The paragraph names block-atomic chunks, the table-row rule, the one-item list rule, and the costs from phase 2. |
| P10-02 | The README threshold disagrees with `MIN_SIMILARITY` in code | The README is updated to the value phase 4 set and phase 9 kept, including the probes. |
| P10-03 | The README lists a source URL that is not in the registry | The list is the seven registry URLs. Landing-page citations stay the publication URLs. |
| P10-04 | The README includes a real `GROQ_API_KEY` | The key is named, not filled in. The reader copies `.env.example`. |
| P10-05 | The run instructions assume the index already exists | The steps include `python -m scripts.build_index` before the API and the UI. |

---

## Cross-phase cases

| ID | Case | Expected |
| --- | --- | --- |
| X-01 | A stale manifest row is ignored and the index is rebuilt from the other files | The publish is refused. The chat keeps the last complete index or serves not-ready. It does not answer as if all seven documents were current. |
| X-02 | A citation is rendered with `downloaded_url` for a WHO landing page | The link is `source_url`, the publication page. The PDF bytes on disk are only the retrieval artifact. |
| X-03 | A personal calorie question is allowed through because the Eatwell Guide contains an energy table | The guard returns `calorie_target` before retrieval. The table never reaches the generator for that question. |
| X-04 | “Protein in 100 g of chicken” is answered from a chunk that happens to mention chicken and a number | The guard returns `nutrient_lookup` before retrieval. |
| X-05 | Cooking oil is answered as one paragraph that attributes a single recommendation to both publishers | The response has one section per surviving document, each with its own citation. There is no merged claim. |
| X-06 | The scope guard is implemented only as a sentence in the Groq prompt | The phase 5 tests fail. “What should I weigh?” does not call the retriever. |
| X-07 | Chunk ids change after a re-chunk, and an old chat turn still holds the previous ids | Each request retrieves again. Citations are built from the hits of that request. Stored chunk ids are not a client cache. |
| X-08 | Phase 5 is developed before the index exists | Classifier and refusal tests run with no Chroma directory and no Groq key. |
| X-09 | Phase 6 tests call the real Groq API | They do not. The suite uses a fake retriever and a fake generator through phase 7. |
| X-10 | A filtered not-in-corpus reply lists all seven documents | It lists the one document in the scope. Listing the other six claims they were searched. |
| X-11 | An out-of-scope reply lists the corpus “so the user knows what we have” | `searched` stays empty. The corpus was not consulted. |
| X-12 | `generation_unavailable` is shown with the not-in-corpus document list | The UI uses the generation state. The list would claim a search result the pipeline did not reach. |

---

## Response matrix

This is the same corner, seen from the response type. A test that checks the type and nothing else is incomplete. The row says what else must be true.

| Situation | `type` | `reason` | `searched` | Groq called | Retriever called |
| --- | --- | --- | --- | --- | --- |
| Empty message | HTTP 422 | — | — | No | No |
| Unknown `document_id` | HTTP 422 | — | — | No | No |
| Medical, calorie target, or weight target | `out_of_scope` | `medical`, `calorie_target`, or `weight_target` | Empty | No | No |
| Per-food nutrient number | `out_of_scope` | `nutrient_lookup` | Empty | No | No |
| In scope, no hit at or above the threshold | `not_in_corpus` | `no_supporting_chunks` | The scope that was searched | No | Yes |
| Hits returned, every claim dropped | `not_in_corpus` | The validator’s reason | The scope that was searched | Yes | Yes |
| Groq error, timeout, or invalid JSON | `generation_unavailable` | `model_error` | Empty | Yes | Yes |
| One or more claims survive | `answer` | — | The scope that was searched | Yes | Yes |

On `answer` and on `not_in_corpus`, a filtered scope has one document in `searched`. An unfiltered scope has seven.
