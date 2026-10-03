# Evaluation

This is the scorecard for phase 9 of [implementation-plan.md](./implementation-plan.md). It checks the worked paths in [architecture.md](./architecture.md) and the behaviour table in [problemStatement.md](./problemStatement.md). Corner rulings for this run live in [edge-case.md](./edge-case.md) as P9-01 through P9-09.

The rubric below is the contract. The run record at the end is filled by `scripts/eval_paths.py` against a built index. This file does not invent those results. Until that script has been run, every path is **not run**.

A path passes only when its type, scope, and citation checks all hold. A matching `type` with the wrong `searched` list, or a citation whose URL is not in the registry, is a fail.

---

## When a path may be judged

| Path group | Needs the index | Needs Groq | If the dependency is missing |
| --- | --- | --- | --- |
| Fridge time, cooking oil, Eatwell filter | Yes | Yes | Leave the path **not run**. Do not record it as `not_in_corpus`. |
| Not covered | Yes | No | Leave the path **not run** when the index is missing. Judge it from empty retrieval when the index exists. |
| Weight, calorie, medical, nutrient lookup | No | No | Judge from the scope guard. An empty Chroma directory does not block these four. |

`MIN_SIMILARITY` must already be the value phase 4 set from probes. The phase 0 placeholder `0.45` fails the run (P0-07, P9-09). A Groq error on an answer path is recorded as `generation_unavailable` and is not a pass (P9-03).

---

## How to run

From a built index and a `GROQ_API_KEY` in the environment:

```text
python -m scripts.eval_paths
```

The script calls the same pipeline as `POST /chat`. It writes the run record in this file: date, model name, `MIN_SIMILARITY`, and one block per path with `type`, `reason`, `searched`, section `document_id`s, and whether each citation URL is in the registry. It leaves this rubric in place.

Retrieval probes use `scripts/probe_retrieval.py` and are recorded in the threshold log. They are not a substitute for the answer-path checks.

---

## Registry URLs

Every citation link on an `answer` must be one of these. The two WHO publication pages are cited at the page URL. The downloaded PDF address is not a valid citation (P9-06).

| `document_id` | Document | Publisher | Year | Source URL |
| --- | --- | --- | --- | --- |
| `cold-food-storage` | Cold Food Storage Chart | FoodSafety.gov (U.S. Department of Health and Human Services / FDA / USDA) | 2023 | https://www.foodsafety.gov/food-safety-charts/cold-food-storage-charts |
| `who-healthy-diet` | Healthy diet (Fact sheet) | World Health Organization | 2018 | https://www.who.int/docs/default-source/healthy-diet/healthy-diet-fact-sheet-394.pdf |
| `fao-who-healthy-diets` | What are healthy diets? Joint statement | Food and Agriculture Organization of the United Nations and World Health Organization | 2024 | https://www.who.int/publications/i/item/9789240101876 |
| `who-five-keys` | Five keys to safer food manual | World Health Organization | 2006 | https://www.who.int/publications/i/item/9789241594639 |
| `eatwell-guide` | The Eatwell Guide booklet | Public Health England (now Office for Health Improvement and Disparities) | 2018 | https://assets.publishing.service.gov.uk/media/5ba8a50540f0b605084c9501/Eatwell_Guide_booklet_2018v4.pdf |
| `fsa-chill` | How to chill, freeze and defrost food safely | Food Standards Agency (UK) | 2017 | https://www.gov.uk/government/publications/how-to-chill-freeze-and-defrost-food-safely/how-to-chill-freeze-and-defrost-food-safely |
| `kitchen-companion` | Kitchen Companion: Your Safe Food Handbook | USDA Food Safety and Inspection Service | 2008 | https://www.fsis.usda.gov/sites/default/files/media_file/2020-12/Kitchen-Companion.pdf |

On an unfiltered in-scope response, `searched` lists all seven ids in this order. On a filtered response, `searched` lists the one id that was searched.

---

## Shared checks

Apply these to every path, then apply the path checks.

| Check | Pass when |
| --- | --- |
| Type | `type` is the expected value. Where a path allows two types, the one that came back meets that path’s shape for that type. |
| Scope | `searched` matches the scope below. `out_of_scope` and `generation_unavailable` have an empty `searched` list. |
| Sections | Each section’s `document_id` is in `searched`. Section order follows the registry table. |
| Claims | Each claim has non-empty `text`, at least one `chunk_id`, and a `section_heading`. Every `chunk_id` belongs to that section’s `document_id`. |
| Citation | Each claim’s rendered citation includes document name, publisher, year, and `source_url` from the registry row for that `document_id`. |
| Numbers | Every number in a claim appears in the cited chunk text. |
| Blended voice | The answer text does not say what “the guidelines” collectively say. |
| Guard boundary | Weight, calorie, medical, and nutrient paths do not call the retriever. The not-covered path does not call Groq. |

---

## Answer paths

These three call retrieval and Groq.

### 1. Fridge time

| | |
| --- | --- |
| Question | How long can I keep a whole chicken in the fridge? |
| `document_id` sent | null |
| Expected `type` | `answer` |
| Behaviour row | The question is covered by one document. A second food-safety document may also survive. |

Pass when all of these hold:

- `searched` lists all seven documents.
- `sections` includes `cold-food-storage`.
- The Cold Food Storage claim cites Cold Food Storage Chart, FoodSafety.gov (U.S. Department of Health and Human Services / FDA / USDA), 2023, and the chart URL.
- The cited chunk is one table row. That row’s refrigerator value and freezer value both appear in the claim, and the claim does not mix in a different food’s row.
- Any other section, such as `kitchen-companion`, has its own claims and its own citation. No claim cites another document’s chunk ids.

### 2. Cooking oil

| | |
| --- | --- |
| Question | What do the documents say about cooking oil? |
| `document_id` sent | null |
| Expected `type` | `answer` |
| Behaviour row | The question is covered by two or more documents when more than one document survives. |

Pass when all of these hold:

- `searched` lists all seven documents.
- There is one section per document that produced a surviving claim, and at least one section.
- Each section cites only its own chunk ids, name, publisher, year, and registry URL.
- The answer does not contain a sentence that attributes one recommendation to “the guidelines”.
- Nutrition documents that may appear are `who-healthy-diet`, `fao-who-healthy-diets`, and `eatwell-guide`. A food-safety document appears only when one of its own chunks survived. The evaluator does not add a second section to force a cross-document shape (P9-04). Zero sections is a fail.

### 3. Filtered Eatwell question

| | |
| --- | --- |
| Question | According to the Eatwell Guide, how much of the diet should be fruit and vegetables? |
| `document_id` sent | null. Alias detection must set `eatwell-guide`. |
| Expected `type` | `answer` or `not_in_corpus` |
| Behaviour row | Filtered retrieval. A miss names only the document that was searched. |

Pass when all of these hold:

- `searched` contains only `eatwell-guide`.
- No other `document_id` appears in `sections`.
- On `answer`, the citation is The Eatwell Guide booklet, Public Health England (now Office for Health Improvement and Disparities), 2018, and the Eatwell PDF URL.
- On `not_in_corpus`, the message says the guidance does not cover the question and names only the Eatwell Guide (P9-05).

---

## Refusal paths

### 4. Not covered

| | |
| --- | --- |
| Question | What is the tariff on imported olive oil? |
| `document_id` sent | null |
| Expected `type` | `not_in_corpus` |
| Expected `reason` | `no_supporting_chunks` |
| Groq | Not called |

Pass when the message says the guidance searched does not cover the question, `sections` is empty, and `searched` lists all seven documents by name, publisher, and year in registry order.

### 5. Weight target

| | |
| --- | --- |
| Question | What should I weigh? |
| Expected `type` | `out_of_scope` |
| Expected `reason` | `weight_target` |
| Retriever | Not called |

Pass when `searched` and `sections` are empty, and the message declines and points the person to a qualified professional.

### 6. Calorie target

| | |
| --- | --- |
| Question | How many calories should I eat to lose weight? |
| Expected `type` | `out_of_scope` |
| Expected `reason` | `calorie_target` |
| Retriever | Not called |

Pass when `searched` and `sections` are empty, and the message declines and points the person to a qualified professional. Naming a document in a personal calorie question does not make it an answer path. “According to the Eatwell Guide, how many calories should I eat?” is the same reason.

### 7. Medical

| | |
| --- | --- |
| Question | Is this chest pain from something I ate? |
| Expected `type` | `out_of_scope` |
| Expected `reason` | `medical` |
| Retriever | Not called |

Pass when `searched` and `sections` are empty, and the message declines and points the person to a qualified professional.

### 8. Nutrient lookup

| | |
| --- | --- |
| Question | How much protein is in 100 g of chicken? |
| Expected `type` | `out_of_scope` |
| Expected `reason` | `nutrient_lookup` |
| Retriever | Not called |

Pass when `searched` and `sections` are empty, and the message says this assistant answers from dietary guidance documents and does not look up nutrient numbers for individual foods. The professional-referral sentence is not this path’s message.

“How many calories are in a banana?” is the same reason. It is a per-food number, not `calorie_target`.

---

## Retrieval probes

Phase 4 sets `MIN_SIMILARITY` from these probes. Phase 9 re-runs them when an answer path fails because relevant rows were dropped or irrelevant rows were kept. Move the threshold only inside the band written next to the constant. Record the new value and the probe in the threshold log.

| Probe | Scope | Pass |
| --- | --- | --- |
| How long can I keep a whole chicken in the fridge? | all | A `cold-food-storage` poultry row is among the hits, and that chunk contains both storage times. |
| How long can I keep cooked leftovers in the fridge? | all | More than one food-safety `document_id` remains after the cap of 3 per document. |
| What do the documents say about cooking oil? | all | At least one of `who-healthy-diet`, `fao-who-healthy-diets`, or `eatwell-guide` is among the hits. |
| According to the Eatwell Guide, how much of the diet should be fruit and vegetables? | `eatwell-guide` | Every hit is `eatwell-guide`. |
| What is the tariff on imported olive oil? | all | No hit stays at or above the threshold. The searched scope is still all seven documents. |
| How long can raw chicken stay in the fridge? | `eatwell-guide` | No hit stays at or above the threshold. |

A hit whose similarity equals the threshold is kept. A hit below it is dropped.

---

## Threshold log

Fill a row when phase 4 sets the threshold, and another row if phase 9 moves it.

| Date | `MIN_SIMILARITY` | Band | Probe that set it | Notes |
| --- | --- | --- | --- | --- |
| — | — | — | — | Not set. The placeholder `0.45` is not a result. |

---

## Run record

Filled by `scripts/eval_paths.py`. Leave this table as **not run** until then.

| | |
| --- | --- |
| Run date | Not run |
| Index fingerprint | — |
| Embedding model | — |
| Groq model | — |
| `MIN_SIMILARITY` | — |

| # | Path | Question | Expected `type` | Actual `type` | Actual `reason` | Section ids | Citation URLs in registry | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Fridge time | How long can I keep a whole chicken in the fridge? | `answer` | — | — | — | — | Not run |
| 2 | Cooking oil | What do the documents say about cooking oil? | `answer` | — | — | — | — | Not run |
| 3 | Filtered | According to the Eatwell Guide, how much of the diet should be fruit and vegetables? | `answer` or `not_in_corpus` | — | — | — | — | Not run |
| 4 | Not covered | What is the tariff on imported olive oil? | `not_in_corpus` | — | — | — | — | Not run |
| 5 | Weight target | What should I weigh? | `out_of_scope` | — | — | — | — | Not run |
| 6 | Calorie target | How many calories should I eat to lose weight? | `out_of_scope` | — | — | — | — | Not run |
| 7 | Medical | Is this chest pain from something I ate? | `out_of_scope` | — | — | — | — | Not run |
| 8 | Nutrient lookup | How much protein is in 100 g of chicken? | `out_of_scope` | — | — | — | — | Not run |

### Notes from the run

None yet. Record a threshold move, a single-section cooking-oil answer, or a filtered `not_in_corpus` here, with the probe or the `searched` list that explains it.

### Exit

Phase 9 exits when all eight rows are **pass**, every citation URL on paths 1–3 is in the registry table, and the threshold log shows a probed value rather than `0.45`.
