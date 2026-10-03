# Dietary Guidance RAG Chatbot

## Problem

People ask everyday questions about food, nutrition, and food safety: whether a way of eating is reasonable, and how long food can stay in the fridge. The answers already exist. National nutrition institutes, food safety regulators, and international health bodies publish long, careful guidance as written prose, usually as PDFs. There is no API for that guidance, and almost nobody reads the documents.

The result is a gap. Official advice is public and authoritative, but it is hard to use in the moment. A chatbot without a retrieval layer would fill that gap with ungrounded answers. A chatbot that only searches the web, or that mixes nutrient tables with policy prose, would answer a different question than the one these documents were written to settle.

This prototype closes that gap with retrieval-augmented generation. The assistant answers only from a small corpus of official public dietary guidance. Every claim carries a citation. When the guidance does not cover a question, the assistant says so.

## What this prototype is

This is the first service in a larger product. In the final project it answers two kinds of questions:

- Is this a reasonable way to eat?
- How long can I keep this in the fridge?

Those questions are answered from written guidance, not from a food-composition database. Nutrient numbers for individual foods are a different kind of data. They come from a structured database in Milestone 3 and do not belong in this corpus or in this answer layer.

## Who it is for

The user is someone with a practical food question who wants an answer they can check against a named public document. They are not looking for a diagnosis, a personal diet plan, or a calorie or weight target.

The system is for them only as a reader of official guidance. It is not a clinician, a dietitian, or a substitute for one.

## Outcome

A person can ask a question about food, nutrition, or food safety and receive one of three results:

1. An answer drawn only from retrieved passages, with a citation on every claim.
2. A clear statement that the retrieved guidance does not cover the question, including what was searched.
3. A refusal when the question is out of scope by design, with a pointer to a qualified professional.

The person can tell which document said what. When two authorities both speak to the same topic, their positions stay separate.

## Scope

### In scope

1. **Corpus.** Use these seven public guidance documents. They come from national nutrition institutes, food safety regulators, and international health bodies, and they cover both sides of the product: dietary pattern guidance and food storage. The documents must be written prose. Anything that already has a clean API behind it does not belong here. Store publisher, year, source URL, and retrieval date with every document.

   | Document | Publisher | Year | Source URL |
   | --- | --- | --- | --- |
   | Cold Food Storage Chart | FoodSafety.gov (U.S. Department of Health and Human Services / FDA / USDA) | 2023 | https://www.foodsafety.gov/food-safety-charts/cold-food-storage-charts |
   | Healthy diet (Fact sheet) | World Health Organization | 2018 | https://www.who.int/docs/default-source/healthy-diet/healthy-diet-fact-sheet-394.pdf |
   | What are healthy diets? Joint statement | Food and Agriculture Organization of the United Nations and World Health Organization | 2024 | https://www.who.int/publications/i/item/9789240101876 |
   | Five keys to safer food manual | World Health Organization | 2006 | https://www.who.int/publications/i/item/9789241594639 |
   | The Eatwell Guide booklet | Public Health England (now Office for Health Improvement and Disparities) | 2018 | https://assets.publishing.service.gov.uk/media/5ba8a50540f0b605084c9501/Eatwell_Guide_booklet_2018v4.pdf |
   | How to chill, freeze and defrost food safely | Food Standards Agency (UK) | 2017 | https://www.gov.uk/government/publications/how-to-chill-freeze-and-defrost-food-safely/how-to-chill-freeze-and-defrost-food-safely |
   | Kitchen Companion: Your Safe Food Handbook | USDA Food Safety and Inspection Service | 2008 | https://www.fsis.usda.gov/sites/default/files/media_file/2020-12/Kitchen-Companion.pdf |

   Healthy diet, What are healthy diets?, and the Eatwell Guide are nutrition guidance. The Cold Food Storage Chart, Five keys to safer food, How to chill, freeze and defrost food safely, and Kitchen Companion are food-safety guidance. The nutrition documents discuss fats and oils. The food-safety documents discuss storage times, chilling, and leftovers. That split is the cross-document case in item 5. The Cold Food Storage Chart is largely a table of food type against refrigerator and freezer time, so it is a direct test of the chunking constraint in item 2.

2. **Chunking.** Split the documents so retrieval can return a usable passage. Every chunk carries the document name, publisher, year, and section heading. These documents are full of tables and numbered recommendations. Fixed-size chunking will cut those structures in half. The README must say what chunking approach was chosen and what it cost.

3. **Retrieval.** Build a vector index over the chunks. Support two modes:
   - Retrieval across all documents.
   - Retrieval filtered to one named document.

4. **Answer layer.** The assistant answers only from retrieved chunks. It does not add claims from model memory, the open web, or nutrient databases. Every claim carries a citation showing document name, publisher, year, and a link.

5. **Cross-document questions.** Some questions have two documents with something to say. Cooking oil is the example: a nutrition institute and a food safety regulator may both weigh in. Answer per document, with separate citations. Never blend two sources into one claim about what "the guidelines say."

6. **Two kinds of refusal.** Both are required, and they are different.
   - **Not in the corpus.** When the retrieved chunks do not hold the answer, the assistant says the guidance does not cover it and names what it searched.
   - **Out of scope by design.** No medical advice, no calorie or weight targets, and nothing about what anyone should weigh. The assistant declines and points the person to a qualified professional. Enforce this in code.

### Out of scope

- Nutrient amounts, calorie counts, and other per-food composition data. That is Milestone 3.
- Medical advice, diagnosis, treatment, personal diet prescriptions, calorie targets, and weight targets.
- Sources that are not official public dietary or food-safety guidance in written prose.
- Documents that are already exposed through a clean API. Those are a different integration.
- A blended "the guidelines say" voice that merges two publishers into one claim.

## Constraints

- **Grounding.** Answers come only from retrieved chunks of the gathered corpus.
- **Citations.** A claim without a citation is not an acceptable answer. A citation includes document name, publisher, year, and a link.
- **Provenance.** Every stored document records publisher, year, source URL, and the date it was retrieved.
- **Chunk identity.** Every chunk records document name, publisher, year, and section heading, so a citation can be traced back to a section rather than to an anonymous span of text.
- **Structure.** Tables and numbered recommendations must survive chunking well enough to be retrieved as coherent guidance. The cost of the chosen method is documented in the README.
- **Separation of authorities.** Cross-document answers stay per document. One source is never paraphrased as the position of "the guidelines" when another source is also in the answer.
- **Refusal in code.** Out-of-scope questions are blocked by an explicit check, not only by a prompt instruction.
- **Corpus size.** The seven documents listed under Corpus. Enough to show cross-document retrieval, small enough to stay a prototype.

## Behaviour

| Situation | Expected behaviour |
| --- | --- |
| The question is covered by one document | Answer from retrieved chunks of that document. Cite document name, publisher, year, and a link on every claim. |
| The question is covered by two or more documents | Answer separately for each document. Give each its own citations. Do not merge them into one claim. |
| Retrieved chunks do not contain the answer | Say the guidance does not cover it. Name what was searched. |
| The question asks for medical advice, a calorie target, a weight target, or what someone should weigh | Decline. Point the person to a qualified professional. This path is enforced in code. |
| The question needs nutrient numbers for a specific food | Do not answer from this corpus. That data is out of scope for this prototype. |

## Acceptance criteria

- The corpus contains the seven public prose documents listed under Corpus, each stored with publisher, year, source URL, and retrieval date.
- No document in the corpus is one that already has a clean API as its primary interface.
- A vector index exists over the chunks and can retrieve across the full corpus or within one named document.
- Every chunk used for retrieval carries document name, publisher, year, and section heading.
- The README states the chunking method and the tradeoff it accepts, especially for tables and numbered recommendations.
- Sample answers contain only claims supported by retrieved chunks, and every claim has a citation with document name, publisher, year, and a link.
- A cross-document question, such as one about cooking oil, produces separate answers and separate citations. It does not say what "the guidelines" collectively say.
- A question absent from the corpus produces a not-in-corpus refusal that names the search.
- A medical, calorie-target, or weight-target question produces an out-of-scope refusal implemented in code, and points to a qualified professional.

## Success

The prototype is successful when a reader can ask a food, nutrition, or food-safety question and either check every sentence against a named public document, or hear a precise reason the assistant will not answer. It is not successful if it sounds confident without a citation, blends two authorities into one voice, or answers questions the corpus and the product boundary were built to refuse.
