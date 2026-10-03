"""Scope guard and document-alias detection.

The guard reads the raw user message. It does not retrieve and it does not
call a model. A match is a reason code. Population guidance stays in scope.

When more than one reason matches, the first hit wins, in this order:
medical, weight_target, calorie_target, nutrient_lookup.

An explicit ``document_id`` overrides aliases. Two named documents leave the
scope unfiltered so both can be retrieved. The alias "healthy diet" is the
product topic: it selects the WHO fact sheet only when a longer alias names
that fact sheet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.constants import CORPUS, DOCUMENT_IDS

MEDICAL = "medical"
WEIGHT_TARGET = "weight_target"
CALORIE_TARGET = "calorie_target"
NUTRIENT_LOOKUP = "nutrient_lookup"

# Precedence. The first match is the reason code.
SCOPE_REASONS: tuple[str, ...] = (
    MEDICAL,
    WEIGHT_TARGET,
    CALORIE_TARGET,
    NUTRIENT_LOOKUP,
)
PERSONAL_REASONS = frozenset({MEDICAL, CALORIE_TARGET, WEIGHT_TARGET})

# "healthy diet" is how people ask the product question. The fact-sheet
# aliases still select who-healthy-diet.
_TOPIC_ALIASES = frozenset({"healthy diet"})

_CONDITION = (
    r"(?:diabet(?:es|ics|ic)|pregnan(?:t|cy)|cancer|hypertension|"
    r"high blood pressure|c(?:o)?eliac|crohn(?:'s)?|allerg(?:y|ies|ic)|"
    r"intoleran(?:t|ce)|kidney disease|heart disease|gout|"
    r"an(?:a)?emi(?:a|c)|asthma|arthritis|thyroid|hiv|ibs|"
    r"chest pain|stomach ache|stomachache|abdominal pain)"
)
_SYMPTOM = (
    r"(?:chest pain|stomach ache|stomachache|abdominal pain|nausea|"
    r"vomiting|diarrhoea|diarrhea|dizziness|dizzy|rash|migraine|headache|"
    r"shortness of breath|palpitations)"
)
_NUTRIENT = (
    r"(?:proteins?|kilocalories|calories|calorie|kcals?|carbohydrates?|"
    r"carbs?|sugars?|fats?|sodium|salt|fib(?:re|er)|cholesterol|"
    r"vitamins?(?:\s+[a-k])?|iron|calcium|potassium|nutrients?)"
)
_NUTRIENT_MODIFIER = r"(?:saturated|trans|total|free|added|dietary)"

_MEDICAL_PATTERNS = (
    rf"\b{_SYMPTOM}\b",
    r"\b(?:medicines?|medications?|prescriptions?|prescribed|prescribe|dosage)\b",
    r"\bdiagnos(?:is|e|ed|es)\b",
    r"\bsymptoms?\b",
    r"\btreatment for\b",
    rf"\b(?:i have|i've got|i am|i'm|if i am|if i'm|if i have)\b.{{0,40}}\b{_CONDITION}\b",
    rf"\b(?:can|should|may)\b.{{0,40}}\b(?:eat|eating)\b.{{0,50}}\b{_CONDITION}\b",
    rf"\b(?:safe|eat|eating)\b.{{0,50}}\b{_CONDITION}\b",
    rf"\b{_CONDITION}\b.{{0,50}}\b(?:eat|eating|safe)\b",
)
# Personal weight, not the weight of a bird or a cut of meat.
# "lose weight" is intentionally absent: "calories ... to lose weight"
# is a calorie target, and that reason is checked after this one.
_WEIGHT_PATTERNS = (
    r"\bideal weight\b",
    r"\b(?:target|goal|desired|personal) weight\b",
    r"\bweight (?:target|goal)\b",
    r"\b(?:should|do|can) (?:i|we) weigh\b",
    r"\bhow much should (?:i|we) weigh\b",
    r"\bmy weight\b",
    r"\bweigh(?:ing)? too (?:much|little)\b",
    r"\b(?:am i|are we|i am|i'm) (?:overweight|underweight)\b",
    r"\b(?:i|we) weigh\s+(?:about\s+|around\s+|over\s+|under\s+)?\d",
)
_CALORIE_PATTERNS = (
    r"\bcalor(?:ie|ies|ic)\s+deficit\b",
    r"\bdeficit\b.{0,30}\bcalor(?:ie|ies|ic)\b",
    r"\bcalor(?:ie|ies|ic)\b.{0,30}\bdeficit\b",
    r"\b(?:calorie|caloric|calories)\s+(?:goal|target|budget|allowance|surplus)\b",
    r"\bhow many (?:calories|kcals?|kilocalories) should (?:i|we)\b",
    r"\b(?:calories|kcals?|kilocalories) should (?:i|we)\b",
    r"\bmy (?:daily )?(?:calorie|caloric|calories)\b",
    r"\bpersonal calorie\b",
)
# A nutrient amount in a named food. "how many calories should I eat"
# does not match: that is a personal target, checked above.
_NUTRIENT_PATTERNS = (
    rf"\bhow (?:much|many)\s+(?:(?:grams|g|mg|kcal)\s+of\s+)?"
    rf"(?:{_NUTRIENT_MODIFIER}\s+)?{_NUTRIENT}\s+(?:is|are)\s+in\b",
    rf"\bhow (?:much|many)\s+(?:grams|g|mg|kcal)\s+of\s+"
    rf"(?:{_NUTRIENT_MODIFIER}\s+)?{_NUTRIENT}\s+(?:is\s+|are\s+)?in\b",
    # "protein in 100 g", not "fat in 10%" — a percent is population guidance.
    rf"\b{_NUTRIENT}\s+in\s+\d+(?:\.\d+)?\s*(?:kilograms?|grams?|kg|mg|g)\b",
    rf"\b{_NUTRIENT}\s+(?:content|amount)\s+(?:of|in)\b",
    rf"\bhow (?:much|many)\s+(?:{_NUTRIENT_MODIFIER}\s+)?{_NUTRIENT}\s+"
    rf"(?:does|do)\b.{{0,40}}\b(?:have|contain)\b",
)

_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    MEDICAL: tuple(re.compile(pattern) for pattern in _MEDICAL_PATTERNS),
    WEIGHT_TARGET: tuple(re.compile(pattern) for pattern in _WEIGHT_PATTERNS),
    CALORIE_TARGET: tuple(re.compile(pattern) for pattern in _CALORIE_PATTERNS),
    NUTRIENT_LOOKUP: tuple(re.compile(pattern) for pattern in _NUTRIENT_PATTERNS),
}
_ALIAS_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_ALIAS_SPACE = re.compile(r"\s+")


class EmptyMessageError(ValueError):
    """The message is empty or whitespace. The guard has not run."""


class UnknownDocumentError(ValueError):
    """``document_id`` is outside the seven-document registry."""


@dataclass(frozen=True)
class Classification:
    """Reason code, and the document scope aliases or the request selected.

    A set ``reason`` means the question is out of scope. Retrieval must not
    run, even when ``document_id`` names a corpus document. ``document_id``
    is None when the search is the whole corpus.
    """

    reason: str | None
    document_id: str | None


def classify(message: str, document_id: str | None = None) -> Classification:
    """Classify ``message``. An explicit ``document_id`` overrides aliases."""
    if not isinstance(message, str) or not message.strip():
        raise EmptyMessageError("message is empty")
    if document_id is not None and document_id not in DOCUMENT_IDS:
        raise UnknownDocumentError(f"document_id {document_id} is not in the registry")
    folded = _fold(message)
    return Classification(
        reason=scope_reason(folded),
        document_id=document_id if document_id is not None else detect_document(folded),
    )


def scope_reason(message: str) -> str | None:
    """First matching reason code, or None when the message is in scope.

    ``message`` may already be folded. Callers passing raw text are fine:
    matching is case-insensitive after folding.
    """
    folded = _fold(message)
    for reason in SCOPE_REASONS:
        if any(pattern.search(folded) for pattern in _PATTERNS[reason]):
            return reason
    return None


def detect_document(message: str) -> str | None:
    """The one registry document named in ``message``, or None.

    None means unfiltered: no alias matched, or more than one document did.
    """
    folded = _alias_text(message)
    matched: list[str] = []
    for document in CORPUS:
        if _document_named(folded, document.aliases):
            matched.append(document.document_id)
    if len(matched) == 1:
        return matched[0]
    return None


def _fold(message: str) -> str:
    return message.casefold().replace("\u2019", "'").replace("\u00a0", " ")


def _alias_text(message: str) -> str:
    text = _ALIAS_PUNCT.sub(" ", _fold(message))
    return _ALIAS_SPACE.sub(" ", text).strip()


def _document_named(folded: str, aliases: tuple[str, ...]) -> bool:
    for alias in aliases:
        key = alias.casefold()
        if key in _TOPIC_ALIASES:
            continue
        if re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", folded):
            return True
    return False
