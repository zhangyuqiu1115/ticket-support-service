"""Extractive offline generator: each factual line quotes retrieved source text."""
from typing import Sequence

from .classification import article_intent
from .domain import Classification, Hit, Ticket


MIN_SCORE = 0.12
REFUSAL = "I'm sorry, but I cannot provide asset recommendations, market predictions, or profit guarantees."
FALLBACK = "I do not have enough relevant help-center information to answer this request. Please contact support for human review."
ESCALATION_NOTE = "Please contact support for a secure account-status review; I cannot determine your account's specific status from this message."
INTRO = {
    "deposit_issue": "For your deposit issue, check the following help-center guidance:",
    "password_reset": "For the missing password reset email, follow this help-center guidance:",
    "withdrawal_issue": "For your declined withdrawal, these are possible reasons and limitations from the help center:",
    "account_lock": "For your locked account, follow this help-center guidance:",
    "other": "The relevant help-center guidance is:",
}


def usable_hits(prediction: Classification, hits: Sequence[Hit]) -> list[Hit]:
    if prediction.intent == "other" or prediction.conflicting:
        return []
    return [hit for hit in hits if hit.score >= MIN_SCORE
            and article_intent(hit.article.title) == prediction.intent]


class ExtractiveGenerator:
    def generate(self, ticket: Ticket, prediction: Classification,
                 hits: Sequence[Hit]) -> str:
        lines = [REFUSAL if prediction.intent == "trading_advice_request"
                 else INTRO[prediction.intent]]
        sources = usable_hits(prediction, hits)
        lines.extend(f"[{hit.article.article_id}] {hit.article.body}" for hit in sources)
        if not sources:
            lines.append(FALLBACK)
        if prediction.intent == "withdrawal_issue":
            lines.append(ESCALATION_NOTE)
        return "\n\n".join(lines)
