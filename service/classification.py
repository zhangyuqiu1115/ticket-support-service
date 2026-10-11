import re

from .domain import Classification, Ticket


RULES = {
    "deposit_issue": r"\b(?:deposit\w*|top[ -]?up|fund(?:ing)? my account)\b",
    "password_reset": r"\b(?:password|reset (?:email|link)|forgot\w* .*credential\w*)\b",
    "withdrawal_issue": r"\bwithdraw\w*\b|\bcash[ -]?out\b",
    "account_lock": r"\b(?:lock\w*|too many (?:failed )?(?:login|sign[ -]?in)|blocked account)\b",
}
TRADING_SUBJECT = r"\b(?:asset\w*|stock\w*|coin\w*|crypto\w*|bitcoin|market|invest\w*|trad\w*|profit\w*|return\w*|buy|sell)\b"
TRADING_REQUEST = r"\b(?:recommend\w*|predict\w*|forecast\w*|profit\w*|go up|rise|guarantee\w*|which|should .*?(?:buy|sell|invest)|what .*?(?:buy|sell|invest)|make money|advice|best .*?(?:buy|sell|invest))\b"


def requests_trading_advice(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(TRADING_SUBJECT, lowered)
                and re.search(TRADING_REQUEST, lowered))


class RuleClassifier:
    """Conservative English rules; never depend on a ticket's ID."""

    def classify(self, ticket: Ticket) -> Classification:
        text = ticket.message.lower()
        signals = tuple(label for label, pattern in RULES.items()
                        if re.search(pattern, text))
        if requests_trading_advice(text):
            return Classification("trading_advice_request", 0.95,
                                  ("trading_advice_request",) + signals,
                                  bool(signals))
        if ticket.language.lower().split("-")[0] != "en":
            return Classification("other", 0.0, signals)
        if not signals:
            return Classification("other", 0.25, ())
        # Account lock plus password recovery is a common coherent request.
        if set(signals) == {"account_lock", "password_reset"}:
            return Classification("account_lock", 0.80, signals)
        return Classification(signals[0], 0.55 if len(signals) > 1 else 0.95,
                              signals, len(signals) > 1)


def article_intent(title: str) -> str:
    return RuleClassifier().classify(Ticket("kb", title, "en")).intent
