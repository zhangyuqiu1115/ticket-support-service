import math
from typing import Any, Mapping, Sequence

from .classification import requests_trading_advice
from .domain import Article, Classification, Hit, INTENTS, Ticket
from .generation import ESCALATION_NOTE, FALLBACK, INTRO, REFUSAL, usable_hits


FIELDS = frozenset({"ticket_id", "intent", "retrieved_articles", "reply_draft",
                    "grounded", "confidence", "needs_human_escalation", "safety_flags"})


def supported_reply(reply: str, intent: str, retrieved: Sequence[str],
                    articles: Mapping[str, Article]) -> tuple[bool, bool]:
    """Allow only approved nonfactual framing and exact, cited article bodies.

    This deliberately restricts the offline implementation. A future paraphrasing
    generator must also supply an appropriately stronger grounding validator.
    """
    lines = reply.split("\n\n")
    expected_intro = REFUSAL if intent == "trading_advice_request" else INTRO[intent]
    if not lines or lines[0] != expected_intro:
        return False, False
    source_lines = {f"[{key}] {articles[key].body}" for key in retrieved}
    allowed = source_lines | {FALLBACK, ESCALATION_NOTE}
    supported = all(line in allowed for line in lines[1:]) and len(lines) > 1
    grounded = supported and any(line in source_lines for line in lines[1:])
    return supported, grounded


def validate_result(result: Any, articles: Mapping[str, Article],
                    ticket: Ticket | None = None) -> None:
    if not isinstance(result, dict) or set(result) != FIELDS:
        raise ValueError("Result must contain exactly the required schema fields")
    for key in ("ticket_id", "intent", "reply_draft"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if result["intent"] not in INTENTS:
        raise ValueError("Unknown intent")
    for key in ("grounded", "needs_human_escalation"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    score = result["confidence"]
    if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("confidence must be a finite number between 0 and 1")
    ids = result["retrieved_articles"]
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 3
            or any(not isinstance(key, str) or key not in articles for key in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("retrieved_articles must contain 1 to 3 unique known IDs")
    flags = result["safety_flags"]
    if (not isinstance(flags, list)
            or any(not isinstance(flag, str) or not flag.strip() for flag in flags)
            or len(flags) != len(set(flags))):
        raise ValueError("safety_flags must contain unique nonempty strings")
    if ticket is not None:
        if result["ticket_id"] != ticket.id:
            raise ValueError("Result ticket_id does not match its input")
        if requests_trading_advice(ticket.message) and result["intent"] != "trading_advice_request":
            raise ValueError("Trading advice request must be classified and refused")
    if result["intent"] == "trading_advice_request":
        if not result["reply_draft"].startswith(REFUSAL + "\n\n") or "advice_request" not in flags:
            raise ValueError("Trading advice requests must include the explicit refusal and flag")
    supported, has_sources = supported_reply(result["reply_draft"], result["intent"], ids, articles)
    if result["grounded"] and (not supported or not has_sources):
        raise ValueError("grounded=true requires exact retrieved source evidence")
    if not supported and not ("unsupported_claim" in flags or result["needs_human_escalation"]):
        raise ValueError("Unsupported claims require a safety flag or escalation")
    if not result["grounded"] and not ("insufficient_grounding" in flags and result["needs_human_escalation"]):
        raise ValueError("Ungrounded output must be flagged and escalated")
    if score < 0.55 and not ("low_confidence" in flags and result["needs_human_escalation"]):
        raise ValueError("Low confidence must be flagged and escalated")
    if result["intent"] == "withdrawal_issue" and not (
            "policy_sensitive" in flags and result["needs_human_escalation"]):
        raise ValueError("Withdrawal review must be flagged and escalated")


def finalize(ticket: Ticket, prediction: Classification, hits: Sequence[Hit],
             draft: str, articles: Mapping[str, Article]) -> tuple[dict, dict]:
    if prediction.intent not in INTENTS or not math.isfinite(prediction.certainty) or not 0 <= prediction.certainty <= 1:
        raise ValueError("Classifier returned an invalid intent or certainty")
    flags: set[str] = set()
    reasons: list[str] = []
    ids = [hit.article.article_id for hit in hits]
    supported, grounded = supported_reply(draft, prediction.intent, ids, articles)
    sources = usable_hits(prediction, hits)
    # A zero-score or unrelated quote cannot make the answer grounded.
    grounded = grounded and bool(sources) and any(
        f"[{hit.article.article_id}] {hit.article.body}" in draft.split("\n\n")
        for hit in sources)
    quality = min(1.0, max((hit.score for hit in sources), default=0.0) / 0.45)
    confidence = round(min(prediction.certainty, quality), 4)
    if not supported or not grounded:
        # Reject unverified model prose and publish a conservative fallback.
        draft = (REFUSAL if prediction.intent == "trading_advice_request"
                 else INTRO[prediction.intent]) + "\n\n" + FALLBACK
        grounded = False
        if not supported:
            flags.add("unsupported_claim")
            reasons.append("Generated text failed the exact-source grounding check; replaced with fallback.")
        flags.add("insufficient_grounding")
        reasons.append("No sufficiently relevant, intent-aligned source supports the answer.")
        confidence = min(confidence, 0.20)
    if prediction.conflicting:
        flags.add("conflicting_intents")
        reasons.append("Multiple independent intent signals require human triage.")
    if ticket.language.lower().split("-")[0] != "en":
        flags.add("unsupported_language")
        reasons.append("The offline classifier supports English only.")
        confidence = min(confidence, 0.20)
    if prediction.intent == "trading_advice_request":
        flags.add("advice_request")
    if prediction.intent == "withdrawal_issue":
        flags.update({"policy_sensitive", "account_specific_request"})
        reasons.append("Withdrawal declines require secure account-status review.")
    if confidence < 0.55:
        flags.add("low_confidence")
        reasons.append("Confidence is below the 0.55 handoff threshold.")
    result = {
        "ticket_id": ticket.id, "intent": prediction.intent,
        "retrieved_articles": ids, "reply_draft": draft, "grounded": grounded,
        "confidence": confidence, "needs_human_escalation": bool(reasons),
        "safety_flags": sorted(flags),
    }
    validate_result(result, articles, ticket)
    debug = {
        "ticket_id": ticket.id, "predicted_intent": prediction.intent,
        "classifier_signals": list(prediction.signals),
        "classifier_certainty": prediction.certainty,
        "retrieved_articles": [{"article_id": h.article.article_id,
                                "title": h.article.title, "score": round(h.score, 6)} for h in hits],
        "quoted_article_ids": [h.article.article_id for h in sources] if grounded else [],
        "retrieval_quality": round(quality, 6), "confidence": confidence,
        "needs_human_escalation": bool(reasons),
        "escalation_reasons": reasons or ["Relevant grounded guidance and sufficient rule/retrieval confidence."],
        "safety_flags": sorted(flags),
    }
    return result, debug
