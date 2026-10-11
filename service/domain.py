from dataclasses import dataclass
from typing import Protocol, Sequence


INTENTS = frozenset({
    "deposit_issue", "password_reset", "withdrawal_issue", "account_lock",
    "trading_advice_request", "other",
})


@dataclass(frozen=True)
class Ticket:
    id: str
    message: str
    language: str


@dataclass(frozen=True)
class Article:
    article_id: str
    title: str
    body: str


@dataclass(frozen=True)
class Classification:
    intent: str
    certainty: float
    signals: tuple[str, ...]
    conflicting: bool = False


@dataclass(frozen=True)
class Hit:
    article: Article
    score: float


class Classifier(Protocol):
    def classify(self, ticket: Ticket) -> Classification: ...


class Retriever(Protocol):
    def index(self, articles: Sequence[Article]) -> None: ...
    def retrieve(self, query: str, top_k: int = 3) -> list[Hit]: ...


class Generator(Protocol):
    def generate(self, ticket: Ticket, prediction: Classification,
                 hits: Sequence[Hit]) -> str: ...
