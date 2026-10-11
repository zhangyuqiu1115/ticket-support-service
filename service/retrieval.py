"""Deterministic word TF-IDF / cosine retrieval, implemented with stdlib."""
import math
import re
from collections import Counter
from typing import Sequence

from .domain import Article, Hit


STOPWORDS = frozenset("a an the and or but by for of to in on with is are was be been it i my me you your can could do does did what why how so still after before up too many not no".split())
ALIASES = {
    "deposited": "deposit", "deposits": "deposit", "depositing": "deposit",
    "withdrawals": "withdrawal", "withdraw": "withdrawal", "withdrawing": "withdrawal",
    "locked": "lock", "locking": "lock", "locks": "lock",
    "declined": "decline", "declines": "decline", "rejected": "decline",
    "payments": "payment", "predictions": "prediction", "predict": "prediction",
    "recommendations": "recommendation", "recommend": "recommendation",
    "profits": "profit", "assets": "asset", "attempts": "attempt",
    "checks": "check", "checking": "check", "failed": "fail",
    "delayed": "delay", "delays": "delay", "receiving": "receive",
    "forgot": "password", "forgotten": "password",
}


def tokenize(text: str) -> list[str]:
    text = re.sub(r"sign[ -]?in|log[ -]?in", "login", text.lower())
    return [ALIASES.get(word, word) for word in re.findall(r"[a-z0-9]+", text)
            if word not in STOPWORDS]


class TfidfRetriever:
    def __init__(self) -> None:
        self.articles: list[Article] = []
        self.idf: dict[str, float] = {}
        self.vectors: list[dict[str, float]] = []

    def index(self, articles: Sequence[Article]) -> None:
        if not articles:
            raise ValueError("Knowledge base must contain at least one article")
        self.articles = sorted(articles, key=lambda article: article.article_id)
        documents = [tokenize(a.title + " " + a.body) for a in self.articles]
        frequency = Counter(token for doc in documents for token in set(doc))
        self.idf = {token: math.log((1 + len(documents)) / (1 + count)) + 1
                    for token, count in frequency.items()}
        self.vectors = [self._vector(doc) for doc in documents]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        weights = {token: (1 + math.log(count)) * self.idf[token]
                   for token, count in Counter(tokens).items() if token in self.idf}
        norm = math.sqrt(sum(value * value for value in weights.values()))
        return {token: value / norm for token, value in weights.items()} if norm else {}

    def retrieve(self, query: str, top_k: int = 3) -> list[Hit]:
        if not self.articles:
            raise ValueError("INDEX_KB must run before RETRIEVE_CONTEXT")
        if type(top_k) is not int or not 1 <= top_k <= 3:
            raise ValueError("top_k must be an integer from 1 to 3")
        vector = self._vector(tokenize(query))
        hits = [Hit(article, sum(vector.get(token, 0.0) * weight
                                 for token, weight in document.items()))
                for article, document in zip(self.articles, self.vectors)]
        return sorted(hits, key=lambda hit: (-hit.score, hit.article.article_id))[:top_k]
