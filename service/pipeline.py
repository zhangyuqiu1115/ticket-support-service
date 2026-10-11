import json
import os
import tempfile
from pathlib import Path

from .classification import RuleClassifier
from .domain import Article, Classifier, Generator, Retriever, Ticket
from .generation import ExtractiveGenerator
from .retrieval import TfidfRetriever
from .validation import finalize


STAGES = ["LOAD_DATA", "INDEX_KB", "CLASSIFY_INTENT", "RETRIEVE_CONTEXT",
          "GENERATE_REPLY", "VALIDATE_OUTPUT", "WRITE_RESULTS"]


def load_records(path: Path, fields: set[str], id_field: str) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError(f"{path.name} must contain a JSON array")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError(f"{path.name}: incorrect input fields")
        if any(not isinstance(record[key], str) or not record[key].strip() for key in fields):
            raise ValueError(f"{path.name}: fields must be nonempty strings")
        if record[id_field] in seen:
            raise ValueError(f"{path.name}: duplicate {id_field}")
        seen.add(record[id_field])
    return records


def load_data(data_dir: Path) -> tuple[list[Ticket], list[Article]]:
    tickets = [Ticket(**row) for row in load_records(
        data_dir / "tickets.json", {"id", "message", "language"}, "id")]
    articles = [Article(**row) for row in load_records(
        data_dir / "kb_articles.json", {"article_id", "title", "body"}, "article_id")]
    if not articles:
        raise ValueError("Knowledge base must contain at least one article")
    return tickets, articles


class Pipeline:
    def __init__(self, classifier: Classifier | None = None,
                 retriever: Retriever | None = None, generator: Generator | None = None):
        self.classifier = classifier if classifier is not None else RuleClassifier()
        self.retriever = retriever if retriever is not None else TfidfRetriever()
        self.generator = generator if generator is not None else ExtractiveGenerator()

    def run(self, data_dir: Path, top_k: int = 3) -> tuple[list[dict], dict]:
        if type(top_k) is not int or not 1 <= top_k <= 3:
            raise ValueError("top_k must be an integer from 1 to 3")
        tickets, articles = load_data(data_dir)  # LOAD_DATA
        self.retriever.index(articles)          # INDEX_KB
        known = {article.article_id: article for article in articles}
        results, debug = [], []
        for ticket in tickets:
            prediction = self.classifier.classify(ticket)  # CLASSIFY_INTENT
            hits = self.retriever.retrieve(ticket.message, top_k)  # RETRIEVE_CONTEXT
            draft = self.generator.generate(ticket, prediction, hits)  # GENERATE_REPLY
            result, trace = finalize(ticket, prediction, hits, draft, known)  # VALIDATE_OUTPUT
            results.append(result)
            debug.append(trace)
        return results, {"pipeline_stages": STAGES, "top_k": top_k, "tickets": debug}


def write_json(path: Path, value: object) -> None:
    """Replace a single JSON artifact atomically after serialization succeeds."""
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(serialized)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def write_results(output_dir: Path, results: list[dict], debug: dict) -> None:
    # All results have been validated before either output is written.
    write_json(output_dir / "results.json", results)
    write_json(output_dir / "debug_report.json", debug)
