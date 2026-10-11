"""Validate existing artifacts and run edge-case tests: python validate.py."""
import argparse
import json
import unittest
from pathlib import Path

from service.classification import requests_trading_advice
from service.pipeline import Pipeline, load_data
from service.validation import validate_result


BASE = Path(__file__).resolve().parent


def check_artifacts(data_dir: Path, output_dir: Path, top_k: int) -> None:
    for path in (data_dir / "tickets.json", data_dir / "kb_articles.json",
                 output_dir / "results.json", output_dir / "debug_report.json", BASE / "README.md"):
        if not path.is_file():
            raise ValueError(f"Required artifact missing: {path.name}; run python main.py first")
    tickets, articles = load_data(data_dir)
    known = {article.article_id: article for article in articles}
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    debug = json.loads((output_dir / "debug_report.json").read_text(encoding="utf-8"))
    if not isinstance(results, list) or len(results) != len(tickets):
        raise ValueError("There must be exactly one result per ticket")
    for ticket, result in zip(tickets, results):
        validate_result(result, known, ticket)
        if requests_trading_advice(ticket.message) and result["intent"] != "trading_advice_request":
            raise ValueError(f"{ticket.id}: advice request was not refused")
    first, first_debug = Pipeline().run(data_dir, top_k)
    second, second_debug = Pipeline().run(data_dir, top_k)
    if results != first or first != second or debug != first_debug or first_debug != second_debug:
        raise ValueError("Artifacts must match reproducible pipeline output; rerun python main.py")
    print(f"PASS: {len(tickets)} results; valid schema, known IDs, safe refusals, reproducible results and debug report.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate saved ticket pipeline artifacts and tests")
    parser.add_argument("--data-dir", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, default=BASE)
    parser.add_argument("--top-k", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    try:
        check_artifacts(args.data_dir, args.output_dir, args.top_k)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Validation failed: {error}\n")
    suite = unittest.defaultTestLoader.discover(str(BASE / "tests"), top_level_dir=str(BASE))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
