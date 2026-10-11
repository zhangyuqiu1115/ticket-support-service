"""Run: python main.py"""
import argparse
import json
from pathlib import Path

from service.pipeline import Pipeline, write_results


BASE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline customer support ticket pipeline")
    parser.add_argument("--data-dir", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, default=BASE)
    parser.add_argument("--top-k", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    try:
        results, debug = Pipeline().run(args.data_dir, args.top_k)
        write_results(args.output_dir, results, debug)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(1, f"Pipeline failed: {error}\n")
    print(f"Processed {len(results)} tickets. Outputs: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
