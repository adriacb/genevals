"""genevals CLI: run an eval from a dataset + Python target callables, or inspect the metric catalog."""

from __future__ import annotations

import argparse
import importlib
import sys

from genevals.core.dataset import Dataset
from genevals.evaluator import Evaluator
from genevals.metrics.catalog import CATALOG


def _cmd_catalog(_: argparse.Namespace) -> None:
    for entry in CATALOG.describe():
        ref = " (needs reference)" if entry["needs_reference"] else ""
        print(f"{entry['name']:<20} [{entry['category']}/{entry['modality']}]{ref}  {entry['description']}")


def _load_attr(spec: str):
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"expected 'module:attribute', got {spec!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _cmd_run(args: argparse.Namespace) -> None:
    dataset = Dataset.load(args.dataset)
    targets = [_load_attr(spec) for spec in args.target]
    metrics = [CATALOG.get(name) for name in args.metric]
    report = Evaluator(dataset, targets, metrics, concurrency=args.concurrency).run()

    if args.json:
        report.to_json(args.json)
    if args.yaml:
        report.to_yaml(args.yaml)
    if args.html:
        report.to_html(args.html)
    if not (args.json or args.yaml or args.html):
        print(report.to_json())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="genevals")
    sub = parser.add_subparsers(required=True)

    p_catalog = sub.add_parser("catalog", help="list registered metrics")
    p_catalog.set_defaults(func=_cmd_catalog)

    p_run = sub.add_parser("run", help="run an eval")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--target", action="append", required=True, help="module:attribute, repeatable")
    p_run.add_argument("--metric", action="append", required=True, help="name from `genevals catalog`, repeatable")
    p_run.add_argument("--concurrency", type=int, default=8)
    p_run.add_argument("--json")
    p_run.add_argument("--yaml")
    p_run.add_argument("--html")
    p_run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
