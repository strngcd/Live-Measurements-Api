"""Command line for the dataset and benchmark kit: python -m benchmark <command> --help."""
import argparse
import json
import os
import sys

from .dataset import DEFAULT_SPLIT_FRACTIONS, SPLITS, Dataset, assign_splits, create_dataset
from .definitions import load_definitions
from .evaluate import DEFAULT_MIN_GROUP_SIZE, compare, comparison_markdown, evaluate, report_markdown
from .schema import data_dictionary


def print_issues(dataset):
    for issue in dataset.issues:
        print(f"{issue.level.upper():7} {issue.where}: {issue.message}")
    errors = len(dataset.errors)
    print(f"{errors} error(s), {len(dataset.issues) - errors} warning(s)")


def load_checked(path):
    """Load and validate a dataset; exit if it has errors."""
    dataset = Dataset(path)
    if dataset.errors:
        print_issues(dataset)
        sys.exit("Fix the errors above first (python -m benchmark validate).")
    return dataset


def write_report(dataset_path, name, report, markdown):
    folder = os.path.join(dataset_path, "reports")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    path = os.path.join(folder, f"{name}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m benchmark", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="create an empty dataset folder").add_argument("dataset")
    commands.add_parser("validate", help="check a dataset folder").add_argument("dataset")

    splits = commands.add_parser("splits", help="assign people without a split to train/calibration/test")
    splits.add_argument("dataset")
    for split in SPLITS:
        splits.add_argument(f"--{split}", type=float, default=DEFAULT_SPLIT_FRACTIONS[split],
                            help=f"target share for {split} (default {DEFAULT_SPLIT_FRACTIONS[split]})")

    commands.add_parser("predict", help="run the current API on every usable sample").add_argument("dataset")

    evaluate_cmd = commands.add_parser("evaluate", help="error report for one algorithm version")
    evaluate_cmd.add_argument("dataset")
    evaluate_cmd.add_argument("--version", required=True, help="algorithm version whose predictions to score")
    evaluate_cmd.add_argument("--split", action="append", choices=SPLITS, help="split(s) to score (default: test)")
    evaluate_cmd.add_argument("--tolerances", help="JSON file mapping API output names to tolerances in cm")
    evaluate_cmd.add_argument("--min-group-size", type=int, default=DEFAULT_MIN_GROUP_SIZE)

    compare_cmd = commands.add_parser("compare", help="compare two algorithm versions on the same samples")
    compare_cmd.add_argument("dataset")
    compare_cmd.add_argument("--base", required=True)
    compare_cmd.add_argument("--candidate", required=True)
    compare_cmd.add_argument("--split", action="append", choices=SPLITS, help="split(s) to score (default: test)")

    commands.add_parser("dictionary", help="print the data dictionary as Markdown")

    args = parser.parse_args(argv)

    if args.command == "init":
        create_dataset(args.dataset)
        print(f"Created {args.dataset}. Keep it on an encrypted disk, outside the code repository.")
    elif args.command == "validate":
        dataset = Dataset(args.dataset)
        print_issues(dataset)
        return 1 if dataset.errors else 0
    elif args.command == "splits":
        load_checked(args.dataset)
        assigned = assign_splits(args.dataset, {split: getattr(args, split) for split in SPLITS})
        for person_id, split in sorted(assigned.items()):
            print(f"{person_id}: {split}")
        print(f"Assigned {len(assigned)} new people; existing splits unchanged.")
    elif args.command == "predict":
        from .predict import predict, write_predictions

        dataset = load_checked(args.dataset)
        records = predict(dataset)
        path = write_predictions(args.dataset, records)
        accepted = sum(1 for r in records if r["http_status"] == 200)
        print(f"Wrote {len(records)} predictions ({accepted} accepted) to {path}")
        if records and str(records[0]["code_revision"]).endswith("-dirty"):
            print("Warning: the code has uncommitted changes; commit and bump ALGORITHM_VERSION before reporting.")
    elif args.command == "evaluate":
        from .predict import load_predictions

        dataset = load_checked(args.dataset)
        tolerances = None
        if args.tolerances:
            with open(args.tolerances, encoding="utf-8") as f:
                tolerances = json.load(f)
        splits_used = tuple(args.split or ["test"])
        report = evaluate(dataset, load_predictions(args.dataset, args.version), args.version,
                          splits_used, tolerances, args.min_group_size)
        path = write_report(args.dataset, f"{args.version}-{'-'.join(splits_used)}", report, report_markdown(report))
        print(f"Wrote {path}")
    elif args.command == "compare":
        from .predict import load_predictions

        dataset = load_checked(args.dataset)
        splits_used = tuple(args.split or ["test"])
        result = compare(dataset, load_predictions(args.dataset, args.base), load_predictions(args.dataset, args.candidate),
                         args.base, args.candidate, splits_used)
        name = f"compare-{args.candidate}-vs-{args.base}-{'-'.join(splits_used)}"
        print(f"Wrote {write_report(args.dataset, name, result, comparison_markdown(result))}")
    elif args.command == "dictionary":
        sys.stdout.write(data_dictionary(load_definitions()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
