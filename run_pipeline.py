from __future__ import annotations

import argparse
from pathlib import Path

from diabetes_pipeline.data import load_data, resolve_csv_path
from diabetes_pipeline.eda import (
    plot_all_eda,
    plot_confusion_matrix,
)
from diabetes_pipeline.model import evaluate_model, scale_features, split_data, train_logistic_regression
from diabetes_pipeline.preprocess import balance_with_nearmiss, basic_clean, select_model_columns


def run(project_root: Path, output_dir: Path, skip_eda: bool = False) -> None:
    csv_path = resolve_csv_path(project_root=project_root)
    data = load_data(csv_path)
    data = basic_clean(data)

    if not skip_eda:
        plot_all_eda(data, output_dir)

    model_data = select_model_columns(data)
    x_sm, y_sm = balance_with_nearmiss(model_data)
    x_train, x_test, y_train, y_test = split_data(x_sm, y_sm)
    x_train, x_test, _ = scale_features(x_train, x_test)

    model = train_logistic_regression(x_train, y_train)
    report, cm = evaluate_model(model, x_test, y_test)

    print("Classification report:")
    print(report)
    plot_confusion_matrix(cm, output_dir)
    print(f"Saved outputs to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run diabetes pipeline extracted from pro.ipynb")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA plot generation.")
    args = parser.parse_args()

    run(project_root=args.project_root, output_dir=args.output_dir, skip_eda=args.skip_eda)


if __name__ == "__main__":
    main()
