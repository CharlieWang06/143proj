from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .preprocess import get_percentage_table


def _save_current_figure(output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_histograms(data, output_dir: Path) -> None:
    data.hist(figsize=(20, 15))
    _save_current_figure(output_dir, "histograms.png")


def plot_correlation_heatmap(data, output_dir: Path) -> None:
    plt.figure(figsize=(20, 10))
    sns.heatmap(data.corr(), annot=True, cmap="YlOrRd")
    plt.title("Correlation of Features")
    _save_current_figure(output_dir, "correlation_heatmap.png")


def plot_correlation_with_target(data, output_dir: Path) -> None:
    data.drop("Diabetes_binary", axis=1).corrwith(data.Diabetes_binary).plot(
        kind="bar",
        grid=True,
        figsize=(20, 8),
        title="Correlation with Diabetes_binary",
        color="purple",
    )
    _save_current_figure(output_dir, "target_correlation_bar.png")


def plot_highbp_highchol_stacked(data, output_dir: Path) -> None:
    cols = ["HighBP", "HighChol"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    axes = ax.ravel()

    for i, col in enumerate(cols):
        get_percentage_table(data, col).plot(kind="bar", stacked=True, ax=axes[i])
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Percentage")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "highbp_highchol_stacked.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diabetes_pie(data, output_dir: Path) -> None:
    labels = ["non-Diabetic", "Diabetic"]
    plt.figure(figsize=(6, 6))
    plt.pie(data["Diabetes_binary"].value_counts(), labels=labels, autopct="%.02f")
    plt.title("Diabetes Class Distribution")
    _save_current_figure(output_dir, "diabetes_pie.png")


def plot_bmi_vs_diabetes(data, output_dir: Path) -> None:
    plt.figure(figsize=(30, 15))
    sns.countplot(
        data=data,
        x="BMI",
        hue="Diabetes_binary",
        palette={0: "r", 1: "g"},
    )
    plt.title("Relation between BMI and Diabetes")
    plt.legend(title="", labels=["No Diabetic", "Diabetic"])
    _save_current_figure(output_dir, "bmi_countplot.png")


def plot_genhlth_vs_diabetes(data, output_dir: Path) -> None:
    pd.crosstab(data.GenHlth, data.Diabetes_binary).plot(
        kind="bar", figsize=(30, 12), color=["Purple", "Green"]
    )
    plt.title("Diabetes Disease Frequency for GenHlth")
    plt.xlabel("GenHlth")
    plt.xticks(rotation=0)
    plt.ylabel("Frequency")
    _save_current_figure(output_dir, "genhlth_bar.png")


def plot_age_vs_diabetes(data, output_dir: Path) -> None:
    pd.crosstab(data.Age, data.Diabetes_binary).plot(kind="bar", figsize=(20, 6))
    plt.title("Diabetes Disease Frequency for Ages")
    plt.xlabel("Age (increasing age group number)")
    plt.xticks(rotation=0)
    plt.ylabel("Frequency")
    _save_current_figure(output_dir, "age_bar.png")


def plot_education_distribution(data, output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data[data.Diabetes_binary == 0], x="Education", color="y", label="No Diabetic")
    sns.kdeplot(data=data[data.Diabetes_binary == 1], x="Education", color="m", label="Diabetic")
    plt.title("Relation between Education and Diabetes")
    plt.legend()
    _save_current_figure(output_dir, "education_kde.png")


def plot_income_distribution(data, output_dir: Path) -> None:
    g = sns.displot(data=data, x="Income", hue="Diabetes_binary", kind="kde", height=6, aspect=2)
    g.figure.suptitle("Relation between Income and Diabetes", y=1.02)
    output_dir.mkdir(parents=True, exist_ok=True)
    g.figure.savefig(output_dir / "income_kde.png", dpi=150, bbox_inches="tight")
    plt.close(g.figure)


def plot_all_eda(data, output_dir: Path) -> None:
    """Generate and save all EDA images used in pro.ipynb."""
    plot_histograms(data, output_dir)
    plot_correlation_heatmap(data, output_dir)
    plot_correlation_with_target(data, output_dir)
    plot_highbp_highchol_stacked(data, output_dir)
    plot_diabetes_pie(data, output_dir)
    plot_bmi_vs_diabetes(data, output_dir)
    plot_genhlth_vs_diabetes(data, output_dir)
    plot_age_vs_diabetes(data, output_dir)
    plot_education_distribution(data, output_dir)
    plot_income_distribution(data, output_dir)


def plot_confusion_matrix(cm, output_dir: Path) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    _save_current_figure(output_dir, "confusion_matrix.png")
