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


def plot_age_bmi_interaction(data, output_dir: Path) -> None:
    plot_data = data.copy()
    plot_data["BMI_cat_3"] = pd.cut(
        plot_data["BMI"],
        bins=[0, 18.5, 25, 100],
        labels=["Low (<18.5)", "Normal (18.5-25)", "High (>=25)"],
    )

    plt.figure(figsize=(12, 6))
    sns.pointplot(
        data=plot_data,
        x="Age",
        y="Diabetes_binary",
        hue="BMI_cat_3",
        errorbar=None,
        palette=["#E6A11D", "#2CA02C", "#D62728"],
    )
    plt.title("Interaction of Age and 3-Class BMI on Diabetes Risk", fontsize=10)
    plt.xlabel("Age Group", fontsize=10)
    plt.ylabel("Probability of Diabetes", fontsize=10)
    plt.legend(title="BMI Category")
    plt.grid(True, linestyle="--", alpha=0.6)
    _save_current_figure(output_dir, "age_bmi_interaction.png")


def plot_metabolic_risk_heatmap(data, output_dir: Path) -> None:
    metabolic_pivot = data.pivot_table(
        values="Diabetes_binary",
        index="HighBP",
        columns="HighChol",
        aggfunc="mean",
    )

    metabolic_pivot.index = ["Normal BP", "High BP"]
    metabolic_pivot.columns = ["Normal Chol", "High Chol"]

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        metabolic_pivot,
        annot=True,
        fmt=".1%",
        cmap="Reds",
        cbar_kws={"label": "Probability of Diabetes"},
        annot_kws={"size": 12, "weight": "bold"},
    )
    plt.title("Synergistic Effect of High BP & High Chol on Diabetes", fontsize=12, pad=20)
    plt.yticks(rotation=0, fontsize=12)
    plt.xticks(fontsize=12)
    _save_current_figure(output_dir, "metabolic_risk_heatmap.png")


def plot_lifestyle_vs_diabetes_by_bmi(data, output_dir: Path) -> None:
    import matplotlib.ticker as mtick

    plot_data = data.copy()
    plot_data["BMI_cat_3"] = pd.cut(
        plot_data["BMI"],
        bins=[0, 18.5, 25, 100],
        labels=["Low (<18.5)", "Normal (18.5-25)", "High (>=25)"],
    )
    plot_data["Non_Smoker"] = 1 - plot_data["Smoker"]
    plot_data["Moderate_Alcohol"] = 1 - plot_data["HvyAlcoholConsump"]
    plot_data["Lifestyle_Score"] = (
        plot_data["PhysActivity"]
        + plot_data["Fruits"]
        + plot_data["Veggies"]
        + plot_data["Non_Smoker"]
        + plot_data["Moderate_Alcohol"]
    )

    plt.figure(figsize=(12, 5))
    ax = sns.lineplot(
        data=plot_data,
        x="Lifestyle_Score",
        y="Diabetes_binary",
        hue="BMI_cat_3",
        palette=["#E6A11D", "#2CA02C", "#D62728"],
        marker="o",
        markersize=10,
        linewidth=3.5,
        errorbar=None,
    )

    plt.title("Can a Perfect Lifestyle Mitigate High BMI Risk?", fontsize=12, pad=15)
    plt.xlabel("Healthy Lifestyle Score (0 = Worst, 5 = Best)", fontsize=12)
    plt.ylabel("Probability of Diabetes", fontsize=14)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.legend(title="BMI Category", title_fontsize=12, fontsize=11, loc="upper right", frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.grid(axis="x", alpha=0.0)
    sns.despine()
    _save_current_figure(output_dir, "lifestyle_vs_diabetes_by_bmi.png")


def plot_income_diabetes_by_education(data, output_dir: Path) -> None:
    import matplotlib.ticker as mtick

    subset_edu = data[data["Education"].isin([2, 4, 6])].copy()

    edu_map = {
        2: "Elementary/Middle",
        4: "High School",
        6: "College Graduate",
    }
    subset_edu["Education_Label"] = subset_edu["Education"].map(edu_map)

    plt.figure(figsize=(10, 6))
    ax = sns.pointplot(
        data=subset_edu,
        x="Income",
        y="Diabetes_binary",
        hue="Education_Label",
        errorbar=None,
        palette="magma",
        markers=["o", "s", "D"],
        linestyles=["-", "--", "-."],
    )

    plt.title("The Buffering Effect: How Education Mitigates Income-Related Health Risks", fontsize=12)
    plt.xlabel("Income Level (1 = Lowest, 8 = Highest)", fontsize=12)
    plt.ylabel("Probability of Diabetes", fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.legend(title="Education Level", fontsize=11, title_fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    sns.despine()
    _save_current_figure(output_dir, "income_diabetes_by_education.png")


def plot_bmi_age_heatmap(data, output_dir: Path) -> None:
    bmi_age_df = data.copy()

    bmi_age_df["BMI_Group"] = pd.cut(
        bmi_age_df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
        include_lowest=True,
    )

    bmi_age_pivot = bmi_age_df.pivot_table(
        values="Diabetes_binary",
        index="BMI_Group",
        columns="Age",
        aggfunc="mean",
    )

    plt.figure(figsize=(12, 5))
    sns.heatmap(bmi_age_pivot, annot=True, fmt=".2f", cmap="YlOrRd")
    plt.title("Diabetes Rate by BMI Group and Age")
    plt.xlabel("Age Category")
    plt.ylabel("BMI Group")
    _save_current_figure(output_dir, "bmi_age_heatmap.png")


def plot_risk_lift(data, output_dir: Path) -> None:
    binary_features = [
        "HighBP",
        "HighChol",
        "Stroke",
        "HeartDiseaseorAttack",
        "DiffWalk",
        "PhysActivity",
        "Smoker",
        "HvyAlcoholConsump",
    ]

    risk_lift = []

    for col in binary_features:
        temp = data.groupby(col)["Diabetes_binary"].mean()
        if 0 in temp.index and 1 in temp.index:
            risk_lift.append(
                {
                    "Feature": col,
                    "Rate_if_0": temp.loc[0],
                    "Rate_if_1": temp.loc[1],
                    "Absolute_Lift": temp.loc[1] - temp.loc[0],
                }
            )

    risk_lift_df = pd.DataFrame(risk_lift).sort_values("Absolute_Lift", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=risk_lift_df, x="Absolute_Lift", y="Feature")
    plt.title("Absolute Increase in Diabetes Rate When Risk Factor = 1")
    plt.xlabel("Absolute Lift in Diabetes Rate")
    plt.ylabel("Feature")
    _save_current_figure(output_dir, "risk_lift_bar.png")


def plot_education_income_trend(data, output_dir: Path) -> None:
    edu_income_trend = data.groupby(["Education", "Income"])["Diabetes_binary"].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=edu_income_trend,
        x="Education",
        y="Diabetes_binary",
        hue="Income",
        marker="o",
    )

    plt.title("Diabetes Rate Across Education Levels by Income Group")
    plt.xlabel("Education Level")
    plt.ylabel("Diabetes Rate")
    _save_current_figure(output_dir, "education_income_trend.png")


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
    plot_age_bmi_interaction(data, output_dir)
    plot_metabolic_risk_heatmap(data, output_dir)
    plot_lifestyle_vs_diabetes_by_bmi(data, output_dir)
    plot_income_diabetes_by_education(data, output_dir)
    plot_bmi_age_heatmap(data, output_dir)
    plot_risk_lift(data, output_dir)
    plot_education_income_trend(data, output_dir)


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