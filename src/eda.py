"""Step 2 - Exploratory Data Analysis.

Loads the dataset, prints summary statistics, and saves visualizations
to ``outputs/eda/``. Run with: python -m src.eda
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import EDA_DIR, TARGET
from src.data_loader import load_raw_data

sns.set_theme(style="whitegrid")


def setup_output_dir() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name: str) -> None:
    path = EDA_DIR / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.relative_to(EDA_DIR.parents[1])}")


def basic_overview(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("1) BASIC OVERVIEW")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicate records:", int(df.duplicated().sum()))
    print("\nFirst 5 rows:\n", df.head())


def numerical_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("2) NUMERICAL SUMMARY")
    print("=" * 60)
    num_cols = df.select_dtypes(include="number").columns
    print(df[num_cols].describe().T)


def categorical_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("3) CATEGORICAL SUMMARY")
    print("=" * 60)
    for col in df.select_dtypes(include=["object", "str", "string"]).columns:
        print(f"\n{col}:\n{df[col].value_counts()}")


def plot_price_distribution(df: pd.DataFrame) -> None:
    print("\nGenerating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df[TARGET], kde=True, ax=axes[0])
    axes[0].set_title("Price Distribution")
    sns.boxplot(x=df[TARGET], ax=axes[1])
    axes[1].set_title("Price Boxplot")
    save_fig(fig, "1_price_distribution.png")


def plot_numerical_vs_price(df: pd.DataFrame) -> None:
    numeric = ["Area", "Bedrooms", "Bathrooms", "Age"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col in zip(axes.ravel(), numeric):
        sns.scatterplot(data=df, x=col, y=TARGET, alpha=0.6, ax=ax)
        ax.set_title(f"{col} vs {TARGET}")
    fig.tight_layout()
    save_fig(fig, "2_numeric_vs_price.png")


def plot_categorical_vs_price(df: pd.DataFrame) -> None:
    categorical = ["Location", "Property_Type"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col in zip(axes.ravel(), categorical):
        sns.boxplot(data=df, x=col, y=TARGET, ax=ax)
        ax.set_title(f"{col} vs {TARGET}")
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    save_fig(fig, "3_categorical_vs_price.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    save_fig(fig, "4_correlation_heatmap.png")


def main() -> None:
    setup_output_dir()
    df = load_raw_data()

    basic_overview(df)
    numerical_summary(df)
    categorical_summary(df)

    plot_price_distribution(df)
    plot_numerical_vs_price(df)
    plot_categorical_vs_price(df)
    plot_correlation_heatmap(df)

    print("\nEDA complete. Plots saved to outputs/eda/")


if __name__ == "__main__":
    main()
