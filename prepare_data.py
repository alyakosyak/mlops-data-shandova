
from pathlib import Path
import argparse

import pandas as pd
from sklearn.datasets import load_iris, load_wine


DATA_DIR = Path("data")

CSV_OPTIONS = {
    "index": False,
    "lineterminator": "\n",
    "float_format": "%.10g",
}


def load_dataset(name):
    if name == "iris":
        dataset = load_iris(as_frame=True)
    else:
        dataset = load_wine(as_frame=True)

    df = dataset.frame.copy()

    # Сортируем по всем исходным признакам и target.
    columns = list(df.columns)
    df = df.sort_values(
        by=columns,
        kind="mergesort"
    ).reset_index(drop=True)

    return df


def prepare_base():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    iris = load_dataset("iris")
    iris.to_csv(DATA_DIR / "iris.csv", **CSV_OPTIONS)


def prepare_wine():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    wine = load_dataset("wine")
    wine.to_csv(DATA_DIR / "wine.csv", **CSV_OPTIONS)


def prepare_iris_features():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    iris = load_dataset("iris")

    iris["sepal_area"] = (
        iris["sepal length (cm)"] *
        iris["sepal width (cm)"]
    )

    iris["petal_area"] = (
        iris["petal length (cm)"] *
        iris["petal width (cm)"]
    )

    iris.to_csv(DATA_DIR / "iris.csv", **CSV_OPTIONS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["base", "wine", "iris_features"]
    )
    args = parser.parse_args()

    if args.mode == "base":
        prepare_base()
    elif args.mode == "wine":
        prepare_wine()
    elif args.mode == "iris_features":
        prepare_iris_features()


if __name__ == "__main__":
    main()
