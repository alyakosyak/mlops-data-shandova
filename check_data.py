
from pathlib import Path
import hashlib
import sys

import pandas as pd
from sklearn.datasets import load_iris, load_wine


DATA_DIR = Path("data")


def sha256_file(path):
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def check_dataset(path, expected_rows, expected_columns=None):
    if not path.exists():
        raise FileNotFoundError(f"Отсутствует файл: {path}")

    df = pd.read_csv(path)

    if any(
        str(column).startswith("Unnamed:")
        for column in df.columns
    ):
        raise ValueError(
            f"{path}: обнаружен безымянный столбец индекса"
        )

    if len(df) != expected_rows:
        raise ValueError(
            f"{path}: ожидалось {expected_rows} строк, "
            f"получено {len(df)}"
        )

    if expected_columns is not None and list(df.columns) != expected_columns:
        raise ValueError(
            f"{path}: неправильный список или порядок столбцов"
        )

    if "sepal_area" in df.columns:
        expected = (
            df["sepal length (cm)"] *
            df["sepal width (cm)"]
        )
        if not (df["sepal_area"] - expected).abs().lt(1e-8).all():
            raise ValueError(f"{path}: неправильные значения sepal_area")

    if "petal_area" in df.columns:
        expected = (
            df["petal length (cm)"] *
            df["petal width (cm)"]
        )
        if not (df["petal_area"] - expected).abs().lt(1e-8).all():
            raise ValueError(f"{path}: неправильные значения petal_area")

    print(f"\nФайл: {path.as_posix()}")
    print(f"SHA-256: {sha256_file(path)}")
    print(f"Строк: {df.shape[0]}")
    print(f"Столбцов: {df.shape[1]}")
    print(f"Столбцы: {df.columns.tolist()}")
    print(f"Пропущенных значений: {df.isna().sum().sum()}")


def main():
    iris_source = load_iris(as_frame=True)
    wine_source = load_wine(as_frame=True)

    iris_columns = list(iris_source.frame.columns)
    wine_columns = list(wine_source.frame.columns)

    iris_path = DATA_DIR / "iris.csv"
    wine_path = DATA_DIR / "wine.csv"

    if not iris_path.exists() and not wine_path.exists():
        raise FileNotFoundError(
            "В data нет ни iris.csv, ни wine.csv"
        )

    if iris_path.exists():
        df = pd.read_csv(iris_path)

        expected_iris_columns = iris_columns.copy()

        if "sepal_area" in df.columns:
            expected_iris_columns.append("sepal_area")

        if "petal_area" in df.columns:
            expected_iris_columns.append("petal_area")

        check_dataset(
            iris_path,
            expected_rows=len(iris_source.frame),
            expected_columns=expected_iris_columns,
        )

    if wine_path.exists():
        check_dataset(
            wine_path,
            expected_rows=len(wine_source.frame),
            expected_columns=wine_columns,
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"ОШИБКА: {error}", file=sys.stderr)
        sys.exit(1)
