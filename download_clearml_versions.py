from pathlib import Path
import hashlib

from clearml import Dataset
import pandas as pd


VERSIONS = {
    "base": "407ddb2597b344329bdff9b66294510f",
    "wine": "4f7f93b99c814a169b72db9162c98561",
    "iris_features": "42df907bda6d4cb2bf7ce6dc102b34cd",
}

OUTPUT_DIR = Path("clearml_downloads")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


for version_name, dataset_id in VERSIONS.items():
    dataset = Dataset.get(dataset_id=dataset_id)
    target_dir = OUTPUT_DIR / version_name
    target_dir.mkdir(parents=True, exist_ok=True)

    local_path = Path(dataset.get_local_copy())
    print(f"\nВерсия: {version_name}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Файлы в Dataset: {dataset.list_files()}")

    for filename in dataset.list_files():
        source = local_path / filename
        destination = target_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)

        destination.write_bytes(source.read_bytes())

        print(f"\nФайл: {filename}")
        print(f"SHA-256: {sha256(destination)}")

        if destination.suffix == ".csv":
            frame = pd.read_csv(destination)
            print(f"Строк: {len(frame)}")
            print(f"Столбцов: {len(frame.columns)}")
            print(f"Столбцы: {list(frame.columns)}")