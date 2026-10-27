from clearml import Dataset

BASE_ID = "407ddb2597b344329bdff9b66294510f"

dataset = Dataset.create(
    dataset_project="MIIT/MLOps/practice_3/shandova",
    dataset_name="sklearn-tabular-shandova",
    dataset_version="1.1.0",
    description="Версия Iris с добавленным датасетом Wine.",
    parent_datasets=[BASE_ID],
)

dataset.add_files(path="data/wine.csv")
dataset.upload()
dataset.finalize()

print("WINE_ID =", dataset.id)