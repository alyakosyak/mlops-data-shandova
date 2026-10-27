from clearml import Dataset

dataset = Dataset.create(
    dataset_project="MIIT/MLOps/practice_3/shandova",
    dataset_name="sklearn-tabular-shandova",
    dataset_version="1.0.0",
    description="Исходный датасет Iris из scikit-learn, 150 строк, 5 столбцов.",
)

dataset.add_files(path="data/iris.csv")
dataset.add_tags(["base"])
dataset.upload()
dataset.finalize()

print("BASE_ID =", dataset.id)