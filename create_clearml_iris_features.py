from clearml import Dataset

BASE_ID = "407ddb2597b344329bdff9b66294510f"

dataset = Dataset.create(
    dataset_project="MIIT/MLOps/practice_3/shandova",
    dataset_name="sklearn-tabular-shandova",
    dataset_version="2.0.0",
    description="Iris с вычисляемыми признаками sepal_area и petal_area.",
    parent_datasets=[BASE_ID],
)

dataset.add_files(path="data/iris.csv")
dataset.upload()
dataset.finalize()

print("IRIS_FEATURES_ID =", dataset.id)