from sparse_index_one_dataset import index_one


def index_multiple(prefix_dataset, model_name, datasets):
    for dataset_name in datasets:
        try:
            index_one(prefix_dataset, dataset_name, model_name)
            print(dataset_name + " DONE")
        except Exception as e:

            print(f"An error occurred: {e}")
