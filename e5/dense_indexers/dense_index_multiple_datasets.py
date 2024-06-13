from e5.dense_indexers.dense_index_one_dataset import index_e5_small_collection, index_e5_base_collection, \
    index_e5_base_unsupervised_collection
from func_timeout import func_timeout

# beir_datasets = ["dbpedia-entity", "fever", "fiqa", "hotpotqa", "nfcorpus", "quora", "scifact"]
# lengths = [200, 221, 6, 8, 8, 6, 9]

# beir_datasets = ["nfcorpus", "fiqa", "scifact", "quora", "dbpedia-entity", "hotpotqa", "fever"]
# lengths = [8, 6, 9, 6, 200, 8, 221]
#
# prefix_dataset = "irds:beir/"
directory = "e5"
# datasets = [prefix_dataset + dataset for dataset in beir_datasets]
lengths = []
datasets = []
datasets.append("irds:msmarco-passage/trec-dl-2019")
lengths.append(7)


def index_e5_base_collections():
    for index, dataset_name in enumerate(datasets):
        try:
            index_e5_base_collection(dataset_name, lengths[index], directory)
            print(dataset_name + " DONE")
        except Exception as e:
            # Handles any other exceptions
            print(f"An error occurred: {e}")


def index_e5_base_unsupervised_collections():
    for index, dataset_name in enumerate(datasets):
        try:
            index_e5_base_unsupervised_collection(dataset_name, lengths[index], directory)
            print(dataset_name + " DONE")
        except Exception as e:
            # Handles any other exceptions
            print(f"An error occurred: {e}")


def index_e5_small_collections():
    for index, dataset_name in enumerate(datasets):
        try:
            index_e5_small_collection(dataset_name, lengths[index], directory)
            print(dataset_name + " DONE")
        except Exception as e:
            # Handles any other exceptions
            print(f"An error occurred: {e} at dataset {dataset_name}")


def main():
    try:
        func_timeout(24 * 3600 - 15 * 60, index_e5_small_collections)
    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()