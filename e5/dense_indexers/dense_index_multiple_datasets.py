from e5.dense_indexers.dense_index_one_dataset import index_e5_small_collection, index_e5_base_collection, \
    index_e5_base_unsupervised_collection
from func_timeout import func_timeout

directory = "e5"

lengths = []
datasets = []
datasets.append("irds:msmarco-passage/trec-dl-2019")
lengths.append(7)


def index_e5_base_collections():
    """
       Function to index datasets using the E5 base model.
    """
    for index, dataset_name in enumerate(datasets):
        try:
            index_e5_base_collection(dataset_name, lengths[index], directory)
            print(dataset_name + " DONE")
        except Exception as e:
            # Handles any other exceptions
            print(f"An error occurred: {e}")


def index_e5_base_unsupervised_collections():
    """
        Function to index datasets using the E5 base pretrained only model.
     """
    for index, dataset_name in enumerate(datasets):
        try:
            index_e5_base_unsupervised_collection(dataset_name, lengths[index], directory)
            print(dataset_name + " DONE")
        except Exception as e:
            # Handles any other exceptions
            print(f"An error occurred: {e}")


def index_e5_small_collections():
    """
          Function to index datasets using the E5 small model.
    """
    for index, dataset_name in enumerate(datasets):
        try:
            index_e5_small_collection(dataset_name, lengths[index], directory)
            print(dataset_name + " DONE")
        except Exception as e:
            # Handles any other exceptions
            print(f"An error occurred: {e} at dataset {dataset_name}")


def main():
    """
         Main function to execute the indexing of datasets. It sets a timeout for the entire indexing process for easier
         debugging on the SuperComputer.
    """
    try:
        func_timeout(24 * 3600 - 15 * 60, index_e5_small_collections)
    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
