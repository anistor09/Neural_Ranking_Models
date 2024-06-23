from nomic.dense_indexers.dense_index_one_dataset import index_nomic_v1_collection
from func_timeout import func_timeout

directory = "nomic"

lengths = []
datasets = []
datasets.append("irds:msmarco-passage/trec-dl-2019")
lengths.append(7)


def index_nomic_collections():
    """
       This function indexes multiple datasets using the Nomic version 1 encoder.
    """
    for index, dataset_name in enumerate(datasets):
        try:
            index_nomic_v1_collection(dataset_name, lengths[index], directory)
            print(dataset_name + " DONE")
        except Exception as e:
            # Handles any other exceptions
            print(f"An error occurred: {e}")


def main():
    """
        Main function that executes the indexing of multiple datasets with Nomic V1 model. It sets a timeout for the
            entire indexing process for easier debugging on the SuperComputer.
       """
    try:
        func_timeout(24 * 3600 - 15 * 60, index_nomic_collections)
    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
