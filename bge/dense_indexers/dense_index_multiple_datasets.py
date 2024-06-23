from bge.dense_indexers.dense_index_one_dataset import index_bge_small_collection, index_bge_base_collection
from func_timeout import func_timeout

# List of dataset names and their corresponding ID lengths
beir_datasets = ["dbpedia-entity", "fever", "fiqa", "hotpotqa", "nfcorpus", "quora", "scifact"]
lengths = [200, 221, 6, 8, 8, 6, 9]  # Specific maximum ID lengths for each dataset
prefix_dataset = "irds:beir/"  # Prefix to form full dataset paths
directory = "bge"  # Directory to store indexed files

# Combining prefix with dataset names for full identifiers and adding an additional dataset

datasets = [prefix_dataset + dataset for dataset in beir_datasets]
datasets.append("irds:msmarco-passage/trec-dl-2019")
lengths.append(7)


def index_bge_base_collections():
    """
    Function to index datasets using the BGE Base model.
    """
    for index, dataset_name in enumerate(beir_datasets):
        try:
            index_bge_base_collection(prefix_dataset + dataset_name, lengths[index], directory)
            print(dataset_name + " DONE")
        except Exception as e:
            # Handles any other exceptions
            print(f"An error occurred: {e}")


def index_bge_small_collections():
    """
      Function to index datasets using the BGE Small model.
      """
    for index, dataset_name in enumerate(datasets):
        try:
            index_bge_small_collection(dataset_name, lengths[index], directory)
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
        func_timeout(24 * 3600 - 15 * 60, index_bge_small_collections)
    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
