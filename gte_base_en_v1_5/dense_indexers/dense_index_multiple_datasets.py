from gte_base_en_v1_5.dense_indexers.dense_index_one_dataset import index_gte_base_collection

datasets = ["quora", "fever"]
lengths = [6, 221]
prefix_dataset = "irds:beir/"
directory = "gte_base_en_v1_5"  # Directory to store the indexed files


def index_gte_base_collections():
    """
       Indexes multiple datasets using the GTE base encoder. This function iterates through a list of datasets,
       constructing full dataset names with prefixes and indexing each using predefined ID lengths.
       """
    for index, dataset_name in enumerate(datasets):
        index_gte_base_collection(prefix_dataset + dataset_name, lengths[index], directory)


def main():
    """
        Main function that executes the indexing of multiple datasets. It handles exceptions that might occur during
        the indexing process and logs them, ensuring that all possible errors are noted and can be addressed.
    """
    try:
        index_gte_base_collections()
    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
