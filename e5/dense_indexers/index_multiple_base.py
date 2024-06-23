from func_timeout import func_timeout
from e5.dense_indexers.dense_index_multiple_datasets import index_e5_base_collections


def main():
    """
          Main function to execute the indexing of datasets with E5 base (finetuned version). It sets a timeout for
          the entire indexing process for easier debugging on the SuperComputer.
     """
    try:
        func_timeout(24 * 3600 - 15 * 60, index_e5_base_collections)
    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
