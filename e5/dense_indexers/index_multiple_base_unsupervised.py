from func_timeout import func_timeout
from e5.dense_indexers.dense_index_multiple_datasets import index_e5_base_unsupervised_collections


def main():
    try:
        func_timeout(24 * 3600 - 15 * 60, index_e5_base_unsupervised_collections)
    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
