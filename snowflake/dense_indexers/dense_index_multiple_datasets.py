from snowflake.dense_indexers.dense_index_one_dataset import index_snowflake_m_collection
from snowflake.dense_indexers.dense_index_one_dataset import index_snowflake_xs_collection
from func_timeout import func_timeout

beir_datasets = ["dbpedia-entity", "fever", "fiqa", "hotpotqa", "nfcorpus", "quora", "scifact"]
lengths = [200, 221, 6, 8, 8, 6, 9]
prefix_dataset = "irds:beir/"
directory = "bge"
datasets = [prefix_dataset + dataset for dataset in beir_datasets]
datasets.append("irds:msmarco-passage/trec-dl-2019")
lengths.append(7)


def index_snowflake_m_collections():
    for index, dataset_name in enumerate(datasets):
        try:
            index_snowflake_m_collection(dataset_name, lengths[index], directory)
            print(dataset_name + " DONE")
        except Exception as e:
            # Handles any other exceptions
            print(f"An error occurred: {e}")

def index_snowflake_xs_collections():
    for index, dataset_name in enumerate(datasets):
        try:
            index_snowflake_xs_collection(dataset_name, lengths[index], directory)
            print(dataset_name + " DONE")
        except Exception as e:
            # Handles any other exceptions
            print(f"An error occurred: {e}")


def main():
    try:
        func_timeout(24 * 3600 - 120, index_snowflake_xs_collections)
    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
