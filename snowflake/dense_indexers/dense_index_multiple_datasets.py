from snowflake.dense_indexers.dense_index_one_dataset import index_snowflake_m_collection
from snowflake.dense_indexers.dense_index_one_dataset import index_snowflake_xs_collection
from func_timeout import func_timeout

beir_datasets = ["dbpedia-entity", "fever", "fiqa", "hotpotqa", "nfcorpus", "quora"]
lengths = [200, 221, 6, 8, 8, 6]
prefix_dataset = "irds:beir/"
directory = "snowflake"


def index_snowflake_m_collections():
    for index, dataset_name in enumerate(beir_datasets):
        index_snowflake_m_collection(prefix_dataset + dataset_name, lengths[index], directory)


def index_snowflake_xs_collections():
    datasets = [prefix_dataset + dataset for dataset in beir_datasets]
    datasets.append("irds:msmarco-passage/trec-dl-2019")
    lengths.append(7)

    for index, dataset_name in enumerate(beir_datasets):
        index_snowflake_xs_collection(prefix_dataset + dataset_name, lengths[index], directory)


def main():
    try:
        func_timeout(24 * 3600 - 120, index_snowflake_xs_collections)
    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
