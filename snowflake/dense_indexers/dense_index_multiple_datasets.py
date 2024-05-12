from snowflake.dense_indexers.dense_index_one_dataset import index_snowflake_m_collection

datasets = ["nfcorpus", "cqadupstack/english", "arguana", "scidocs", "scifact", "fiqa"]
lengths = [8, 6, 47, 40, 9, 6]
prefix_dataset = "irds:beir/"
directory = "snowflake"


def index_snowflake_m_collections():
    for index, dataset_name in enumerate(datasets):
        index_snowflake_m_collection(prefix_dataset + dataset_name, lengths[index], directory)


def main():
    index_snowflake_m_collections()


if __name__ == '__main__':
    main()
