from tct_colbert.dense_indexers.dense_index_one_dataset import index_castorini_collection

datasets = ["quora", "fever"]
lengths = [6, 221]
prefix_dataset = "irds:beir/"
directory = "tct_colbert"


def index_bge_base_collections():
    for index, dataset_name in enumerate(datasets):
        index_castorini_collection(prefix_dataset + dataset_name, lengths[index], directory)


def main():
    index_bge_base_collections()


if __name__ == '__main__':
    main()
