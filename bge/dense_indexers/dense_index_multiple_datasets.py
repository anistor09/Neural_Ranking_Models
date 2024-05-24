from bge.dense_indexers.dense_index_one_dataset import index_bge_base_collection

datasets = ["quora", "fever"]
lengths = [6, 221]
prefix_dataset = "irds:beir/"
directory = "bge"


def index_bge_base_collections():
    for index, dataset_name in enumerate(datasets):
        index_bge_base_collection(prefix_dataset + dataset_name, lengths[index], directory)


def main():
    try:
        index_bge_base_collections()
    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
