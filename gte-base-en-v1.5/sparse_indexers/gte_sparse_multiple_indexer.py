from ...general_sparse_INDEXERS.sparse_index_multiple_datasets import index_multiple

datasets = ["nfcorpus", "cqadupstack/english", "scifact"]


def main():
    model_name = "gte-base-en-v1.5"
    prefix_dataset = "irds:beir/"
    index_multiple(prefix_dataset, model_name, datasets)


if __name__ == '__main__':
    main()
