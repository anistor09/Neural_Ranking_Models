from sparse_index_multiple_datasets import index_multiple

datasets = ["nfcorpus", "cqadupstack/english", "arguana", "scidocs", "scifact", "fiqa"]
lengths = [8, 6, 47, 40, 9, 6]


def main():
    model_name = "gte-base-en-v1.5"
    prefix_dataset = "irds:beir/"
    index_multiple(prefix_dataset, model_name, datasets, lengths)


if __name__ == '__main__':
    main()
