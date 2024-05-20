from sparse_index_one_dataset import index_one

datasets = ["nfcorpus", "cqadupstack/english", "arguana", "scidocs", "scifact", "fiqa"]
lengths = [8, 6, 47, 40, 9, 6]
prefix_dataset = "irds:beir/"


def index_multiple(prefix_dataset, datasets, lengths):
    for index, dataset_name in enumerate(datasets):
        try:
            index_one(prefix_dataset, dataset_name, lengths[index])
            print(dataset_name + " DONE")
        except Exception as e:

            print(f"An error occurred: {e}")


def main():
    try:
        index_multiple(prefix_dataset, datasets, lengths)

    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
