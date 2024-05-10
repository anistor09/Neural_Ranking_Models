from index_one_dataset import index_collection

datasets = ["nfcorpus", "cqadupstack/english", "scifact"]


def main():
    for dataset_name in datasets:
        try:
            index_collection(dataset_name)
            print(dataset_name + " DONE")
        except Exception as e:

            print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
