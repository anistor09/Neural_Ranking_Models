from tct_colbert.dense_indexers.dense_index_one_dataset import index_tct_colbert

# Define the list of datasets and their corresponding ID lengths
datasets = ["quora", "fever"]
lengths = [6, 221]  # Specific maximum ID lengths for each dataset
prefix_dataset = "irds:beir/"  # Prefix to form full dataset identifiers
directory = "tct_colbert"  # Directory to store the indexed files


def index_tct_colbert_collections():
    """
      Indexes multiple datasets using the TCT-Colbert indexer from the Castorini collection. Iterates through each dataset,
      combining the provided prefix with dataset names, and applies indexing with specified ID lengths.
      """
    for index, dataset_name in enumerate(datasets):
        index_tct_colbert(prefix_dataset + dataset_name, lengths[index], directory)


def main():
    """
       Main function to initiate the indexing process for the defined datasets using the TCT-Colbert framework.
    """
    index_tct_colbert_collections()


if __name__ == '__main__':
    main()
