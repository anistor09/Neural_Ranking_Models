import os


def get_file_path(dir):
    return os.path.abspath(os.getcwd()) + '/../../../.ir_datasets/msmarco-passage/' + dir + '/qrels'


def get_file_path_beir(dataset):
    return os.path.abspath(os.getcwd()) + '/../../../.ir_datasets/' + dataset + '.qrels'


def get_relevant_queries_per_doc(file_path, separator, min_rellevance, description):
    """
     Calculate and print the average number of relevant documents per query from a given  (dataset).

     Args:
     file_path (str): Path to the dataset file.
     separator (str): Character used to split data lines into parts.
     min_rellevance (int): Minimum relevance score to consider a document relevant.
     description (str): Description of the dataset being processed.
     """

    with open(file_path, 'r') as file:
        # Skip the header if there is one
        query_rel_docs = {}
        next(file)

        for line in file:

            parts = line.strip().split(separator)

            # Check if the line has the correct number of parts
            if len(parts) >= 4:
                query_id, query_type, doc_id, relevance = parts
                relevance = int(relevance)  # Convert relevance to integer

                # Store or increment the count if relevance meets the threshold
                if relevance >= min_rellevance:
                    if query_id not in query_rel_docs:
                        query_rel_docs[query_id] = 1
                    else:
                        query_rel_docs[query_id] += 1

            elif len(parts) == 3:
                query_id, doc_id, relevance = parts
                relevance = int(relevance)  # Convert relevance to integer

                if relevance >= min_rellevance:

                    if query_id not in query_rel_docs:
                        query_rel_docs[query_id] = 1
                    else:
                        query_rel_docs[query_id] += 1
            else:
                print(parts)  # Output the incomplete parts for debugging

        x = 0.0
        for query_id in query_rel_docs.keys():
            x += query_rel_docs[query_id]

        x /= len(query_rel_docs.keys())

        x = round(x, 2)

        print(f"Avearage number relevant docs per query: {x} for {description}")


if __name__ == '__main__':

    # Process MSMARCO dev and TREC evaluation datasets
    dev_qrels_location = get_file_path("dev")

    get_relevant_queries_per_doc(dev_qrels_location, "\t", 1, "msmarco/dev")

    trec_qrels_location = get_file_path("trec-dl-2019")
    get_relevant_queries_per_doc(trec_qrels_location, " ", 2, "trec-dl-2019")

    # Process BEIR datasets, including beir/msmarco
    datasets = ['msmarco', 'quora', 'fever']

    for dataset in datasets:
        dataset_complete = "beir/" + dataset + "/dev"
        beir_test = get_file_path_beir(dataset_complete)
        get_relevant_queries_per_doc(beir_test, "\t", 1, dataset_complete)

        dataset_complete = "beir/" + dataset + "/test"
        beir_test = get_file_path_beir(dataset_complete)
        get_relevant_queries_per_doc(beir_test, "\t", 1, dataset_complete)
