from func_timeout import func_timeout
from experiment_utils.evaluator_helper import get_ranking_performance, merge_dataset_names
from encoders.bge_base_en import BgeQueryEncoder


def get_datasets():
    """
    Generates a list of dataset identifiers with prefixes and specific suffixes for different dataset sections.

    Returns:
    list: A list of full dataset identifiers with prefixed sources and respective development or test section suffixes.
    """

    prefix = "irds:"

    dataset_names = [
        "beir/dbpedia-entity", "beir/fever", "msmarco-passage"
    ]

    n = len(dataset_names)
    devset_suffixes = ["/dev"] * n

    test_suffixes = ["/test"] * n
    # devset_suffixes[0] = "/train"
    test_suffixes[n - 1] = "/trec-dl-2019"

    return merge_dataset_names(prefix, dataset_names, devset_suffixes, test_suffixes)


def main():
    """
    Main function to run ranking performance metrics for the BGE small model.

    Encodes queries using BGE small model and evaluates the ranking performance across multiple datasets.
    """

    package = "BAAI/"
    model_name = "bge-small-en-v1.5"
    q_encoder = BgeQueryEncoder(package + model_name)
    project_directory = "bge"

    func_timeout(7 * 3600 - 120, get_ranking_performance, args=(q_encoder, project_directory, model_name, get_datasets))


if __name__ == '__main__':
    main()
