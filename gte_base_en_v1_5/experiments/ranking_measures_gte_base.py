from experiment_utils.evaluator_helper import get_ranking_performance, merge_dataset_names
from encoders.gte_base_en_encoder import GTEBaseEncoder
from func_timeout import func_timeout


def get_datasets():
    """
        Constructs a list of full dataset paths by merging a prefix with dataset names and respective suffixes.

        Returns:
        list: A list of complete dataset paths with the appropriate suffixes for development and testing phases.
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
    Executes ranking performance measurement for gte_base_en_v1_5  by encoding queries and evaluating ranking performance.

    Args:
    model_name (str): The model identifier used for query encoding.
    """
    package = "Alibaba-NLP/"
    model_name = "gte-base-en-v1.5"
    q_encoder = GTEBaseEncoder(package + model_name)
    project_directory = "gte_base_en_v1_5"
    func_timeout(9 * 3600 - 120, get_ranking_performance, args=(q_encoder, project_directory, model_name, get_datasets))


if __name__ == '__main__':
    main()
