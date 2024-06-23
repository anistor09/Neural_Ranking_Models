from func_timeout import func_timeout
from experiment_utils.evaluator_helper import get_ranking_performance, merge_dataset_names
from encoders.e5 import E5QueryEncoder


def get_datasets():
    """
        Constructs a list of full dataset paths by merging a prefix with dataset names and respective suffixes.

        Returns:
        list: A list of complete dataset paths for both development and test phases, with custom modifications for
              specific datasets.

        """
    prefix = "irds:"

    dataset_names = ["beir/scifact", "beir/nfcorpus", "beir/fiqa", "beir/quora", "beir/dbpedia-entity", "beir/hotpotqa",
                     "beir/fever", "msmarco-passage"]

    n = len(dataset_names)
    devset_suffixes = ["/dev"] * n

    test_suffixes = ["/test"] * n
    devset_suffixes[0] = "/train"
    test_suffixes[n - 1] = "/trec-dl-2019"

    return merge_dataset_names(prefix, dataset_names, devset_suffixes, test_suffixes)


def run_metrics_e5(model_name="e5-base-v2"):
    """
    Executes ranking performance measurement for e5-base-v2 by encoding queries and evaluating ranking performance.

    Args:
    model_name (str): The model identifier used for query encoding.
    """
    package = "intfloat/"
    q_encoder = E5QueryEncoder(package + model_name)
    project_directory = "e5"

    func_timeout(9 * 3600 - 120, get_ranking_performance,
                 args=(q_encoder, project_directory, model_name, get_datasets))


if __name__ == '__main__':
    run_metrics_e5()
