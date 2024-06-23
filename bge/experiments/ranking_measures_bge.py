from func_timeout import func_timeout
from experiment_utils.evaluator_helper import get_ranking_performance, merge_dataset_names
from encoders.bge_base_en import BgeQueryEncoder


def get_datasets():
    """
      Generates full dataset names with the appropriate suffixes for development and testing sets.

      Returns:
      list: A list of dataset names concatenated with their respective suffixes.
      """
    prefix = "irds:"

    # List of dataset base names
    dataset_names = [
        "beir/nfcorpus"
    ]

    n = len(dataset_names)
    devset_suffixes = ["/dev"] * n  # Default suffix for development datasets

    test_suffixes = ["/test"] * n  # Default suffix for test datasets

    # devset_suffixes[0] = "/train"
    # test_suffixes[n - 1] = "/trec-dl-2019"

    return merge_dataset_names(prefix, dataset_names, devset_suffixes, test_suffixes)


def run_metrics_bge(model_name="bge-base-en-v1.5"):
    """
     Runs ranking performance metrics for BGE base.

     Args:
     model_name (str): The name of the model used for encoding queries. Defaults to 'bge-base-en-v1.5'.
     """

    package = "BAAI/"
    q_encoder = BgeQueryEncoder(package + model_name)
    project_directory = "bge"

    func_timeout(9 * 3600 - 120, get_ranking_performance,
                 args=(q_encoder, project_directory, model_name, get_datasets))


if __name__ == '__main__':
    run_metrics_bge()
