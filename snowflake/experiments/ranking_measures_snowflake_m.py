from experiment_utils.evaluator_helper import get_ranking_performance, merge_dataset_names
from encoders.snowflake_arctic_embed_m import SnowFlakeQueryEncoder
from func_timeout import func_timeout


def get_datasets():
    """
        Constructs a list of full dataset paths by merging a prefix with dataset names and respective suffixes.

        Returns:
        list: A list of complete dataset paths for both development and test phases, with custom modifications for
              specific datasets.

    """
    prefix = "irds:"

    dataset_names = ["beir/dbpedia-entity",
                     "beir/fever",
                     "beir/hotpotqa",
                     "msmarco-passage"]

    n = len(dataset_names)
    devset_suffixes = ["/dev"] * n

    test_suffixes = ["/test"] * n
    # devset_suffixes[0] = "/train"
    test_suffixes[n - 1] = "/trec-dl-2019"

    return merge_dataset_names(prefix, dataset_names, devset_suffixes, test_suffixes)


def main():
    """
        Main function to execute ranking performance evaluation for the Snowflake medium model. It sets a timeout for
        the entire ranking process for easier debugging on the SuperComputer.
    """
    package = "Snowflake/"
    model_name = "snowflake-arctic-embed-m"
    q_encoder = SnowFlakeQueryEncoder(package + model_name)
    project_directory = "snowflake"
    func_timeout(11 * 3600 - 120, get_ranking_performance,
                 args=(q_encoder, project_directory, model_name, get_datasets))


if __name__ == '__main__':
    main()
