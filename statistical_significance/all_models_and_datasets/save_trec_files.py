from func_timeout import func_timeout
from experiment_utils.evaluator_helper import get_ranking_performance, merge_dataset_names
from encoders.bge_base_en import BgeQueryEncoder
from encoders.e5 import E5QueryEncoder
from encoders.gte_base_en_encoder import GTEBaseEncoder
from encoders.nomic import NomicQueryEncoder
from encoders.snowflake_arctic_embed_m import SnowFlakeQueryEncoder
from fast_forward.encoder import TCTColBERTQueryEncoder


def get_datasets():
    """Combines dataset names with suffixes to generate dataset identifiers for testing and development."""
    prefix = "irds:"

    dataset_names = [ "beir/hotpotqa",
                     "beir/fever", "beir/quora", "beir/dbpedia-entity"]

    # dataset_names = ["msmarco-passage"]

    n = len(dataset_names)
    devset_suffixes = ["/dev"] * n

    test_suffixes = ["/test"] * n
    # devset_suffixes[0] = "/train"
    # test_suffixes[0] = "/trec-dl-2019"

    return merge_dataset_names(prefix, dataset_names, devset_suffixes, test_suffixes)


def run_metrics_bge():
    """Runs ranking performance metrics for BGE model variants."""
    model_names = ["bge-base-en-v1.5", "bge-small-en-v1.5"]
    for model_name in model_names:
        try:
            package = "BAAI/"
            q_encoder = BgeQueryEncoder(package + model_name)
            project_directory = "bge"

            func_timeout(9 * 3600 - 120, get_ranking_performance,
                         args=(q_encoder, project_directory, model_name, get_datasets))

        except Exception as e:
            # Handles any other exceptions
            print(f"An error occurred: {e}")


def run_metrics_e5():
    """Runs ranking performance metrics for E5 model variants."""
    model_names = ["e5-small-v2", "e5-base-v2", "e5-base-unsupervised"]
    for model_name in model_names:
        try:
            package = "intfloat/"
            q_encoder = E5QueryEncoder(package + model_name)
            project_directory = "e5"

            func_timeout(9 * 3600 - 120, get_ranking_performance,
                         args=(q_encoder, project_directory, model_name, get_datasets))

        except Exception as e:
            # Handles any other exceptions
            print(f"An error occurred: {e}")


def run_metrics_gte():
    """Runs ranking performance metrics for the GTE Base model."""
    try:
        package = "Alibaba-NLP/"
        model_name = "gte-base-en-v1.5"
        q_encoder = GTEBaseEncoder(package + model_name)
        project_directory = "gte_base_en_v1_5"
        func_timeout(9 * 3600 - 120, get_ranking_performance,
                     args=(q_encoder, project_directory, model_name, get_datasets))

    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


def run_metrics_nomic(model_name="nomic-embed-text-v1"):
    """Runs ranking performance metrics for Nomic model."""
    try:
        package = "nomic-ai/"
        q_encoder = NomicQueryEncoder(package + model_name)
        project_directory = "nomic"

        func_timeout(11 * 3600 - 120, get_ranking_performance,
                     args=(q_encoder, project_directory, model_name, get_datasets))

    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


def run_metrics_snowflake():
    """Runs ranking performance metrics for Snowflake model variants."""
    model_names = ["snowflake-arctic-embed-xs", "snowflake-arctic-embed-m"]
    for model_name in model_names:
        try:
            package = "Snowflake/"
            q_encoder = SnowFlakeQueryEncoder(package + model_name)
            project_directory = "snowflake"
            func_timeout(11 * 3600 - 120, get_ranking_performance,
                         args=(q_encoder, project_directory, model_name, get_datasets))

        except Exception as e:
            # Handles any other exceptions
            print(f"An error occurred: {e}")


def run_metrics_tct_colbert():
    """Runs ranking performance metrics for the TCT-Colbert model."""
    try:
        model_name = "tct_colbert_msmarco"
        q_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")
        project_directory = "tct_colbert"
        func_timeout(9 * 3600 - 120, get_ranking_performance,
                     args=(q_encoder, project_directory, model_name, get_datasets))
    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    """
        The purpose of this function is to save all the TREC Files or run all experiments again. If the formers is
        desired the save_trec_files_only flag should be set to true within experiment_utils/experiments_helper.py.
    """

    run_metrics_tct_colbert()
    print("TCT Colbert done")

    run_metrics_gte()
    print("GTE DONE")

    run_metrics_snowflake()
    print("Snowflake DONE")

    run_metrics_bge()
    print("BGE DONE")

    run_metrics_e5()
    print("E5 DONE")

    run_metrics_nomic()
    print("Nomic DONE")
