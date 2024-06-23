import pyterrier as pt
from encoders.bge_base_en import BgeQueryEncoder
from experiment_utils.experiments_helper import run_pipeline_multiple_datasets_metrics
import os


def merge_dataset_names(prefix, dataset_names, devset_suffixes, test_suffixes):
    """
        Combines dataset names with prefixes and respective suffixes for development and test sets.
    """
    test_set_names = [prefix + dataset for dataset in [a + b for a, b in zip(dataset_names, test_suffixes)]]
    dev_set_names = [prefix + dataset for dataset in [a + b for a, b in zip(dataset_names, devset_suffixes)]]
    dataset_names = [prefix + dataset for dataset in dataset_names]
    return dataset_names, dev_set_names, test_set_names


def get_datasets():
    """
          Generates full dataset names with the appropriate suffixes for development and testing sets.
    """
    prefix = "irds:"

    # dataset_names = ["beir/scifact", "beir/nfcorpus", "beir/fiqa", "beir/dbpedia-entity", "beir/quora", "beir/hotpotqa",
    #                  "beir/fever",
    #                  "msmarco-passage"]

    dataset_names = [
        "beir/fever"
    ]

    n = len(dataset_names)
    devset_suffixes = ["/dev"] * n

    test_suffixes = ["/test"] * n
    # devset_suffixes[0] = "/train"
    # test_suffixes[n - 1] = "/trec-dl-2019"

    return merge_dataset_names(prefix, dataset_names, devset_suffixes, test_suffixes)


def get_ranking_performance(q_encoder, project_directory, model_name, get_datasets_func=get_datasets):
    """
        Initializes PyTerrier and runs ranking performance metrics across multiple datasets.
    """
    if not pt.started():
        pt.init()

    dataset_names, dev_set_names, test_set_names = get_datasets_func()

    path_to_root = os.path.abspath(os.getcwd())

    res_metrics = run_pipeline_multiple_datasets_metrics(dataset_names,
                                                         test_set_names,
                                                         dev_set_names, q_encoder,
                                                         model_name,
                                                         path_to_root, project_directory)
    print(res_metrics)


def main():
    """
        Main function to initialize the query encoder and execute the ranking performance evaluation.
    """
    package = "BAAI/"
    model_name = "bge-base-en-v1.5"
    q_encoder = BgeQueryEncoder(package + model_name)
    project_directory = "bge"
    get_ranking_performance(q_encoder, project_directory, model_name)


if __name__ == '__main__':
    main()
