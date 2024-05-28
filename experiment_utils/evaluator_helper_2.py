import pyterrier as pt
from encoders.bge_base_en import BgeQueryEncoder
from experiment_utils.experiments_helper import run_pipeline_multiple_datasets_metrics
import os


def get_ranking_performance(q_encoder, project_directory, model_name):
    if not pt.started():
        pt.init()

    dataset_names, dev_set_names, test_set_names = get_datasets()

    path_to_root = os.path.abspath(os.getcwd())

    res_metrics = run_pipeline_multiple_datasets_metrics(dataset_names,
                                                         test_set_names,
                                                         dev_set_names, q_encoder,
                                                         model_name,
                                                         path_to_root, project_directory)
    print(res_metrics)


def get_datasets():
    prefix = "irds:"

    # dataset_names = ["beir/scifact", "beir/nfcorpus", "beir/fiqa", "beir/dbpedia-entity", "beir/quora", "beir/hotpotqa",
    #                  "msmarco-passage"]

    dataset_names = ["beir/dbpedia-entity"]

    n = len(dataset_names)
    devset_sufixes = ["/dev"] * n

    test_suffixes = ["/test"] * n
    # devset_sufixes[0] = "/train"
    # test_suffixes[n - 1] = "/trec-dl-2019"

    test_set_names = [prefix + dataset for dataset in [a + b for a, b in zip(dataset_names, test_suffixes)]]
    dev_set_names = [prefix + dataset for dataset in [a + b for a, b in zip(dataset_names, devset_sufixes)]]
    dataset_names = [prefix + dataset for dataset in dataset_names]
    return dataset_names, dev_set_names, test_set_names


def main():
    # can add any encoder
    package = "BAAI/"
    model_name = "bge-base-en-v1.5"
    q_encoder = BgeQueryEncoder(package + model_name)
    project_directory = "bge"
    get_ranking_performance(q_encoder, project_directory, model_name)


if __name__ == '__main__':
    main()
