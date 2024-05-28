from func_timeout import func_timeout
from experiment_utils.evaluator_helper import get_ranking_performance, merge_dataset_names
from encoders.bge_base_en import BgeQueryEncoder


def get_datasets():
    prefix = "irds:"
    dataset_names = ["beir/scifact", "beir/nfcorpus", "beir/fiqa", "beir/dbpedia-entity", "beir/quora", "beir/hotpotqa",
                     "beir/fever",
                     "msmarco-passage"]
    n = len(dataset_names)
    devset_suffixes = ["/dev"] * n

    test_suffixes = ["/test"] * n
    devset_suffixes[0] = "/train"
    test_suffixes[n - 1] = "/trec-dl-2019"

    return merge_dataset_names(prefix, dataset_names, devset_suffixes, test_suffixes)


def main():
    package = "BAAI/"
    model_name = "bge-small-en-v1.5"
    q_encoder = BgeQueryEncoder(package + model_name)
    project_directory = "bge"

    func_timeout(13 * 3600 - 120, get_ranking_performance, args=(q_encoder, project_directory, model_name, get_datasets))


if __name__ == '__main__':
    main()
