from func_timeout import func_timeout
from experiment_utils.evaluator_helper import get_ranking_performance, merge_dataset_names
from encoders.nomic import NomicQueryEncoder


def get_datasets():
    prefix = "irds:"

    dataset_names = ["beir/scifact", "beir/nfcorpus", "beir/fiqa", "beir/quora", "beir/dbpedia-entity", "beir/hotpotqa",
                     "beir/fever", "msmarco-passage"]

    n = len(dataset_names)
    devset_suffixes = ["/dev"] * n

    test_suffixes = ["/test"] * n
    devset_suffixes[0] = "/train"
    test_suffixes[n - 1] = "/trec-dl-2019"

    return merge_dataset_names(prefix, dataset_names, devset_suffixes, test_suffixes)


def run_metrics_nomic(model_name="nomic-embed-text-v1"):
    package = "nomic-ai/"
    q_encoder = NomicQueryEncoder(package + model_name)
    project_directory = "nomic"

    func_timeout(9 * 3600 - 120, get_ranking_performance,
                 args=(q_encoder, project_directory, model_name, get_datasets))


if __name__ == '__main__':
    run_metrics_nomic()
