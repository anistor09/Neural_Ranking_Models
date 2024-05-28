from experiment_utils.evaluator_helper import get_ranking_performance, merge_dataset_names
from fast_forward.encoder import TCTColBERTQueryEncoder
from func_timeout import func_timeout

def get_datasets():
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

def main():
    model_name = "tct_colbert_msmarco"
    q_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")
    project_directory = "tct_colbert"
    func_timeout(3 * 3600 - 120, get_ranking_performance, args=(q_encoder, project_directory, model_name, get_datasets))


if __name__ == '__main__':
    main()
