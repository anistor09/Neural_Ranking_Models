from experiment_utils.evaluator_helper import get_ranking_performance
from fast_forward.encoder import TCTColBERTQueryEncoder
from func_timeout import func_timeout


def main():
    model_name = "tct_colbert_msmarco"
    q_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")
    project_directory = "tct_colbert"
    func_timeout(5 * 3600 - 120, get_ranking_performance, args=(q_encoder, project_directory, model_name))


if __name__ == '__main__':
    main()
