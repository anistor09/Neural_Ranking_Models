from func_timeout import func_timeout
from experiment_utils.evaluator_helper import get_ranking_performance_bge
from encoders.bge_base_en import BgeQueryEncoder


def main():
    package = "BAAI/"
    model_name = "bge-base-en-v1.5"
    q_encoder = BgeQueryEncoder(package + model_name)
    project_directory = "bge"

    func_timeout(5 * 3600 - 120, get_ranking_performance_bge, args=(q_encoder, project_directory, model_name))


if __name__ == '__main__':
    main()
