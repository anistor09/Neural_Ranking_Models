from experiment_utils.evaluator_helper import get_ranking_performance
from encoders.gte_base_en_encoder import GTEBaseEncoder
from func_timeout import func_timeout


def main():
    package = "Alibaba-NLP/"
    model_name = "gte-base-en-v1.5"
    q_encoder = GTEBaseEncoder(package + model_name)
    project_directory = "gte_base_en_v1_5"
    func_timeout(5 * 3600 - 120, get_ranking_performance, args=(q_encoder, project_directory, model_name))


if __name__ == '__main__':
    main()
