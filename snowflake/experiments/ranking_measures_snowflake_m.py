from experiment_utils.evaluator_helper import get_ranking_performance_bge
from encoders.snowflake_arctic_embed_m import SnowFlakeQueryEncoder
from func_timeout import func_timeout


def main():
    package = "Snowflake/"
    model_name = "snowflake-arctic-embed-m"
    q_encoder = SnowFlakeQueryEncoder(package + model_name)
    project_directory = "snowflake"
    func_timeout(5 * 3600 - 120, get_ranking_performance_bge, args=(q_encoder, project_directory, model_name))


if __name__ == '__main__':
    main()
