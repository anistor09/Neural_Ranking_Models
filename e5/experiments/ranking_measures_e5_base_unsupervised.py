from e5.experiments.ranking_measures_e5_base import run_metrics_e5

if __name__ == '__main__':
    """
    Executes ranking performance measurement for e5-base-unsupervised  by encoding queries and evaluating ranking performance.

    Args:
    model_name (str): The model identifier used for query encoding.
    """
    run_metrics_e5("e5-base-unsupervised")
