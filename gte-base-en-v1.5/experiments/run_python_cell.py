from experiments_helper import split_dev_test, time_fct_print_results, default_complete_test_pipeline
from pyterrier.measures import RR, nDCG, MAP
from gte_base_en_encoder import GTEBaseDocumentEncoder
from memory_profiler import profile


def runtest():
    import pyterrier as pt

    if not pt.started():
        pt.init()

    eval_metrics = [RR @ 10, nDCG @ 10, MAP @ 100]

    q_encoder = GTEBaseDocumentEncoder("Alibaba-NLP/gte-base-en-v1.5")

    dataset_name = "arguana"
    dataset = pt.get_dataset("irds:beir/arguana")
    topics = dataset.get_topics()

    dev_topics, test_topics = split_dev_test(topics, test_size=0.8)

    time_fct_print_results(default_complete_test_pipeline, dataset_name, dataset.get_qrels(), test_topics, q_encoder,
                           eval_metrics)


@profile
def main():
    runtest()


if __name__ == '__main__':
    main()
