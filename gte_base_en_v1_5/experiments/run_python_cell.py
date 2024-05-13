from experiment_utils.experiments_helper import default_test_pipeline_name
from pyterrier.measures import RR, nDCG, MAP
from encoders.gte_base_en_encoder import GTEBaseDocumentEncoder
from memory_profiler import profile


def runtest():
    import pyterrier as pt

    if not pt.started():
        pt.init()

    eval_metrics = [RR @ 10, nDCG @ 10, MAP @ 100]

    q_encoder = GTEBaseDocumentEncoder("Alibaba-NLP/gte-base-en-v1.5")
    model_name = "gte-base-en-v1.5"
    dataset_name = "irds:beir/arguana"
    pipeline_name = "BM25 >> " + model_name
    path_to_root = "../../"

    default_test_pipeline_name(dataset_name, dataset_name, q_encoder, eval_metrics, model_name, pipeline_name,
                               path_to_root, timed=True)


@profile
def main():
    runtest()


if __name__ == '__main__':
    main()
