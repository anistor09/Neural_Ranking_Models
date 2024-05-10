import pyterrier as pt
from pyterrier.measures import RR, nDCG, MAP
from fast_forward import OnDiskIndex, Mode
from pathlib import Path


def load_sparse_index_from_disk(dataset_name, wmodel="BM25"):
    model_name = "gte-base-en-v1.5"

    index_path = "../" + "sparse_indexes/sparse_index_" + model_name + "_" + dataset_name

    # Load index to memory

    index = pt.IndexFactory.of(index_path, memory=True)

    bm25 = pt.BatchRetrieve(index, wmodel=wmodel)

    return bm25


def load_dense_index_from_disk(dataset_name, query_encoder, mode=Mode.MAXP):
    model_name = "gte-base-en-v1.5"

    index_path = "../" + "dense_indexes/ffindex_" + dataset_name + "_" + model_name + ".h5"

    # Retrieve from disk

    ff_index = OnDiskIndex.load(
        Path(index_path), query_encoder=query_encoder, mode=mode)
    # Return index loaded into memory

    return ff_index.to_memory()


def run_experiment(pipeline, dataset):
    test_set = pt.get_dataset(dataset)
    return pt.Experiment(
        [pipeline],
        test_set.get_topics(),
        test_set.get_qrels(),
        eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100],
    )

def run_experiments(pipelines, dataset):
    test_set = pt.get_dataset(dataset)
    return pt.Experiment(
        [pipelines],
        test_set.get_topics(),
        test_set.get_qrels(),
        eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100],
    )