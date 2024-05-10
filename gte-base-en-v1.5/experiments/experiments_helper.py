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


def find_optimal_alpha(pipeline, ff_int, dev_set, alpha_vals=[0.25, 0.05, 0.1, 0.5, 0.9]):
    pt.GridSearch(
        pipeline,
        {ff_int: {"alpha": alpha_vals}},
        dev_set.get_topics(),
        dev_set.get_qrels(),
        "map",
        verbose=True,
    )


def run_single_experiment_name(pipeline, dataset_name, evaluation_metrics, name):
    test_set = pt.get_dataset(dataset_name)
    run_single_experiment(pipeline, test_set, evaluation_metrics, name)


def run_single_experiment(pipeline, test_set, evaluation_metrics, name):
    return pt.Experiment(
        pipeline,
        test_set.get_topics(),
        test_set.get_qrels(),
        eval_metrics=evaluation_metrics,
        names=[name]
    )


def run_multiple_experiment_name(pipeline, dataset_name, evaluation_metrics, names):
    test_set = pt.get_dataset(dataset_name)
    run_multiple_experiment(pipeline, test_set, evaluation_metrics, names)


def run_multiple_experiment(pipeline, test_set, evaluation_metrics, names):
    return pt.Experiment(
        pipeline,
        test_set.get_topics(),
        test_set.get_qrels(),
        eval_metrics=evaluation_metrics,
        names=names
    )
