import pyterrier as pt
from pyterrier.measures import RR, nDCG, MAP
from fast_forward import OnDiskIndex, Mode
from pathlib import Path
from sklearn.model_selection import train_test_split
from fast_forward.util.pyterrier import FFScore

from fast_forward.util.pyterrier import FFInterpolate

SEED = 42


def load_sparse_index_from_disk(dataset_name, wmodel="BM25"):
    model_name = "gte-base-en-v1.5"

    index_path = "../" + "sparse_indexes/sparse_index_" + model_name + "_" + dataset_name

    # Load index to memory

    index = pt.IndexFactory.of(index_path, memory=True)

    bm25 = pt.BatchRetrieve(index, wmodel=wmodel)

    return bm25


def split_dev_test(qrels, test_size):
    dev_qrels, test_qrels = train_test_split(qrels, test_size=test_size, random_state=SEED)
    return dev_qrels, test_qrels


def format_filename(text):
    if '/' in text:
        return text.split('/')[0] + "_" + text.split('/')[1]
    else:
        return text


def load_dense_index_from_disk(dataset_name, query_encoder, mode=Mode.MAXP):
    model_name = "gte-base-en-v1.5"

    index_path = "../" + "dense_indexes/ffindex_" + format_filename(dataset_name) + "_" + model_name + ".h5"

    # Retrieve from disk

    ff_index = OnDiskIndex.load(
        Path(index_path), query_encoder=query_encoder, mode=mode)
    # Return index loaded into memory

    return ff_index.to_memory()


def find_optimal_alpha_name(pipeline, ff_int, dev_set_name, alpha_vals=[0.25, 0.05, 0.1, 0.5, 0.9]):
    dev_set = pt.get_dataset(dev_set_name)
    find_optimal_alpha(pipeline, ff_int, dev_set.get_topics(),
                       dev_set.get_qrels(), alpha_vals)


def find_optimal_alpha(pipeline, ff_int, topics, qrels, alpha_vals=[0.25, 0.05, 0.1, 0.5, 0.9]):
    pt.GridSearch(
        pipeline,
        {ff_int: {"alpha": alpha_vals}},
        topics,
        qrels,
        "map",
        verbose=True,
    )


def run_single_experiment_name(pipeline, dataset_name, evaluation_metrics, name):
    test_set = pt.get_dataset(dataset_name)
    return run_single_experiment(pipeline, test_set.get_topics(),
                                 test_set.get_qrels(), evaluation_metrics, name)


def run_single_experiment(pipeline, topics, qrels, evaluation_metrics, name):
    return pt.Experiment(
        [pipeline],
        topics,
        qrels,
        eval_metrics=evaluation_metrics,
        names=[name]
    )


def run_multiple_experiment_name(pipeline, dataset_name, evaluation_metrics, names):
    test_set = pt.get_dataset(dataset_name)
    return run_multiple_experiment(pipeline, test_set.get_topics(),
                                   test_set.get_qrels(), evaluation_metrics, names)


def run_multiple_experiment(pipeline, topics, qrels, evaluation_metrics, names):
    return pt.Experiment(
        pipeline,
        topics,
        qrels,
        eval_metrics=evaluation_metrics,
        names=names
    )


def default_complete_test_pipeline_name(dataset_name, dev_set_name, test_set_name, q_encoder, eval_metrics):
    # Spare index
    retriever = load_sparse_index_from_disk(dataset_name)

    # Dense index
    dense_index = load_dense_index_from_disk(dataset_name, q_encoder)

    ff_score = FFScore(dense_index)
    ff_int = FFInterpolate(alpha=0.05)

    # Pipeline for finding optimal alpha
    pipeline_find_alpha = retriever % 100 >> ff_score >> ff_int
    find_optimal_alpha_name(pipeline_find_alpha, ff_int, dev_set_name)

    experiment_name = dataset_name + ": BM25 >> gte-base-en-v1.5"
    default_pipeline = retriever % 1000 >> ff_score >> ff_int
    return run_single_experiment_name(default_pipeline, test_set_name, eval_metrics, experiment_name)


def default_complete_test_pipeline(dataset_name, qrels, dev_topics, test_topics, q_encoder, eval_metrics):
    # Spare index
    retriever = load_sparse_index_from_disk(dataset_name)

    # Dense index
    dense_index = load_dense_index_from_disk(dataset_name, q_encoder)

    ff_score = FFScore(dense_index)
    ff_int = FFInterpolate(alpha=0.05)

    # Pipeline for finding optimal alpha
    pipeline_find_alpha = retriever % 100 >> ff_score >> ff_int
    find_optimal_alpha(pipeline_find_alpha, ff_int, dev_topics, qrels)

    experiment_name = dataset_name + ": BM25 >> gte-base-en-v1.5"
    default_pipeline = retriever % 1000 >> ff_score >> ff_int
    return run_single_experiment(default_pipeline, test_topics, qrels, eval_metrics, experiment_name)


def default_complete_test_pipeline_nogrid(dataset_name, qrels, dev_topics, test_topics, q_encoder, eval_metrics):
    # Spare index
    retriever = load_sparse_index_from_disk(dataset_name)

    # Dense index
    dense_index = load_dense_index_from_disk(dataset_name, q_encoder)

    ff_score = FFScore(dense_index)
    ff_int = FFInterpolate(alpha=0.05)

    experiment_name = dataset_name + ": BM25 >> gte-base-en-v1.5"
    default_pipeline = retriever % 1000 >> ff_score >> ff_int
    return run_single_experiment(default_pipeline, test_topics, qrels, eval_metrics, experiment_name)
