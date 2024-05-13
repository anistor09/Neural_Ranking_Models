import pyterrier as pt
from fast_forward import OnDiskIndex, Mode
from pathlib import Path
from sklearn.model_selection import train_test_split
from fast_forward.util.pyterrier import FFScore
import time
from fast_forward.util.pyterrier import FFInterpolate
from general_dense_indexers.dense_index_one_dataset import get_dataset_name, format_name

SEED = 42


def load_sparse_index_from_disk(dataset_name, path_to_root, in_memory=True, wmodel="BM25"):
    index_path = path_to_root + "sparse_indexes/sparse_index_" + get_dataset_name(dataset_name)

    # Load index to memory

    index = pt.IndexFactory.of(index_path, memory=in_memory)

    bm25 = pt.BatchRetrieve(index, wmodel=wmodel)

    return bm25


def split_dev_test(qrels, test_size):
    dev_qrels, test_qrels = train_test_split(qrels, test_size=test_size, random_state=SEED)
    return dev_qrels, test_qrels


def load_dense_index_from_disk(dataset_name, query_encoder, model_name, mode=Mode.MAXP):
    index_path = "../" + "dense_indexes/ffindex_" + get_dataset_name(dataset_name) + "_" + format_name(
        model_name) + ".h5"

    # Retrieve from disk

    ff_index = OnDiskIndex.load(
        Path(index_path), query_encoder=query_encoder, mode=mode)
    # Return index loaded into memory
    return ff_index.to_memory()


def find_optimal_alpha_name(pipeline, ff_int, dev_set_name, alpha_vals=None):
    if alpha_vals is None:
        # alpha_vals = [0.025, 0.05, 0.1, 0.5, 0.9]
        alpha_vals = [0.01, 0.001, 0.005, 0.02]
    dev_set = pt.get_dataset(dev_set_name)
    find_optimal_alpha(pipeline, ff_int, dev_set.get_topics(),
                       dev_set.get_qrels(), alpha_vals)


def find_optimal_alpha(pipeline, ff_int, topics, qrels, alpha_vals=None):
    if alpha_vals is None:
        # alpha_vals = [0.025, 0.05, 0.1, 0.5, 0.9]
        alpha_vals = [0.01, 0.001, 0.005, 0.02]
    pt.GridSearch(
        pipeline,
        {ff_int: {"alpha": alpha_vals}},
        topics,
        qrels,
        "map",
        verbose=True,
    )


def run_single_experiment_name(pipeline, dataset_name, evaluation_metrics, name, timed=False):
    test_set = pt.get_dataset(dataset_name)
    return run_single_experiment(pipeline, test_set.get_topics(),
                                 test_set.get_qrels(), evaluation_metrics, name, timed)


def run_single_experiment(pipeline, topics, qrels, evaluation_metrics, name, timed=False):
    if timed:
        return time_fct(pt.Experiment, [pipeline],
                        topics,
                        qrels,
                        eval_metrics=evaluation_metrics,
                        names=[name])
    else:
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


def run_multiple_experiment(pipelines, topics, qrels, evaluation_metrics, names):
    return pt.Experiment(
        pipelines,
        topics,
        qrels,
        eval_metrics=evaluation_metrics,
        names=names
    )


def get_test_dev_sets(test_set_name, dev_set_name):
    test_set = pt.get_dataset(test_set_name)

    dev_topics = None
    dev_qrels = None
    if dev_set_name is not None:
        dev_set = pt.get_dataset(dev_set_name)
        dev_topics = dev_set.get_topics()
        dev_qrels = dev_set.get_qrels()
    return test_set.get_topics(), test_set.get_qrels(), dev_topics, dev_qrels


def default_test_pipeline_name(dataset_name, test_set_name, q_encoder, eval_metrics, model_name, pipeline_name,
                               path_to_root, dev_set_name=None, timed=False, alpha=0.05):
    test_topics, test_qrels, dev_topics, dev_qrels = get_test_dev_sets(test_set_name, dev_set_name)

    return default_test_pipeline(dataset_name, test_topics, test_qrels, q_encoder,
                                 eval_metrics, model_name, pipeline_name, path_to_root,
                                 dev_topics, dev_qrels, timed=timed, alpha=alpha)


def load_pipeline_dependencies(dataset_name, q_encoder, model_name, pipeline_name,
                               path_to_root, dev_topics=None, dev_qrels=None, alpha=0.05):
    # Spare index
    retriever = load_sparse_index_from_disk(dataset_name, path_to_root)

    # Dense index
    dense_index = load_dense_index_from_disk(dataset_name, q_encoder, model_name)

    ff_score = FFScore(dense_index)
    ff_int = FFInterpolate(alpha=alpha)

    # If devset is present run alpha optimization
    if dev_topics is not None:
        # Pipeline for finding optimal alpha
        pipeline_find_alpha = retriever % 100 >> ff_score >> ff_int
        find_optimal_alpha(pipeline_find_alpha, ff_int, dev_topics, dev_qrels)

    experiment_name = get_dataset_name(dataset_name) + ": " + pipeline_name
    default_pipeline = retriever % 1000 >> ff_score >> ff_int

    return default_pipeline, experiment_name


def default_test_pipeline(dataset_name, test_topics, test_qrels, q_encoder, eval_metrics, model_name, pipeline_name,
                          path_to_root, dev_topics=None, dev_qrels=None, timed=False, alpha=0.05):
    default_pipeline, experiment_name = load_pipeline_dependencies(dataset_name, q_encoder, model_name,
                                                                   pipeline_name,
                                                                   path_to_root, dev_topics=dev_topics,
                                                                   dev_qrels=dev_qrels, alpha=alpha)

    return run_single_experiment(default_pipeline, test_topics, test_qrels, eval_metrics, experiment_name, timed)


def test_first_stage_retrieval_name(dataset_name, test_set_name, eval_metrics, pipeline_name,
                                    path_to_root):
    test_set = pt.get_dataset(test_set_name)

    return test_first_stage_retrieval(dataset_name, test_set.get_qrels(), test_set.get_topics(), eval_metrics,
                                      pipeline_name,
                                      path_to_root)


def test_first_stage_retrieval(dataset_name, test_topics, test_qrels, eval_metrics, pipeline_name,
                               path_to_root):
    # Spare index
    retriever = load_sparse_index_from_disk(dataset_name, path_to_root)

    experiment_name = dataset_name + ": " + pipeline_name

    return run_single_experiment(retriever, test_topics, test_qrels, eval_metrics, experiment_name)


def time_fct(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Experiment took {elapsed_time:.3f} seconds to execute.")
    return result


def time_fct_print_results(func, *args, **kwargs):
    print(time_fct(func, *args, **kwargs))
