import pyterrier as pt
# from fast_forward import OnDiskIndex, Mode
from fast_forward import Mode
from pathlib import Path
from fast_forward.util.pyterrier import FFScore
import time
from fast_forward.util.pyterrier import FFInterpolate
from pyterrier.measures import RR, nDCG, MAP
import os
import pandas as pd
from fast_forward_indexes_library_enhancements.pipeline_transformers import FFInterpolateNormalized, EncodeUTF
from fast_forward_indexes_library_enhancements.disk import OnDiskIndex
from general_dense_indexers.dense_index_one_dataset import get_dataset_name, format_name
import re
import traceback

SEED = 42
eval_metrics = [RR @ 10, nDCG @ 10, MAP @ 100]
eval_metrics_general = [RR @ 10, nDCG @ 10, MAP @ 100]
eval_metrics_msmarco = [RR(rel=2) @ 10, nDCG @ 10, MAP(rel=2) @ 100]


def load_sparse_index_from_disk(dataset_name, path_to_root, in_memory=True, wmodel="BM25", index_path=None):
    if index_path is None:
        index_path = path_to_root + "/sparse_indexes/sparse_index_" + get_dataset_name(dataset_name)

    # Load index to memory if not specified otherwise

    index = pt.IndexFactory.of(index_path, memory=in_memory)

    bm25 = pt.BatchRetrieve(index, wmodel=wmodel)

    return bm25


def load_dense_index_from_disk(dataset_name, query_encoder, model_name, path_to_root, model_directory, mode=Mode.MAXP,
                               in_memory=True):
    index_path = path_to_root + "/" + model_directory + "/dense_indexes/ffindex_" + get_dataset_name(
        dataset_name) + "_" + format_name(
        model_name) + ".h5"

    # Retrieve from disk

    ff_index = OnDiskIndex.load(
        Path(index_path), query_encoder=query_encoder, mode=mode)
    # Return index loaded into memory
    if in_memory:
        return ff_index.to_memory()
    else:
        return ff_index


def find_optimal_alpha(pipeline_no_interpolation, topics, qrels, dataset_name, model_name, path_to_root,
                       model_directory, alpha_vals=None):
    file_path = path_to_root + "/" + model_directory + "/results/alpha_search_dev.csv"

    if alpha_vals is None:
        alpha_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    experiment_name = get_dataset_name(dataset_name) + ": " + "BM25 >> " + model_name
    maxi = 0
    max_alpha = 0.1

    print("START ALPHA TUNING FOR " + model_name + " on " + dataset_name)
    start_tun = time.time()

    for alpha in alpha_vals:
        print("START " + str(alpha))
        start = time.time()

        ff_int = FFInterpolateNormalized(alpha=alpha)
        pipeline = pipeline_no_interpolation >> ff_int
        result = run_single_experiment(pipeline, topics, qrels, eval_metrics, experiment_name)

        result['alpha'] = alpha
        result.to_csv(file_path, mode='a', header=not os.path.isfile(file_path), index=False)

        if result["nDCG@10"].iloc[0] > maxi:
            maxi = result["nDCG@10"].iloc[0]
            max_alpha = alpha

        end = time.time()
        lantecy = round(((end - start) / 60), 2)
        print("END " + str(alpha) + " in " + str(lantecy) + "mins")

    duplicate_dataframe = pd.read_csv(file_path)
    unique_dataframe = duplicate_dataframe.groupby('name').tail(9)
    unique_dataframe.to_csv(file_path, index=False)

    end_tun = time.time()
    lantecy = round(((end_tun - start_tun) / 60), 2)
    print("TUNING TOOK " + str(lantecy) + " minutes")

    return max_alpha


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

        if "msmarco-passage" in dev_set_name:
            dev_topics = dev_topics.sample(frac=0.3, random_state=SEED)

    return test_set.get_topics(), test_set.get_qrels(), dev_topics, dev_qrels


def default_test_pipeline_name(dataset_name, test_set_name, q_encoder, eval_metrics, model_name, pipeline_name,
                               path_to_root, model_directory, dev_set_name=None, timed=False, alpha=0.005,
                               in_memory_sparse=True,
                               in_memory_dense=True, index_path=None):
    test_topics, test_qrels, dev_topics, dev_qrels = get_test_dev_sets(test_set_name, dev_set_name)

    return default_test_pipeline(dataset_name, test_topics, test_qrels, q_encoder,
                                 eval_metrics, model_name, pipeline_name, path_to_root, model_directory,
                                 dev_topics, dev_qrels, timed=timed, alpha=alpha, in_memory_sparse=in_memory_sparse,
                                 in_memory_dense=in_memory_dense,
                                 index_path=index_path)


def load_pipeline_dependencies(dataset_name, q_encoder, model_name, pipeline_name,
                               path_to_root, model_directory, dev_topics=None, dev_qrels=None, alpha=0.005,
                               in_memory_sparse=True,
                               in_memory_dense=True, index_path=None):
    # Get sparse retriever and semantic reranker for pipeline creation

    sparse_retriever, semantic_reranker, optimal_alpha = get_pipeline_transformers(dataset_name, q_encoder, model_name,
                                                                                   path_to_root, model_directory,
                                                                                   dev_topics=dev_topics,
                                                                                   dev_qrels=dev_qrels, alpha=alpha,
                                                                                   in_memory_sparse=in_memory_sparse,
                                                                                   in_memory_dense=in_memory_dense,
                                                                                   index_path=index_path)

    experiment_name = get_dataset_name(dataset_name) + ": " + pipeline_name
    if 'dbpedia' in dataset_name or 'fever' in dataset_name:
        encode_utf = EncodeUTF()
        default_pipeline = sparse_retriever % 1000 >> encode_utf >> semantic_reranker
    else:
        default_pipeline = sparse_retriever % 1000 >> semantic_reranker

    return default_pipeline, experiment_name, optimal_alpha


def default_test_pipeline(dataset_name, test_topics, test_qrels, q_encoder, eval_metrics, model_name, pipeline_name,
                          path_to_root, model_directory, dev_topics=None, dev_qrels=None, timed=False, alpha=0.005,
                          in_memory_sparse=True,
                          in_memory_dense=True, index_path=None):
    default_pipeline, experiment_name, optimal_alpha = load_pipeline_dependencies(dataset_name, q_encoder, model_name,
                                                                                  pipeline_name,
                                                                                  path_to_root, model_directory,
                                                                                  dev_topics=dev_topics,
                                                                                  dev_qrels=dev_qrels, alpha=alpha,
                                                                                  in_memory_sparse=in_memory_sparse,
                                                                                  in_memory_dense=in_memory_dense,
                                                                                  index_path=index_path)
    result_metrics = run_single_experiment(default_pipeline, test_topics, test_qrels, eval_metrics, experiment_name,
                                           timed)
    if dev_topics is not None:
        return result_metrics, optimal_alpha
    else:
        return result_metrics


def test_first_stage_retrieval_name(dataset_name, test_set_name, eval_metrics, pipeline_name,
                                    path_to_root, timed=False, in_memory_sparse=True,
                                    index_path=None):
    test_set = pt.get_dataset(test_set_name)

    return test_first_stage_retrieval(dataset_name, test_set.get_topics(), test_set.get_qrels(), eval_metrics,
                                      pipeline_name,
                                      path_to_root, timed, in_memory_sparse=in_memory_sparse,
                                      index_path=index_path)


def test_first_stage_retrieval(dataset_name, test_topics, test_qrels, eval_metrics, pipeline_name,
                               path_to_root, timed=False, in_memory_sparse=True,
                               index_path=None):
    # Spare index
    retriever = load_sparse_index_from_disk(dataset_name, path_to_root, in_memory=in_memory_sparse,
                                            index_path=index_path)

    experiment_name = get_dataset_name(dataset_name) + ": " + pipeline_name

    return run_single_experiment(retriever, test_topics, test_qrels, eval_metrics, experiment_name, timed)


def get_timeit_dependencies_name(dataset_name, test_set_name, q_encoder, model_name,
                                 path_to_root, model_directory, dev_set_name=None, alpha=0.005, in_memory_sparse=True,
                                 in_memory_dense=True, index_path=None):
    test_topics, test_qrels, dev_topics, dev_qrels = get_test_dev_sets(test_set_name, dev_set_name)

    return get_timeit_dependencies(dataset_name, test_topics, q_encoder
                                   , model_name, path_to_root, model_directory,
                                   dev_topics, dev_qrels, alpha=alpha, in_memory_sparse=in_memory_sparse,
                                   in_memory_dense=in_memory_dense,
                                   index_path=index_path)


def get_pipeline_transformers(dataset_name, q_encoder, model_name,
                              path_to_root, model_directory, dev_topics=None, dev_qrels=None, alpha=0.005,
                              in_memory_sparse=True,
                              in_memory_dense=True, index_path=None):
    # Spare index
    retriever = load_sparse_index_from_disk(dataset_name, path_to_root, in_memory=in_memory_sparse,
                                            index_path=index_path)

    # Dense index
    dense_index = load_dense_index_from_disk(dataset_name, q_encoder, model_name, path_to_root, model_directory,
                                             in_memory=in_memory_dense)

    ff_score = FFScore(dense_index)
    optimal_alpha = alpha

    # If devset is present run alpha optimization
    if dev_topics is not None:
        # Pipeline for finding optimal alpha
        if 'dbpedia' in dataset_name or 'fever' in dataset_name:
            encode_utf = EncodeUTF()
            pipeline_find_alpha = retriever % 100 >> encode_utf >> ff_score
        else:
            pipeline_find_alpha = retriever % 100 >> ff_score

        optimal_alpha = find_optimal_alpha(pipeline_find_alpha, dev_topics, dev_qrels, dataset_name, model_name,
                                           path_to_root, model_directory)

    ff_int = FFInterpolateNormalized(alpha=optimal_alpha)
    semantic_ranker = ff_score >> ff_int

    if dev_topics is not None:
        return retriever, semantic_ranker, ff_int.alpha
    else:
        return retriever, semantic_ranker, None


def get_timeit_dependencies(dataset_name, test_topics, q_encoder, model_name,
                            path_to_root, model_directory, dev_topics=None, dev_qrels=None, alpha=0.005,
                            in_memory_sparse=True,
                            in_memory_dense=True, index_path=None):
    sparse_retriever, semantic_reranker, optimal_alpha = get_pipeline_transformers(dataset_name, q_encoder, model_name,
                                                                                   path_to_root, model_directory,
                                                                                   dev_topics=dev_topics,
                                                                                   dev_qrels=dev_qrels, alpha=alpha,
                                                                                   in_memory_sparse=in_memory_sparse,
                                                                                   in_memory_dense=in_memory_dense,
                                                                                   index_path=index_path)
    first_stage_results = sparse_retriever(test_topics)

    if dev_topics is not None:
        return first_stage_results, semantic_reranker, optimal_alpha
    else:
        return first_stage_results, semantic_reranker


def run_pipeline_multiple_datasets_metrics(dataset_names, test_set_names, dev_set_names, q_encoder, model_name,
                                           path_to_root, model_directory):
    pipeline_name = "BM25 >> " + model_name
    file_path = path_to_root + "/" + model_directory + "/results/ranking_metrics_alpha.csv"
    global eval_metrics

    for i in range(0, len(dataset_names)):
        try:
            if "msmarco-passage" in dataset_names[i]:
                eval_metrics = eval_metrics_msmarco
            else:
                eval_metrics = eval_metrics_general

            print(eval_metrics)
            result, optimal_alpha = default_test_pipeline_name(dataset_names[i], test_set_names[i], q_encoder,
                                                               eval_metrics,
                                                               model_name, pipeline_name,
                                                               path_to_root, model_directory,
                                                               dev_set_name=dev_set_names[i], timed=True)
            result['alpha'] = optimal_alpha
            result.to_csv(file_path, mode='a', header=not os.path.isfile(file_path), index=False)
            print(dataset_names[i] + " DONE")

        except Exception as e:
            # Handles any other exceptions
            print(dataset_names[i] + " FAILED")
            print(f"An error occurred: {e} for dataset {dataset_names[i]}")
            print(traceback.print_exc())

    duplicate_dataframe = pd.read_csv(file_path)

    print("BEFORE REMOVING DUPLICATES")
    print(duplicate_dataframe)

    unique_dataframe = duplicate_dataframe.groupby('name').tail(1)
    unique_dataframe.to_csv(file_path, index=False)

    return pd.read_csv(file_path)


def getOptimalAlpha(dataset_name, pipeline_name, model_directory):
    experiment_name = get_dataset_name(dataset_name) + ": " + pipeline_name
    path_to_root = os.path.abspath(os.getcwd())
    df = pd.read_csv(path_to_root + '/../../' + model_directory + '/results/ranking_metrics_alpha.csv')
    optimal_alpha = df[df['name'] == experiment_name]['alpha'].iloc[0]
    return optimal_alpha


def time_fct(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Experiment took {elapsed_time:.3f} seconds to execute.")
    return result


def latency_per_query(timeit_output, dataset_name, test_suffix, pipeline_name, model_directory):
    text_input = timeit_output.split(" s +- ")

    mean_time = float(text_input[0])
    standard_dev_time = float(text_input[1].split(" ms")[0])
    len_topics = len(pt.get_dataset(dataset_name + test_suffix).get_topics())
    mean_time_per_query = mean_time / len_topics

    # Transform in ms
    mean_time_per_query = mean_time_per_query * 1000
    mean_time_per_query = round(mean_time_per_query, 4)
    store_latency(dataset_name, pipeline_name, mean_time_per_query, mean_time, standard_dev_time, len_topics,
                  timeit_output, model_directory)
    return "Latency per query: " + str(mean_time_per_query) + " ms. " + "Experiment details: " + timeit_output


def store_latency(dataset_name, pipeline_name, mean_time_per_query, exp_mean_time, standard_dev_time, len_topics,
                  timeit_output, model_directory):
    path_to_root = os.path.abspath(os.getcwd())
    file_path = path_to_root + '/../../' + model_directory + '/results/latency_data.csv'

    data = {
        'dataset': [get_dataset_name(dataset_name)],
        'pipeline_name': [pipeline_name],
        'mean_time_per_query (ms)': [mean_time_per_query],
        'exp_time (s)': [exp_mean_time],
        'standard_dev_time (ms)': [standard_dev_time],
        'nr_queries': [len_topics],
        'timeit_details': [timeit_output]
    }
    new_data_df = pd.DataFrame(data)

    new_data_df.to_csv(file_path, mode='a', header=not os.path.isfile(file_path), index=False)


def time_fct_print_results(func, *args, **kwargs):
    print(time_fct(func, *args, **kwargs))
