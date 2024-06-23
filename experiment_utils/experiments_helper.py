import pyterrier as pt
from fast_forward_indexes_library_enhancements import Mode
from pathlib import Path
from fast_forward.util.pyterrier import FFScore
import time
from pyterrier.measures import RR, nDCG, MAP
import os
import pandas as pd
from fast_forward_indexes_library_enhancements.pipeline_transformers import FFInterpolateNormalized, EncodeUTF
from fast_forward_indexes_library_enhancements.disk import OnDiskIndex
from general_dense_indexers.dense_index_one_dataset import get_dataset_name, format_name
import traceback

# Constant for reproducibility in randomized processes, such as data sampling
SEED = 42

# Evaluation metrics used for testing the Fast Forward indexes pipeline within Pyterrier.
# These metrics measure ranking performance at various cutoffs to evaluate the effectiveness of retrieval methods.

eval_metrics = [RR @ 10, nDCG @ 10, MAP @ 100]
eval_metrics_general = [RR @ 10, nDCG @ 10, MAP @ 100]
eval_metrics_msmarco = [RR(rel=2) @ 10, nDCG @ 10, MAP(rel=2) @ 100]

# This flag controls whether to only save the TREC-formatted result files without executing further analyses like
# ranking metrics computation, results saving, or alpha tuning on development sets. When True, the optimal alpha value
# for each model is assumed to be pre-determined based on prior experiments.
save_trec_files_only = False


def load_sparse_index_from_disk(dataset_name, path_to_root, in_memory=True, wmodel="BM25", index_path=None):
    """
       Loads a sparse index from disk for the specified dataset using PyTerrier. If no index path is provided,
       it constructs one based on the dataset name.
    """
    if index_path is None:
        index_path = path_to_root + "/sparse_indexes/sparse_index_" + get_dataset_name(dataset_name)

    # Load index to memory if not specified otherwise

    index = pt.IndexFactory.of(index_path, memory=in_memory)

    bm25 = pt.BatchRetrieve(index, wmodel=wmodel)

    return bm25


def load_dense_index_from_disk(dataset_name, query_encoder, model_name, path_to_root, model_directory, mode=Mode.MAXP,
                               in_memory=True):
    """
        Loads a dense index from disk. The dense index is loaded into memory if the on_memory flag is set to True. This
        might result to OOM exceptions if the code runs on a 16 G RAM machine for all datasets besides Fiqa, NFCorpus and
        SciFact.
    """

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
    """
        Tunes the interpolation parameter alpha for combining sparse and dense retrieval results, aiming to optimize
        the retrieval performance based on nDCG. The results for each alpha values are saved to support the final
        parameter choices.
    """
    if save_trec_files_only:
        return getOptimalAlpha(dataset_name, "BM25 >> " + model_name, model_directory)

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

    # Clean duplicate records and save unique entries

    duplicate_dataframe = pd.read_csv(file_path)
    unique_dataframe = duplicate_dataframe.groupby('name').tail(9)
    unique_dataframe.to_csv(file_path, index=False)

    end_tun = time.time()
    lantecy = round(((end_tun - start_tun) / 60), 2)
    print("TUNING TOOK " + str(lantecy) + " minutes")

    return max_alpha


def run_single_experiment_name(pipeline, dataset_name, evaluation_metrics, name, timed=False):
    """
        Fetches the desired test set and calls run_single_experiment for running an experiment in Pyterrier.

    """
    test_set = pt.get_dataset(dataset_name)
    return run_single_experiment(pipeline, test_set.get_topics(),
                                 test_set.get_qrels(), evaluation_metrics, name, timed)


def get_model_name(exp_name):
    """
        Gets model name from experiment name. E.g., bge-base-en-v1.5 from -> nfcorpus: BM25 >> bge-base-en-v1.5

    """
    tokens = exp_name.split(":")
    model = tokens[1].split(" ")[3]

    return model


def run_single_experiment(pipeline, topics, qrels, evaluation_metrics, name, timed=False):
    """
        Runs an experiment in Pyterrier.

    """
    if save_trec_files_only:
        result = pipeline(topics)
        final_significance_testing = pt.model.add_ranks(result)

        dataset = name.split(":")[0]
        dir_path = os.path.abspath(os.getcwd()) + '/results/trec_runs/' + dataset
        os.makedirs(dir_path, exist_ok=True)

        pt.io.write_results(final_significance_testing, dir_path + "/" + get_model_name(name) + ".trec")
        return None
    else:
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
    """
        Fetches the desired test set and calls run_single_experiment for running an experiment in Pyterrier by
        evalauting multiple models within same experiment.

    """
    test_set = pt.get_dataset(dataset_name)
    return run_multiple_experiment(pipeline, test_set.get_topics(),
                                   test_set.get_qrels(), evaluation_metrics, names)


def run_multiple_experiment(pipelines, topics, qrels, evaluation_metrics, names):
    """
        Runs an experiment in Pyterrier by evalauting multiple models within same experiment.
    """
    return pt.Experiment(
        pipelines,
        topics,
        qrels,
        eval_metrics=evaluation_metrics,
        names=names
    )


def get_test_dev_sets(test_set_name, dev_set_name):
    """
        Fetches the desired test and dev topics (queries) and qrels from Pyterrier.
    """
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
    """
        Fetches the test and dev topics (queries) and qrels from Pyterrier and evaluates those by calling
        default_test_pipeline.
    """
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
    """
        Loads the dependencies of the default pipeline and returns the default_pipeline, the experiment name and the
        alpha hyperparameter. For DBPedia and Fever, as they used characters such as è, é, ê, ë in the doc ids and an
        additional transformer is used. The elements are transformed in their byte encoding before
        adding them in the sparse index because otherwise Pyterrier truncates them and the doc ids from the first
        (sparse) stage and the second (dense) stage will not be the same.
    """
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
    """
            Loads the dependencies of the default pipeline and runs an experiment. If alpha tuning is enabled, the 'best'
            alpha values is also returned.
    """
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
    """
        Instead of running the complete pipeline, this method fetches the dataset and calls test_first_stage_retrieval
        for running.
    """
    test_set = pt.get_dataset(test_set_name)

    return test_first_stage_retrieval(dataset_name, test_set.get_topics(), test_set.get_qrels(), eval_metrics,
                                      pipeline_name,
                                      path_to_root, timed, in_memory_sparse=in_memory_sparse,
                                      index_path=index_path)


def test_first_stage_retrieval(dataset_name, test_topics, test_qrels, eval_metrics, pipeline_name,
                               path_to_root, timed=False, in_memory_sparse=True,
                               index_path=None):
    """
            Instead of running the complete pipeline, this method runs only the first stage retrieval.
    """
    # Spare index
    retriever = load_sparse_index_from_disk(dataset_name, path_to_root, in_memory=in_memory_sparse,
                                            index_path=index_path)
    if 'fever' in dataset_name or 'dbpedia' in dataset_name:
        encode_utf = EncodeUTF()
        retriever = retriever >> encode_utf

    experiment_name = get_dataset_name(dataset_name) + ": " + pipeline_name

    return run_single_experiment(retriever, test_topics, test_qrels, eval_metrics, experiment_name, timed)


def get_timeit_dependencies_name(dataset_name, test_set_name, q_encoder, model_name,
                                 path_to_root, model_directory, dev_set_name=None, alpha=0.005, in_memory_sparse=True,
                                 in_memory_dense=True, index_path=None):
    """
           Fetches test and dev topics and qrels and the calls get_timeit_dependencies for getting the dependencies
           needed for finding the latency of the second stage retrieval.
    """
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
    """
        Loads the sparse and dense indexes and returns these alongside the most optimal alpha value.
    """
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
    """
        Return the dependencies needed for running the latency experiments. Latency is computed only for the second
        stage retrieval. For that reason, we return the results of the first stage retrieval and the semantic re-ranker.
    """
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
    """
        Runs the default pipeline for multiple datasets and saves the corresponding ranking results on both evaluation
        and development sets.
    """
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

            if not save_trec_files_only:
                result['alpha'] = optimal_alpha
                result.to_csv(file_path, mode='a', header=not os.path.isfile(file_path), index=False)
                print(dataset_names[i] + " DONE")

        except Exception as e:
            # Handles any other exceptions
            print(dataset_names[i] + " FAILED")
            print(f"An error occurred: {e} for dataset {dataset_names[i]}")
            print(traceback.print_exc())

    if not save_trec_files_only:
        duplicate_dataframe = pd.read_csv(file_path)

        print("BEFORE REMOVING DUPLICATES")
        print(duplicate_dataframe)

        unique_dataframe = duplicate_dataframe.groupby('name').tail(1)
        unique_dataframe.to_csv(file_path, index=False)

        return pd.read_csv(file_path)
    else:
        return None


def getOptimalAlpha(dataset_name, pipeline_name, model_directory):
    """
        Fetches the default stored optimal alpha for a specific dataset, model pair.
    """
    experiment_name = get_dataset_name(dataset_name) + ": " + pipeline_name
    path_to_root = os.path.abspath(os.getcwd()) + '/'

    if not save_trec_files_only:
        path_to_root += '../../'

    df = pd.read_csv(path_to_root + model_directory + '/results/ranking_metrics_alpha.csv')
    optimal_alpha = df[df['name'] == experiment_name]['alpha'].iloc[0]
    return optimal_alpha


def time_fct(func, *args, **kwargs):
    """
        Times any function for debugging purposes.
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Experiment took {elapsed_time:.3f} seconds to execute.")
    return result


def latency_per_query(timeit_output, dataset_name, test_suffix, pipeline_name, model_directory,
                      result_filename=""):
    """
        This function parses the timeit out, extract the mean and standard deviation of the altency per query and then
        stores this results.
    """

    if timeit_output.split(" ")[1] == "ms":
        text_input = timeit_output.split(" ms +- ")
        mean_time = float(text_input[0]) * 0.001
    else:
        text_input = timeit_output.split(" s +- ")
        mean_time = float(text_input[0])

    # mean_time = float(text_input[0])
    print(timeit_output)
    try:
        standard_dev_time = float(text_input[1].split(" ms")[0])
    except Exception as e:
        try:
            standard_dev_time = float(text_input[1].split(" s")[0]) * 1000
        except Exception as e:
            standard_dev_time = float(text_input[1].split(" us")[0]) * 0.001

    len_topics = len(pt.get_dataset(dataset_name + test_suffix).get_topics())
    mean_time_per_query = mean_time / len_topics

    # Transform in ms
    mean_time_per_query = mean_time_per_query * 1000
    mean_time_per_query = round(mean_time_per_query, 4)
    store_latency(dataset_name, pipeline_name, mean_time_per_query, mean_time, standard_dev_time, len_topics,
                  timeit_output, model_directory, result_filename)
    return "Latency per query: " + str(mean_time_per_query) + " ms. " + "Experiment details: " + timeit_output


def store_latency(dataset_name, pipeline_name, mean_time_per_query, exp_mean_time, standard_dev_time, len_topics,
                  timeit_output, model_directory, result_filename):
    """
        Stores the latency results.
    """
    path_to_root = os.path.abspath(os.getcwd())
    file_path = path_to_root + '/../../' + model_directory + '/results/' + result_filename + "latency" '_data.csv'

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
    """
        Times a function and prints the results
    """
    print(time_fct(func, *args, **kwargs))
