import os
import pandas as pd

# Usage
path_to_root = "/"
# if run from file location -> path_to_root =  "/../"
path_to_root = os.path.abspath(os.getcwd()) + path_to_root
directories = ["bge", "gte_base_en_v1_5", "snowflake", "tct_colbert", "e5", "nomic"]


def splits(name):
    parts = name.split(" ", 1)
    x = parts[1], parts[0]
    return x


def name_comparator(names):
    return [splits(name) for name in names]


def get_short_name(name):
    if "gte" in name or "bge" in name:
        return '-'.join(name.split('-')[:2])
    elif "arctic" in name:
        return 'arctic' + '-' + name.split('-')[-1]
    elif "tct" in name:
        return '-'.join(name.split('_')[:2])
    elif "e5" in name:
        if "unsupervised" in name:
            return "e5-base-pt"
        else:
            return '-'.join(name.split('-')[:2])
    elif "nomic" in name:
        return "nomic"
    else:
        return name


def aggregate_csv_files(input_files, output_file):
    # Create a list to hold data from each CSV file
    data_frames = []

    input_files = "/results/" + input_files
    output_file = path_to_root + "/results/" + output_file

    file_paths = [path_to_root + dir_path + input_files for dir_path in directories]

    # Loop over all files in the directory
    for file_path in file_paths:
        df = pd.read_csv(file_path)

        data_frames.append(df)

    # Concatenate all the data frames into one
    duplicate_dataframe = pd.concat(data_frames, ignore_index=True)

    print("BEFORE REMOVING DUPLICATES")
    print(duplicate_dataframe)

    duplicate_dataframe = duplicate_dataframe.copy()

    if "ranking_metrics" in input_files:
        duplicate_dataframe[['dataset', 'pipeline_name']] = duplicate_dataframe['name'].str.split(' ', n=1, expand=True)
        duplicate_dataframe['dataset'] = duplicate_dataframe['dataset'].str[:-1]
        duplicate_dataframe['model'] = duplicate_dataframe['pipeline_name'].apply(lambda x: x.split(' ')[2])

    else:
        duplicate_dataframe['model'] = duplicate_dataframe['pipeline_name'].apply(lambda x: x.split(' ')[2])
        duplicate_dataframe['name'] = duplicate_dataframe['dataset'] + ": " + duplicate_dataframe['pipeline_name']

    unique_dataframe = duplicate_dataframe.groupby('name').tail(1)

    unique_dataframe['model'] = unique_dataframe['model'].apply(lambda x: get_short_name(x))

    sorted_df = unique_dataframe.sort_values(by='name', key=name_comparator)

    sorted_df = sorted_df.round(4)

    # Save the combined data frame to a new CSV file, write header only once
    sorted_df.to_csv(output_file, index=False)


def merge_ranking_latency_aggregations():
    ranking_results = path_to_root + "/results/" + "aggreagated_ranking_results.csv"
    latency_results = path_to_root + "/results/" + "aggreagated_latency_results.csv"
    output_file = path_to_root + "/results/" + "aggreagated_ranking_latency_results.csv"

    df_ranking = pd.read_csv(ranking_results)
    df_latency = pd.read_csv(latency_results)
    merged_df = pd.merge(df_ranking, df_latency, on=['dataset', 'model', 'name', 'pipeline_name'], how='inner')
    columns_to_keep = ['dataset', 'model', 'mean_time_per_query (ms)', 'nDCG@10']
    merged_df = merged_df[columns_to_keep]

    sorted_df = merged_df.sort_values(by='model')
    sorted_df.to_csv(output_file, index=False)


def merge_latency_parts_aggregations():
    query_encoding_latency = path_to_root + "/results/" + "aggreagated_query_encoding_latency_results.csv"
    vector_embedding_latency = path_to_root + "/results/" + "aggreagated_vector_embedding_latency_results.csv"
    complete_latency_results = path_to_root + "/results/" + "aggreagated_latency_results.csv"
    output_file = path_to_root + "/results/" + "aggreagated_latency_parts_results.csv"

    columns_to_keep = ['dataset', 'model', 'mean_time_per_query (ms)']

    df_qe = pd.read_csv(query_encoding_latency)
    df_qe = df_qe[columns_to_keep]
    df_ver = pd.read_csv(vector_embedding_latency)
    df_ver = df_ver[columns_to_keep]
    df_complete = pd.read_csv(complete_latency_results)
    df_complete = df_complete[columns_to_keep]

    df_complete['mean_time_per_query (ms)_total_latency'] = df_complete['mean_time_per_query (ms)']

    df_complete = df_complete.drop(columns=['mean_time_per_query (ms)'])

    segments_df = pd.merge(df_qe, df_ver, on=['dataset', 'model'],
                           suffixes=('_query_embedding', '_embeddings_retrieval'))
    # print(segments_df)

    merged_df = pd.merge(segments_df, df_complete, on=['dataset', 'model'])

    merged_df['query_embedding'] = merged_df['mean_time_per_query (ms)_query_embedding']
    merged_df['doc_retrieval'] = merged_df['mean_time_per_query (ms)_embeddings_retrieval']
    merged_df['total_latency'] = merged_df['mean_time_per_query (ms)_total_latency']

    dropped_cols = ['mean_time_per_query (ms)_query_embedding', 'mean_time_per_query (ms)_embeddings_retrieval',
                    'mean_time_per_query (ms)_total_latency']

    merged_df = merged_df.drop(columns=dropped_cols)

    sorted_df = merged_df.sort_values(by='model')

    filtered = sorted_df[
        sorted_df['total_latency'] - sorted_df['query_embedding'] < 3]

    # Displaying the filtered DataFrame
    print(filtered)

    sorted_df.to_csv(output_file, index=False)


def remove_duplicated_results(input_files):
    # Create a list to hold data from each CSV file

    file_paths_ranking_metrics = [path_to_root + dir_path + "/results/" + input_files for dir_path in directories]

    print(file_paths_ranking_metrics)
    # Loop over all files in the directory
    for file_path in file_paths_ranking_metrics:
        duplicate_dataframe = pd.read_csv(file_path)

        print("BEFORE REMOVING DUPLICATES")
        print(duplicate_dataframe)
        duplicate_dataframe['name'] = duplicate_dataframe['dataset'] + ": " + duplicate_dataframe['pipeline_name']
        unique_dataframe = duplicate_dataframe.groupby('name').tail(1)
        unique_dataframe.to_csv(file_path, index=False)

        print("AFTER REMOVING DUPLICATES")
        print(unique_dataframe)


bm25_values_rr = {
    "passage": 0.7944,
    "nfcorpus": 0.5344,
    "hotpotqa": 0.6624,
    "fiqa": 0.3103,
    "quora": 0.7584,
    "dbpedia_entity": 0.5268,
    "fever": 0.3839,
    "scifact": 0.6324
}

bm25_values_ndcg = {
    "passage": 0.4795,
    "nfcorpus": 0.3223,
    "hotpotqa": 0.5128,
    "fiqa": 0.2526,
    "quora": 0.7676,
    "dbpedia_entity": 0.2744,
    "fever": 0.4273,
    "scifact": 0.6722
}

best_ndcg = [0.7137, 0.3649, 0.7307, ]
best_rr = []


def create_latex_table(target_value):
    models = ["BM25", "tct-colbert", "gte-base", "bge-base", "arctic-m", "e5-base", "e5-base-pt", "nomic", "bge-small",
              "arctic-xs", "e5-small"]

    datasets = ['passage', 'nfcorpus', 'hotpotqa', 'fiqa', 'quora', 'dbpedia_entity', 'fever', 'scifact']
    latex_table = ""

    ranking_results_path = path_to_root + "/results/" + "aggreagated_ranking_results.csv"

    signifiance_path = path_to_root + "/results/significance_reports/extracted_signifiance_relations"

    df = pd.read_csv(ranking_results_path)

    for dataset in datasets:
        df_signfiance = None
        if 'fever' not in dataset:
            df_signfiance = pd.read_csv(signifiance_path + "/" + dataset + '.csv')
        row = dataset + " & "

        for model in models:
            is_max = False
            if model == "BM25":
                if target_value == "RR@10":
                    value = bm25_values_rr[dataset]
                elif target_value == "nDCG@10":
                    value = bm25_values_ndcg[dataset]
                else:
                    value = ""

            else:

                try:
                    max_val = df[(df['dataset'] == dataset)][target_value].max()

                    value = df[(df['model'] == model) & (df['dataset'] == dataset)][target_value].iloc[0]

                    if value == max_val:
                        is_max = True

                except Exception as e:
                    value = "-"
            value = str(value)
            if is_max:
                value = "\\col{" + value + "}"

            filtered_df = None

            if df_signfiance is not None:
                filtered_df = df_signfiance[(df_signfiance['Model'] == model)]
            sig = None

            if filtered_df is not None and not filtered_df.empty:
                # print(dataset + " " + model + " " + filtered_df[target_value])
                sig = filtered_df[target_value].iloc[0]

            if sig is not None and pd.notna(sig) and 'a' in sig:
                value = "\\textbf{" + value + "}"

            row += value + " & "
        row = row[:-2]
        row += " \\\\" + " \n"
        latex_table += row

    print(latex_table)


if __name__ == '__main__':
    # input_files = "ranking_metrics_alpha.csv"
    # output_file = "aggreagated_ranking_results.csv"
    # aggregate_csv_files(input_files, output_file)
    #
    # input_files = "latency_data.csv"
    # output_file = "aggreagated_latency_results.csv"
    # aggregate_csv_files(input_files, output_file)
    # merge_ranking_latency_aggregations()

    # input_files = "query_encoding_latency_data.csv"
    # output_file = "aggreagated_query_encoding_latency_results.csv"
    # aggregate_csv_files(input_files, output_file)
    #
    # input_files = "vector_embedding_latency_data.csv"
    # output_file = "aggreagated_vector_embedding_latency_results.csv"
    # aggregate_csv_files(input_files, output_file)
    #
    # merge_latency_parts_aggregations()

    extracted_value = "RR@10"
    create_latex_table(extracted_value)
