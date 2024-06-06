import os
import pandas as pd

# Usage
path_to_root = "/"
# if run from file location -> path_to_root =  "/../"
path_to_root = os.path.abspath(os.getcwd()) + path_to_root
directories = ["bge", "gte_base_en_v1_5", "snowflake", "tct_colbert"]


def splits(name):
    parts = name.split(" ", 1)
    x = parts[1], parts[0]
    return x


def name_comparator(names):
    return [splits(name) for name in names]


def aggregate_csv_files(input_files, output_file):
    # Create a list to hold data from each CSV file
    data_frames = []

    input_files = "/results/" + input_files
    output_file = path_to_root + "/results/" + output_file

    file_paths = [path_to_root + dir_path + input_files for dir_path in directories]
    print(file_paths)
    # Loop over all files in the directory
    for file_path in file_paths:
        df = pd.read_csv(file_path)

        # Append a blank row at the end of each DataFrame
        # blank_row = pd.DataFrame({col: [''] for col in df.columns})
        # df = pd.concat([df, blank_row], ignore_index=True)

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

    unique_dataframe = duplicate_dataframe.groupby('name').tail(1)

    sorted_df = unique_dataframe.sort_values(by='name', key=name_comparator)

    sorted_df = sorted_df.round(4)

    # Save the combined data frame to a new CSV file, write header only once
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


if __name__ == '__main__':
    input_files = "latency_data.csv"
    output_file = "aggreagated_latency_results.csv"
    aggregate_csv_files(input_files, output_file)
    # remove_duplicated_results(input_files)
