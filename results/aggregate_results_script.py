import os
import pandas as pd

# Usage
path_to_root = os.path.abspath(os.getcwd()) + "/../"
output_file = 'aggreagated_latency_results.csv'
directories = ["bge", "gte_base_en_v1_5", "snowflake", "tct_colbert"]
res_path = "/results/latency_data.csv"


def aggregate_csv_files():
    # Create a list to hold data from each CSV file
    data_frames = []

    file_paths = [path_to_root + dir_path + res_path for dir_path in directories]
    print(file_paths)
    # Loop over all files in the directory
    for file_path in file_paths:
        df = pd.read_csv(file_path)

        # Append a blank row at the end of each DataFrame
        blank_row = pd.DataFrame({col: [''] for col in df.columns})
        df = pd.concat([df, blank_row], ignore_index=True)

        data_frames.append(df)

    # Concatenate all the data frames into one
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Save the combined data frame to a new CSV file, write header only once
    combined_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    aggregate_csv_files()
