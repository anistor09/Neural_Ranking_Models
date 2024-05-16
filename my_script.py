import subprocess

def run_command():
    # Define the command to be run
    command = "python -m gte_base_en_v1_5.dense_INDEXERS.dense_index_one_dataset_msmarco_passage"

    # Run the command
    try:
        output = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Command executed successfully:")
        print(output.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred while executing the command:")
        print(e.stderr)

if __name__ == "__main__":
    run_command()