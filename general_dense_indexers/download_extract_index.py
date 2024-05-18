import os
import subprocess
from general_dense_indexers.dense_index_one_dataset import format_name, get_dataset_name
import faiss
from pathlib import Path
from fast_forward import OnDiskIndex


# Function to get the full project path dynamically
def get_project_path():
    return os.path.abspath(os.getcwd())


def get_filename(url):
    return url.split('/')[-1]


def transform_index_to_h5(old_file_dir, new_index_path, index_filename, index_dimension=768):
    index = faiss.read_index(old_file_dir + "/index")
    with open(old_file_dir + "/docid") as fp:
        docids = list(fp.read().splitlines())
    vectors = index.reconstruct_n(0, len(docids))

    OnDiskIndex(Path(new_index_path + "/" + index_filename), index_dimension).add(vectors, doc_ids=docids)


def download_index(model_directory, path_to_root, url, index_filename):
    # Define the base path for operations
    base_path = os.path.join(get_project_path(), path_to_root, model_directory, "dense_indexes")

    # URL of the file to download
    filename = get_filename(url)
    download_path = os.path.join(base_path, filename)

    # Command to download the file
    wget_command = f"wget {url} -P {base_path}"
    print("Downloading file...")
    # subprocess.run(wget_command, shell=True)

    # Command to extract the file
    extract_command = f"tar xf {download_path} -C {base_path}"
    print("Extracting file...")
    # subprocess.run(extract_command, shell=True)

    # Command to remove the downloaded tar file
    remove_command = f"rm {download_path}"
    print("Removing downloaded .tar.gz file...")
    # subprocess.run(remove_command, shell=True)

    new_filename = filename.rsplit('.tar.gz', 1)[0]

    old_file_dir = os.path.join(base_path, new_filename)

    transform_index_to_h5(old_file_dir, base_path, index_filename)

    # Remove old file location
    remove_command = f"rm -r {old_file_dir}"
    print("Removing old file location...")
    subprocess.run(remove_command, shell=True)

    print("Operations completed.")


def main():
    model_directory = "bge"

    # if run from project root path-to_root = ""

    path_to_root = ""
    prefix_url = "https://rgw.cs.uwaterloo.ca/pyserini/indexes/faiss/"
    downloaded_file = "faiss-flat.msmarco-v1-passage.bge-base-en-v1.5.20240107.tar.gz"
    dataset_name = "irds:msmarco-passage/trec-dl-2019"
    model_name = "bge-base-en-v1.5"
    index_filename = "ffindex_" + get_dataset_name(dataset_name) + "_" + format_name(model_name) + ".h5"
    try:
        download_index(model_directory, path_to_root, prefix_url + downloaded_file, index_filename)

    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
