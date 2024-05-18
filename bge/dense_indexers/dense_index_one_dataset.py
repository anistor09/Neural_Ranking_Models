import torch
from fast_forward import Mode
from general_dense_indexers.dense_index_one_dataset_2 import index_collection
from encoders.bge_base_en import BgeEncoder


def index_bge_collection(dataset_name, max_id_length, directory, model_name):
    q_encoder = BgeEncoder("BAAI/" + model_name)
    d_encoder = BgeEncoder(
        "BAAI/" + model_name,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    index_collection(dataset_name, model_name, q_encoder, d_encoder, max_id_length, directory, batch_size=8, dim=768,
                     mode=Mode.MAXP)


def index_bge_base_collection(dataset_name, max_id_length, directory):
    model_name = "bge-base-en-v1.5"
    index_bge_collection(dataset_name, max_id_length, directory, model_name)


def main():
    dataset_name = "irds:beir/dbpedia-entity"
    max_id_length = 200
    directory = "bge"
    try:
        index_bge_base_collection(dataset_name, max_id_length, directory)

    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
