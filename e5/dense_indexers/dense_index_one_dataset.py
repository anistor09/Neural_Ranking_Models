import torch
from fast_forward import Mode
from general_dense_indexers.dense_index_one_dataset import index_collection
from encoders.e5 import E5QueryEncoder, E5PassageEncoder


def index_e5_collection(dataset_name, max_id_length, directory, model_name, dim=768):
    q_encoder = E5QueryEncoder("intfloat/" + model_name)
    d_encoder = E5PassageEncoder(
        "intfloat/" + model_name,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    index_collection(dataset_name, model_name, q_encoder, d_encoder, max_id_length, directory, batch_size=8, dim=dim,
                     mode=Mode.MAXP)


def index_e5_base_collection(dataset_name, max_id_length, directory):
    model_name = "e5-base-v2"
    index_e5_collection(dataset_name, max_id_length, directory, model_name)


def index_e5_base_unsupervised_collection(dataset_name, max_id_length, directory):
    model_name = "e5-base-unsupervised"
    index_e5_collection(dataset_name, max_id_length, directory, model_name)


def index_e5_small_collection(dataset_name, max_id_length, directory):
    model_name = "e5-small-v2"
    index_e5_collection(dataset_name, max_id_length, directory, model_name, dim=384)


def main():
    dataset_name = "irds:msmarco-passage"
    max_id_length = 7
    directory = "e5"
    try:
        index_e5_small_collection(dataset_name, max_id_length, directory)

    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
