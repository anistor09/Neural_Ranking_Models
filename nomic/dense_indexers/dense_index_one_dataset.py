import torch
from fast_forward import Mode
from general_dense_indexers.dense_index_one_dataset import index_collection
from encoders.nomic import NomicQueryEncoder, NomicDocumentEncoder


def index_nomic_collection(dataset_name, max_id_length, directory, model_name, dim=768):
    q_encoder = NomicQueryEncoder("nomic-ai/" + model_name)
    d_encoder = NomicDocumentEncoder(
        "nomic-ai/" + model_name,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    index_collection(dataset_name, model_name, q_encoder, d_encoder, max_id_length, directory, batch_size=8, dim=dim,
                     mode=Mode.MAXP)


def index_nomic_v1_collection(dataset_name, max_id_length, directory):
    model_name = "nomic-embed-text-v1"
    index_nomic_collection(dataset_name, max_id_length, directory, model_name)


def main():
    dataset_name = "irds:msmarco-passage"
    max_id_length = 7
    directory = "nomic"
    try:
        index_nomic_v1_collection(dataset_name, max_id_length, directory)

    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
