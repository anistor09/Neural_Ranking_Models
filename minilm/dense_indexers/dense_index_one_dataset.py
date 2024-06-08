import torch
from fast_forward import Mode
from general_dense_indexers.dense_index_one_dataset import index_collection
from encoders.minilm import MiniLMEncoder


def index_miniLM_collection(dataset_name, max_id_length, directory, model_name, dim=768):
    q_encoder = MiniLMEncoder("sentence-transformers/" + model_name)
    d_encoder = MiniLMEncoder(
        "sentence-transformers/" + model_name,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    index_collection(dataset_name, model_name, q_encoder, d_encoder, max_id_length, directory, batch_size=8, dim=dim,
                     mode=Mode.MAXP)


def index_miniLM_v2_collection(dataset_name, max_id_length, directory):
    model_name = "all-MiniLM-L6-v2"
    index_miniLM_collection(dataset_name, max_id_length, directory, model_name, dim=384)


def main():
    dataset_name = "irds:msmarco-passage"
    max_id_length = 7
    directory = "minilm"
    try:
        index_miniLM_v2_collection(dataset_name, max_id_length, directory)

    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
