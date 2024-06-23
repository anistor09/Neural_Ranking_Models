import torch
from fast_forward import Mode
from general_dense_indexers.dense_index_one_dataset import index_collection
from fast_forward.encoder import TCTColBERTQueryEncoder, TCTColBERTDocumentEncoder


def index_castorini_collection(dataset_name, max_id_length, directory, model_name):
    q_encoder = TCTColBERTQueryEncoder("castorini/" + model_name)
    d_encoder = TCTColBERTDocumentEncoder(
        "castorini/" + model_name,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    index_collection(dataset_name, model_name, q_encoder, d_encoder, max_id_length, directory, batch_size=8, dim=768,
                     mode=Mode.MAXP)


def index_tct_colbert(dataset_name, max_id_length, directory):
    model_name = "tct_colbert-msmarco"
    index_castorini_collection(dataset_name, max_id_length, directory, model_name)


def main():
    dataset_name = "irds:beir/hotpotqa"
    max_id_length = 8
    directory = "tct_colbert"

    try:
        index_tct_colbert(dataset_name, max_id_length, directory)

    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
