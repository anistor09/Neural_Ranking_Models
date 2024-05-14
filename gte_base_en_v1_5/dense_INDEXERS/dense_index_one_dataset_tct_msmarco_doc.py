import torch
from fast_forward import Mode
from general_dense_indexers.dense_index_one_dataset_2 import index_collection
from fast_forward.encoder import TCTColBERTDocumentEncoder, TCTColBERTQueryEncoder


def index_gte_collection(dataset_name, max_id_length, directory, model_name):
    q_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")
    d_encoder = TCTColBERTDocumentEncoder(
        "castorini/tct_colbert-msmarco",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    index_collection(dataset_name, model_name, q_encoder, d_encoder, max_id_length, directory, batch_size=8, dim=768,
                     mode=Mode.MAXP)


def index_tct_collection(dataset_name, max_id_length, directory):
    model_name = "tct_colbert"
    index_gte_collection(dataset_name, max_id_length, directory, model_name)


def main():
    dataset_name = "irds:msmarco-document/trec-dl-2019"
    max_id_length = 8
    directory = "gte_base_en_v1_5"
    index_tct_collection(dataset_name, max_id_length, directory)


if __name__ == '__main__':
    main()
