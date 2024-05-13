import torch
from fast_forward import Mode
from general_dense_indexers.dense_index_one_dataset import index_collection
from encoders.gte_base_en_encoder import GTEBaseDocumentEncoder


def index_gte_collection(dataset_name, max_id_length, directory, model_name):
    q_encoder = GTEBaseDocumentEncoder("Alibaba-NLP/" + model_name)
    d_encoder = GTEBaseDocumentEncoder(
        "Snowflake/" + model_name,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    index_collection(dataset_name, model_name, q_encoder, d_encoder, max_id_length, directory, batch_size=8, dim=768,
                     mode=Mode.MAXP)


def index_snowflake_m_collection(dataset_name, max_id_length, directory):

    model_name = "gte-base-en-v1.5"
    index_gte_collection(dataset_name, max_id_length, directory, model_name)


def main():
    dataset_name = "irds:beir/fiqa"
    max_id_length = 6
    directory = "snowflake"
    index_snowflake_m_collection(dataset_name, max_id_length, directory)


if __name__ == '__main__':
    main()
