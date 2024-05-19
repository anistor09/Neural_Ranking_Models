import torch
from fast_forward import Mode
from general_dense_indexers.dense_index_one_dataset_2 import index_collection
from encoders.snowflake_arctic_embed_m import SnowFlakeDocumentEncoder, SnowFlakeQueryEncoder


def index_snowflake_collection(dataset_name, max_id_length, directory, model_name):
    q_encoder = SnowFlakeQueryEncoder("Snowflake/" + model_name)
    d_encoder = SnowFlakeDocumentEncoder(
        "Snowflake/" + model_name,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    index_collection(dataset_name, model_name, q_encoder, d_encoder, max_id_length, directory, batch_size=8, dim=768,
                     mode=Mode.MAXP)


def index_snowflake_m_collection(dataset_name, max_id_length, directory):
    model_name = "snowflake-arctic-embed-m"
    index_snowflake_collection(dataset_name, max_id_length, directory, model_name)


def main():
    dataset_name = "irds:beir/dbpedia-entity"
    max_id_length = 200
    directory = "snowflake"
    index_snowflake_m_collection(dataset_name, max_id_length, directory)


if __name__ == '__main__':
    main()
