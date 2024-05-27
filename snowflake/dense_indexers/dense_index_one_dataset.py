import torch
from fast_forward import Mode
from general_dense_indexers.dense_index_one_dataset import index_collection
from encoders.snowflake_arctic_embed_m import SnowFlakeDocumentEncoder, SnowFlakeQueryEncoder
from func_timeout import func_timeout

def index_snowflake_collection(dataset_name, max_id_length, directory, model_name, dim=768):
    q_encoder = SnowFlakeQueryEncoder("Snowflake/" + model_name)
    d_encoder = SnowFlakeDocumentEncoder(
        "Snowflake/" + model_name,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    index_collection(dataset_name, model_name, q_encoder, d_encoder, max_id_length, directory, batch_size=8, dim=dim,
                     mode=Mode.MAXP)


def index_snowflake_m_collection(dataset_name, max_id_length, directory):
    model_name = "snowflake-arctic-embed-m"
    index_snowflake_collection(dataset_name, max_id_length, directory, model_name)


def index_snowflake_xs_collection(dataset_name, max_id_length, directory):
    model_name = "snowflake-arctic-embed-xs"
    index_snowflake_collection(dataset_name, max_id_length, directory, model_name, dim=384)


def main():
    dataset_name = "irds:msmarco-passage"
    max_id_length = 7
    directory = "snowflake"

    try:
        func_timeout(9 * 3600 - 120, index_snowflake_xs_collection, args =(dataset_name, max_id_length, directory))
    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")



if __name__ == '__main__':
    main()
