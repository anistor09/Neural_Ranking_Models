import torch
from fast_forward import Mode
from general_dense_indexers.dense_index_one_dataset import index_collection
from encoders.gte_base_en_encoder import GTEBaseEncoder


def index_gte_collection(dataset_name, max_id_length, directory, model_name):
    """
         Indexes a dataset using the GTEBaseEncoder for both query and document encoders.

         Args:
         dataset_name (str): The name of the dataset to index.
         max_id_length (int): Maximum length of the IDs in the dataset.
         directory (str): Directory to store the indexed files.
         model_name (str): Name of the model used for encoding.

         Details:
         Uses the GTEBaseEncoder from Alibaba-NLP with the specified model. If available, uses GPU for document encoding
         , otherwise defaults to CPU.
         """

    q_encoder = GTEBaseEncoder("Alibaba-NLP/" + model_name)
    d_encoder = GTEBaseEncoder(
        "Alibaba-NLP/" + model_name,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    index_collection(dataset_name, model_name, q_encoder, d_encoder, max_id_length, directory, batch_size=8, dim=768,
                     mode=Mode.MAXP)


def index_gte_base_collection(dataset_name, max_id_length, directory):
    """
      Indexes a dataset using the 'gte-base-en-v1.5' model.

      Args:
      dataset_name (str): The name of the dataset to index.
      max_id_length (int): Maximum length of the IDs in the dataset.
      directory (str): Directory to store the indexed files.
      """
    model_name = "gte-base-en-v1.5"
    index_gte_collection(dataset_name, max_id_length, directory, model_name)


def main():
    dataset_name = "irds:beir/fever"
    max_id_length = 221
    directory = "gte_base_en_v1_5"
    try:
        index_gte_base_collection(dataset_name, max_id_length, directory)

    except Exception as e:
        # Handles any other exceptions
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
