from fast_forward.encoder import Encoder
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from typing import Sequence, Union
from pathlib import Path
import torch


class BgeEncoder(Encoder):
    """
        BGE Encoder class that utilizes a transformer model for generating embeddings of documents.
        This encoder can handle any text and is designed to be used with documents for generating
        dense vector representations.
        """

    def __init__(
            self, model: Union[str, Path], device: str = "cpu", **tokenizer_args
    ) -> None:
        """
                Initializes the encoder with a specified transformer model and tokenizer.

                The encoder can be initialized with a model path or model identifier from Hugging Face's
                transformer library. The device parameter specifies where the tensor computations will occur,
                i.e., on CPU or GPU.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.tokenizer_args = tokenizer_args

    def encode(self, documents: Sequence[str]) -> Tensor:
        """
            Converts a batch of documents into their corresponding embeddings using the transformer model.

            This method handles tokenization and converts the documents into embeddings by passing the tokenized
            text through the transformer model. The embeddings are normalized before returning.
        """
        # Tokenize sentences
        document_tokens = self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt')
        document_tokens.to(self.device)

        # Tokenize sentences and perform pooling.
        with torch.no_grad():
            document_embeddings = self.model(**document_tokens)[0][:, 0].detach().cpu()

        document_embeddings = torch.nn.functional.normalize(document_embeddings, p=2, dim=1)

        return document_embeddings

    def __call__(self, documents: Sequence[str]) -> Tensor:
        return self.encode(documents)


class BgeQueryEncoder(BgeEncoder):
    """
        Specialized version of the BgeEncoder tailored for encoding search queries.

        This encoder prefixes a specific instruction to each query to guide the model's
        contextual understanding, thereby enhancing the relevance of the generated embeddings
        for search applications.
    """

    def __call__(self, queries: Sequence[str]) -> Tensor:
        """
            Encodes a batch of search queries after appending a predefined instruction that
            helps in generating more contextual embeddings.
        """
        instruction = "Represent this sentence for searching relevant passages: "
        return self.encode([instruction + q for q in queries])
