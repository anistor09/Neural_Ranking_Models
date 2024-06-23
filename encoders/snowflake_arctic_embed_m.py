from fast_forward.encoder import Encoder
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from typing import Sequence, Union
from pathlib import Path
import torch


class SnowFlakeDocumentEncoder(Encoder):
    """
        Encoder using a pre-trained transformer model to generate embeddings for documents.
        It omits the pooling layer for direct extraction of token embeddings.
    """

    def __init__(
            self, model: Union[str, Path], device: str = "cpu", **tokenizer_args
    ) -> None:
        """
            Initializes the encoder with a specified transformer model. Configures the tokenizer and model for usage.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, add_pooling_layer=False)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.tokenizer_args = tokenizer_args

    def encode(self, documents: Sequence[str]) -> Tensor:
        """
            Encodes documents into normalized embeddings using the transformer model.
        """
        document_tokens = self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=512)
        document_tokens.to(self.device)

        with torch.no_grad():
            document_embeddings = self.model(**document_tokens)[0][:, 0].detach().cpu()

        document_embeddings = torch.nn.functional.normalize(document_embeddings, p=2, dim=1)

        return document_embeddings

    def __call__(self, documents: Sequence[str]) -> Tensor:
        return self.encode(documents)


class SnowFlakeQueryEncoder(SnowFlakeDocumentEncoder):
    """
        Extends the SnowFlakeDocumentEncoder to tailor embeddings specifically for query data by prepending a standard prefix.
    """

    def __call__(self, queries: Sequence[str]) -> Tensor:
        """
            Encodes queries by prepending a fixed instruction to enhance relevance for search applications.
        """
        query_prefix = 'Represent this sentence for searching relevant passages: '
        queries_with_prefix = ["{}{}".format(query_prefix, i) for i in queries]

        return self.encode(queries_with_prefix)
