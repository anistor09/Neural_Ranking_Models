from fast_forward.encoder import Encoder
from transformers import AutoModel, AutoTokenizer
from typing import Callable, Sequence, Union
from pathlib import Path
import numpy as np
import torch


class SnowFlakeDocumentEncoder(Encoder):
    """Uses a pre-trained transformer model for encoding. Returns the pooler output."""

    def __init__(
            self, model: Union[str, Path], device: str = "cpu", **tokenizer_args
    ) -> None:
        """Create a transformer encoder.

        Args:
            model (Union[str, Path]): Pre-trained transformer model (name or path).
            device (str, optional): PyTorch device. Defaults to "cpu".
            **tokenizer_args: Additional tokenizer arguments.
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model)
        # self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model, add_pooling_layer=False)
        self.model.eval()
        # self.device = device
        self.tokenizer_args = tokenizer_args

    def encode(self, documents: Sequence[str]) -> np.ndarray:
        document_tokens = self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=512)
        # random gpt line
        #document_tokens = {key: val.to(self.device) for key, val in document_tokens.items()}

        with torch.no_grad():
            document_embeddings = self.model(**document_tokens)[0][:, 0]
        document_embeddings = torch.nn.functional.normalize(document_embeddings, p=2, dim=1)

        return document_embeddings

    def __call__(self, documents: Sequence[str]) -> np.ndarray:
        return self.encode(documents)


class SnowFlakeQueryEncoder(SnowFlakeDocumentEncoder):
    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        query_prefix = 'Represent this sentence for searching relevant passages: '
        queries_with_prefix = ["{}{}".format(query_prefix, i) for i in queries]
        return self.encode(queries_with_prefix)
