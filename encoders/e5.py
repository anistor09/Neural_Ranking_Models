from fast_forward.encoder import Encoder
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from typing import Sequence, Union
from pathlib import Path
import torch
import torch.nn.functional as F


class E5Encoder(Encoder):
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
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.tokenizer_args = tokenizer_args

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(self, input_texts: Sequence[str]) -> Tensor:
        batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict.to(self.device)

        with torch.no_grad():
            outputs = self.model(**batch_dict).last_hidden_state.detach().cpu()
            attention_mask = batch_dict['attention_mask'].detach().cpu()

        embeddings = self.average_pool(outputs, attention_mask)

        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def __call__(self, input_text: Sequence[str]) -> Tensor:
        return self.encode(input_text)


class E5PassageEncoder(E5Encoder):
    def __call__(self, passage: Sequence[str]) -> Tensor:
        passage_prefix = 'passage: '
        return self.encode([passage_prefix + p for p in passage])


class E5QueryEncoder(E5Encoder):
    def __call__(self, queries: Sequence[str]) -> Tensor:
        query_prefix = 'query: '
        return self.encode([query_prefix + q for q in queries])
