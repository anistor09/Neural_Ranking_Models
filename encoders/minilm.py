from fast_forward.encoder import Encoder
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from typing import Sequence, Union
from pathlib import Path
import torch
import torch.nn.functional as F


class MiniLMEncoder(Encoder):
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

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0].detach().cpu()  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentences: Sequence[str]) -> Tensor:
        batch_dict = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        batch_dict.to(self.device)

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            attention_mask = batch_dict['attention_mask'].detach().cpu()

        embeddings = self.mean_pooling(outputs, attention_mask)

        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def __call__(self, input_text: Sequence[str]) -> Tensor:
        return self.encode(input_text)
