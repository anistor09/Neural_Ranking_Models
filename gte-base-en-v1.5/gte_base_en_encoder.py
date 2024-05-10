from fast_forward.encoder import Encoder
from transformers import AutoModel, AutoTokenizer
from typing import Callable, Sequence, Union
from pathlib import Path
import numpy as np
import torch


class GTEBaseDocumentEncoder(Encoder):
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
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True)
        self.model.to(device)
        # self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.device = device
        self.tokenizer_args = tokenizer_args

    def __call__(self, input_texts: Sequence[str]) -> np.ndarray:
        # Tokenize the input texts
        batch_dict = self.tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        batch_dict.to(self.device)

        ## NO GRad added by me
        with torch.no_grad():
            outputs = self.model(**batch_dict)

        embeddings = outputs.last_hidden_state[:, 0].detach().cpu()

        # (Optionally) normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
