from fast_forward.encoder import Encoder
from transformers import AutoModel, AutoTokenizer
from typing import Callable, Sequence, Union
from pathlib import Path
import numpy as np
import torch


class GTEBaseEncoder(Encoder):
    """
        Encoder class utilizing the General Transformer Encoder (GTE) from Hugging Face's Transformers library.
        This encoder is tailored for generating embeddings from a variety of texts using a specified transformer model.
    """

    def __init__(
            self, model: Union[str, Path], device: str = "cpu", **tokenizer_args
    ) -> None:
        """
            Initializes the encoder with a transformer model and tokenizer, specifying the device for tensor computations.
            The encoder can be initialized with a model path or model identifier and is designed to handle the
            specifics of tokenizer and model configurations.
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True)
        self.model.to(device)
        # self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.device = device
        self.tokenizer_args = tokenizer_args

    def __call__(self, input_texts: Sequence[str]) -> np.ndarray:
        """
            Processes input texts into embeddings using the transformer model.
            This method handles tokenization, computation of embeddings, and optional normalization.
        """
        # Tokenize the input texts
        batch_dict = self.tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        batch_dict.to(self.device)

        # No need to update the weights during inference
        with torch.no_grad():
            outputs = self.model(**batch_dict)

        embeddings = outputs.last_hidden_state[:, 0].detach().cpu()

        # (Optionally) normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
