from fast_forward.encoder import Encoder
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from typing import Sequence, Union
from pathlib import Path
import torch
import torch.nn.functional as F


class NomicEncoder(Encoder):
    """
        An encoder that uses a transformer model to generate embeddings for text inputs. It is built
        to handle general text encoding tasks with mean pooling to combine token embeddings.
    """

    def __init__(
            self, model: Union[str, Path], device: str = "cpu", **tokenizer_args
    ) -> None:
        """
            Initializes the encoder with a specified transformer model and tokenizer. It sets up the device
            for computations and puts the model in evaluation mode to disable training specific behaviors.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.model = AutoModel.from_pretrained(model, trust_remote_code=True)

        self.model.to(device)
        self.model.eval()
        self.device = device
        self.tokenizer_args = tokenizer_args

    def mean_pooling(self, model_output, attention_mask):
        """
           Applies mean pooling to the output of the transformer model using the attention mask to
           handle padding correctly. This method enhances the representation by considering only valid tokens.
        """
        token_embeddings = model_output[0].detach().cpu()
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentences: Sequence[str]) -> Tensor:
        """
            Encodes a list of sentences into embeddings using the transformer model, applying mean pooling
            on the outputs and normalizing the resulting embeddings.
        """
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


class NomicDocumentEncoder(NomicEncoder):
    """
        Specialized encoder for documents that prefixes each document with 'search_document: ' to tailor the
        model's understanding towards document-like contexts.
    """

    def __call__(self, document: Sequence[str]) -> Tensor:
        passage_prefix = 'search_document: '
        return self.encode([passage_prefix + d for d in document])


class NomicQueryEncoder(NomicEncoder):
    """
        Specialized encoder for queries that prefixes each query with 'search_query: ' to tailor the
        model's understanding towards query-like contexts.
    """

    def __call__(self, queries: Sequence[str]) -> Tensor:
        query_prefix = 'search_query: '
        return self.encode([query_prefix + q for q in queries])
