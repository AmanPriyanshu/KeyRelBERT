import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import torch
from keyrelbert.backend import BaseEmbedder
from tqdm import trange


class SentenceTransformerBackend(BaseEmbedder):
    """Sentence-transformers embedding model
    The sentence-transformers embedding model used for generating document and
    word embeddings.

    Arguments:
        embedding_model: A sentence-transformers embedding model

    Usage:

    To create a model, you can load in a string pointing to a
    sentence-transformers model:

    ```python
    from keybert.backend import SentenceTransformerBackend
    sentence_model = SentenceTransformerBackend("all-MiniLM-L6-v2")
    ```

    or  you can instantiate a model yourself:

    ```python
    from keybert.backend import SentenceTransformerBackend
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_model = SentenceTransformerBackend(embedding_model)
    ```
    """

    def __init__(self, embedding_model: Union[str, SentenceTransformer]):
        super().__init__()

        if isinstance(embedding_model, SentenceTransformer):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            raise ValueError(
                "Please select a correct SentenceTransformers model: \n"
                "`from sentence_transformers import SentenceTransformer` \n"
                "`model = SentenceTransformer('all-MiniLM-L6-v2')`"
            )

    def embed(self, documents: List[str], verbose: bool = False, return_detached: bool = False, batch_size: int = 32) -> np.ndarray:
        """Embed a list of n documents/words into an n-dimensional
        matrix of embeddings

        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        if not return_detached:
            embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose, output_value=None)
            embeddings = torch.stack([e['sentence_embedding'].detach() for e in embeddings])
            return embeddings
        else:
            self.embedding_model.train()
            self.embedding_model.to(self.embedding_model._target_device)
            all_embeddings = []
            sentences = documents
            length_sorted_idx = np.argsort([-self.embedding_model._text_length(sen) for sen in sentences])
            sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
            for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not verbose):
                sentences_batch = sentences_sorted[start_index:start_index+batch_size]
                features = self.embedding_model.tokenize(sentences_batch)
                for key in features:
                    if isinstance(features[key], torch.Tensor):
                        features[key] = features[key].to(self.embedding_model._target_device)
                out_features = self.embedding_model.forward(features)
                embeddings = []
                for sent_idx in range(len(out_features['sentence_embedding'])):
                    row =  {name: out_features[name][sent_idx] for name in out_features}
                    embeddings.append(row)
                all_embeddings.extend(embeddings)
            all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
            return all_embeddings