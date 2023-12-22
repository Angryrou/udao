from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from .base_predicate_embedder import BasePredicateEmbedder
from .word2vec_embedder import Word2VecParams


@dataclass
class Doc2VecParams(Word2VecParams):
    pass


class Doc2VecEmbedder(BasePredicateEmbedder):
    """A class to embed query plans using Doc2Vec.
    To use it:
    - first call fit_transform on a list of training query plans,
    - then call transform on a list of query plans.

    N.B. To ensure reproducibility, several things need to be done:
    - set the seed in the Doc2VecParams
    - set the PYTHONHASHSEED
    - set the number of workers to 1

    Parameters
    ----------
    d2v_params : Doc2VecParams
        Parameters to pass to the gensim.Doc2Vec model
        (ref: https://radimrehurek.com/gensim/models/doc2vec.html)
    """

    def __init__(self, d2v_params: Optional[Doc2VecParams] = None) -> None:
        if d2v_params is None:
            d2v_params = Doc2VecParams()
        self.d2v_params = d2v_params
        self.d2v_model = Doc2Vec(
            min_count=d2v_params.min_count,
            window=d2v_params.window,
            vector_size=d2v_params.vec_size,
            sample=d2v_params.sample,
            alpha=d2v_params.alpha,
            min_alpha=d2v_params.min_alpha,
            workers=d2v_params.workers,
            seed=d2v_params.seed,
            epochs=d2v_params.epochs,
        )
        self._is_trained = False

    def _prepare_corpus(self, texts: Sequence[str], /) -> List[TaggedDocument]:
        """Transform strings into a list of TaggedDocument

        a TaggedDocument consists in a list of tokens and a tag
        (here the index of the plan)
        """
        tokens_list = list(map(lambda x: x.split(), texts))
        corpus = [TaggedDocument(d, [i]) for i, d in enumerate(tokens_list)]
        return corpus

    def fit(
        self, training_texts: Sequence[str], /, epochs: Optional[int] = None
    ) -> None:
        """Train the Doc2Vec model on the training plans

        Parameters
        ----------
        training_plans : Sequence[str]
            list of training plans
        epochs : int, optional
            number of epochs for training the model, by default
            will use the value in the Doc2VecParams.

        Returns
        -------
        np.ndarray
            Normalized (L2) embeddings of the training plans
        """
        if epochs is None:
            epochs = self.d2v_params.epochs
        corpus = self._prepare_corpus(training_texts)
        self.d2v_model.build_vocab(corpus)
        self.d2v_model.train(
            corpus, total_examples=self.d2v_model.corpus_count, epochs=epochs
        )
        self._is_trained = True

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Transform a list of query plans into normalized embeddings.

        Parameters
        ----------
        plans : Sequence[str]
            list of query plans
        epochs: int, optional
            number of epochs for infering a document's embedding,
            by default will use the value in the Doc2VecParams.

        Returns
        -------
        np.ndarray
            Normalized (L2) embeddings of the query plans

        Raises
        ------
        ValueError
            If the model has not been trained
        """
        epochs = self.d2v_params.epochs
        if not self._is_trained:
            raise ValueError("Must call fit_transform before calling transform")
        encodings = [
            self.d2v_model.infer_vector(doc.split(), epochs=epochs) for doc in texts
        ]
        norms = np.linalg.norm(encodings, axis=1)
        # normalize the embeddings
        return encodings / norms[..., np.newaxis]

    def fit_transform(
        self, training_texts: Sequence[str], /, epochs: Optional[int] = None
    ) -> np.ndarray:
        self.fit(training_texts, epochs)
        return self.transform(training_texts)
