from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, Word2Vec
from gensim.models.phrases import Phraser, Phrases

from .base_predicate_embedder import BasePredicateEmbedder


@dataclass
class Word2VecParams:
    """
    Parameters to pass to the gensim.Word2Vec model
    (ref: https://radimrehurek.com/gensim/models/word2vec.html)
    """

    min_count: int = 1
    window: int = 3
    vec_size: int = 32
    alpha: float = 0.025
    sample: float = 0.1
    min_alpha: float = 0.0007
    workers: int = 1  # mp.cpu_count() - 1
    seed: int = 42
    epochs: int = 10


class Word2VecEmbedder(BasePredicateEmbedder):
    """
    A class to embed query plans using Word2Vec.
    The embedding is computed as the average of the word
    embeddings of the words in the query plan.

    To use it:
    - first call fit_transform on a list of training query plans,
    - then call transform on a list of query plans.

    N.B. To ensure reproducibility, several things need to be done:
    - set the seed in the Word2VecParams
    - set the PYTHONHASHSEED
    - set the number of workers to 1

    Parameters
    ----------
    w2v_params : Word2VecParams
        Parameters to pass to the gensim.Word2Vec model
        (ref: https://radimrehurek.com/gensim/models/word2vec.html)
    """

    def __init__(self, w2v_params: Optional[Word2VecParams] = None) -> None:
        if w2v_params is None:
            w2v_params = Word2VecParams()
        self.w2v_params = w2v_params
        self.w2v_model = Word2Vec(
            min_count=w2v_params.min_count,
            window=w2v_params.window,  # 3
            vector_size=w2v_params.vec_size,  # 32
            sample=w2v_params.sample,
            alpha=w2v_params.alpha,  # 0.025
            min_alpha=w2v_params.min_alpha,
            workers=w2v_params.workers,
            seed=w2v_params.seed,
            epochs=w2v_params.epochs,
        )
        self._bigram_model: Optional[Phraser] = None
        self.dictionary = Dictionary()
        self.tfidf_model = TfidfModel(smartirs="ntc")

    def _get_encodings_from_corpus(
        self, bow_corpus: List[List[Tuple[int, int]]]
    ) -> np.ndarray:
        """Compute the encodings from a bow_corpus"""
        return np.array(
            [
                np.mean(
                    [
                        self.w2v_model.wv.get_vector(self.dictionary[wid], norm=True)
                        * freq
                        for wid, freq in wt
                    ],
                    0,
                )
                for wt in self.tfidf_model[bow_corpus]
            ]
        )

    def fit_transform(
        self, training_texts: Sequence[str], epochs: Optional[int] = None
    ) -> np.ndarray:
        """Train the Word2Vec model on the training texts and return the embeddings.

        Parameters
        ----------
        training_texts : Sequence[str]
            list of training texts
        epochs : int, optional
            number of epochs for training the model, by default will use the value
            in the Word2VecParams.

        Returns
        -------
        np.ndarray
            Embeddings of the training plans
        """
        if epochs is None:
            epochs = self.w2v_params.epochs
        training_sentences = [row.split() for row in training_texts]
        phrases = Phrases(training_sentences, min_count=30)  # Extract parameter
        self._bigram_model = Phraser(phrases)
        training_descriptions = self._bigram_model[training_sentences]
        self.w2v_model.build_vocab(training_descriptions)
        self.w2v_model.train(
            training_descriptions,
            total_examples=self.w2v_model.corpus_count,
            epochs=epochs,
            report_delay=1,
        )
        # print(f"get {self.w2v_model.wv.vectors.shape[0]} words from word2vec")
        bow_corpus: List[List[Tuple[int, int]]] = [
            self.dictionary.doc2bow(doc, allow_update=True, return_missing=False)  # type: ignore
            for doc in training_descriptions
        ]
        self.tfidf_model.initialize(bow_corpus)
        # print(f"get {len(self.tfidf.idfs)} words from tfidf")
        return self._get_encodings_from_corpus(bow_corpus=bow_corpus)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Transform a list of query plans into embeddings.

        Parameters
        ----------
        texts : Sequence[str]
            list of texts to transform
        epochs : int, optional
            number of epochs for infering a document's embedding,
            by default will use the value in the Word2VecParams.

        Returns
        -------
        np.ndarray
            Embeddings of the query plans

        Raises
        ------
        ValueError
            If the model has not been trained
        """
        sentences = [row.split() for row in texts]
        if self._bigram_model is None:
            raise ValueError("Must call fit_transform before calling transform")
        descriptions = self._bigram_model[sentences]

        bow_corpus: List[List[Tuple[int, int]]] = [
            self.dictionary.doc2bow(doc, allow_update=False, return_missing=False)  # type: ignore
            for doc in descriptions
        ]

        return self._get_encodings_from_corpus(bow_corpus=bow_corpus)
