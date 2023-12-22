import numpy as np
import pytest

from ...predicate_embedders import Word2VecEmbedder, Word2VecParams


@pytest.fixture
def word2vec_embedder() -> Word2VecEmbedder:
    return Word2VecEmbedder(Word2VecParams())


class TestWord2VecEmbedder:
    def test_init(self, word2vec_embedder: Word2VecEmbedder) -> None:
        assert word2vec_embedder.w2v_model is not None
        assert word2vec_embedder._bigram_model is None

    def test_fit_transform(self, word2vec_embedder: Word2VecEmbedder) -> None:
        training_plans = ["a b c", "a b d"]
        training_encodings = word2vec_embedder.fit_transform(training_plans)
        assert word2vec_embedder._bigram_model is not None
        # 4 words
        assert word2vec_embedder.w2v_model.wv.vectors.shape[0] == 4
        # 32 dimensions - corresponds to param vec_size
        assert word2vec_embedder.w2v_model.wv.vectors.shape[1] == 32
        # 2 training plans with dimension 32
        assert training_encodings.shape == (2, 32)

    def test_transform_not_trained(self, word2vec_embedder: Word2VecEmbedder) -> None:
        with pytest.raises(ValueError):
            word2vec_embedder.transform(["a b c"])

    def test_transform_trained(self, word2vec_embedder: Word2VecEmbedder) -> None:
        training_plans = ["a b c", "a b d"]
        training_encodings = word2vec_embedder.fit_transform(training_plans)
        encoding = word2vec_embedder.transform(["a b c", "a b x"])
        assert np.array_equal(training_encodings[0], encoding[0])
        assert encoding.shape == (2, 32)
