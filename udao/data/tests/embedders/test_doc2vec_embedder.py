import numpy as np
import pytest

from ...predicate_embedders import Doc2VecEmbedder, Doc2VecParams


@pytest.fixture
def doc2vec_embedder() -> Doc2VecEmbedder:
    return Doc2VecEmbedder(Doc2VecParams())


class TestDoc2Vec:
    def test_init(self, doc2vec_embedder: Doc2VecEmbedder) -> None:
        assert doc2vec_embedder.d2v_model is not None
        assert doc2vec_embedder._is_trained is False

    def test_fit_sets_is_trained(self, doc2vec_embedder: Doc2VecEmbedder) -> None:
        training_plans = ["a b c", "a b d"]
        doc2vec_embedder.fit(training_plans)
        assert doc2vec_embedder._is_trained is True

    def test_transform_not_trained_raises_error(
        self, doc2vec_embedder: Doc2VecEmbedder
    ) -> None:
        with pytest.raises(ValueError):
            doc2vec_embedder.transform(["a b c"])

    def test_transform_trained_output_values(
        self, doc2vec_embedder: Doc2VecEmbedder
    ) -> None:
        training_plans = ["a b c", "a b d"]
        doc2vec_embedder.fit(training_plans)
        training_encodings = doc2vec_embedder.transform(training_plans)
        encoding = doc2vec_embedder.transform(["a b c", "a b x"])
        dot = np.dot(training_encodings[0], encoding[0])
        norm_a = np.linalg.norm(training_encodings[0])
        norm_b = np.linalg.norm(encoding[0])
        cosine_similarity = dot / (norm_a * norm_b)
        # similary superior to 0.999
        assert cosine_similarity > 0.999
