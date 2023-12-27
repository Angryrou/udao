from typing import Any

import pytest
import torch as th
from attr import dataclass

from ...utils.interfaces import UdaoEmbedInput, UdaoEmbedItemShape
from ..embedders import BaseEmbedder
from ..model import FixedEmbeddingUdaoModel, UdaoModel
from ..regressors import BaseRegressor


class DummyEmbedder(BaseEmbedder):
    @dataclass
    class Params(BaseEmbedder.Params):
        output_size: int

    @classmethod
    def from_iterator_shape(
        cls, iterator_shape: UdaoEmbedItemShape, **kwargs: Any
    ) -> "DummyEmbedder":
        return cls(cls.Params(**kwargs))

    def forward(self, input: th.Tensor) -> th.Tensor:
        return input


class DummyRegressor(BaseRegressor):
    def forward(self, embedding: th.Tensor, inst_feat: th.Tensor) -> th.Tensor:
        return embedding + inst_feat


@pytest.fixture
def model() -> UdaoModel:
    iterator_shape = UdaoEmbedItemShape(
        embedding_input_shape=1,
        feature_names=["a"],
        output_names=["b"],
    )
    return UdaoModel.from_config(
        regressor_cls=DummyRegressor,
        embedder_cls=DummyEmbedder,
        iterator_shape=iterator_shape,
        regressor_params={},
        embedder_params={"output_size": 1},
    )


@pytest.fixture
def obj_model(model: UdaoModel) -> FixedEmbeddingUdaoModel:
    return FixedEmbeddingUdaoModel(model, None)


class TestUdaoModel:
    def test_from_config(self, model: UdaoModel) -> None:
        assert model.regressor.input_dim == 2
        assert model.embedder.embedding_size == 1
        assert model.regressor.output_dim == 1

    def test_forward(self, model: UdaoModel) -> None:
        embedding = th.tensor([1.0])
        inst_feat = th.tensor([2.0])
        out = model.forward(
            UdaoEmbedInput(embedding_input=embedding, features=inst_feat)
        )
        assert out == th.tensor([3.0])


class TestFixedEmbeddingUdaoModel:
    def test_init(self, obj_model: FixedEmbeddingUdaoModel) -> None:
        assert obj_model.model.regressor.input_dim == 2
        assert obj_model.model.embedder.embedding_size == 1
        assert obj_model.model.regressor.output_dim == 1
        assert obj_model.embedding is None

    def test_forward(self, obj_model: FixedEmbeddingUdaoModel, mocker: Any) -> None:
        embedding = th.tensor([1.0])
        inst_feat = th.tensor([2.0])

        mocked_embedder = mocker.spy(obj_model.model.embedder, "forward")
        mocked_regressor = mocker.spy(obj_model.model.regressor, "forward")
        out1 = obj_model.forward(
            UdaoEmbedInput(embedding_input=embedding, features=inst_feat)
        )
        assert out1 == th.tensor([3.0])
        assert obj_model.embedding == embedding
        out2 = obj_model.forward(
            UdaoEmbedInput(embedding_input=embedding, features=inst_feat)
        )
        assert out2 == th.tensor([3.0])
        assert mocked_embedder.call_count == 1
        assert mocked_regressor.call_count == 2
