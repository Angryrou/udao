from typing import Any, Dict, List

import numpy as np
import pytest
import torch as th
import torch.nn as nn

from ....data.containers.tabular_container import TabularContainer
from ....data.extractors.tabular_extractor import TabularFeatureExtractor
from ....data.handler.data_processor import DataProcessor
from ....data.preprocessors.base_preprocessor import StaticPreprocessor
from ....data.tests.iterators.dummy_udao_iterator import DummyUdaoIterator
from ....model.utils.utils import set_deterministic_torch
from ....utils.interfaces import UdaoInput
from ....utils.logging import logger
from ... import concepts as co
from ...soo.mogd import MOGD


class SimpleModel1(nn.Module):
    def forward(self, x: UdaoInput) -> th.Tensor:
        return x.features[:, :1]


class SimpleModel2(nn.Module):
    def forward(self, x: UdaoInput) -> th.Tensor:
        return x.features[:, 1:]


@pytest.fixture()
def data_processor() -> DataProcessor:
    class TabularFeaturePreprocessor(StaticPreprocessor):
        def preprocess(self, tabular_feature: TabularContainer) -> TabularContainer:
            tabular_feature.data.loc[:, "v1"] = tabular_feature.data["v1"] / 1
            tabular_feature.data.loc[:, "v2"] = tabular_feature.data["v2"] - 2
            return tabular_feature

        def inverse_transform(
            self, tabular_feature: TabularContainer
        ) -> TabularContainer:
            tabular_feature.data.loc[:, "v1"] = tabular_feature.data["v1"] * 1
            tabular_feature.data.loc[:, "v2"] = tabular_feature.data["v2"] + 2
            return tabular_feature

    return DataProcessor(
        iterator_cls=DummyUdaoIterator,
        feature_extractors={
            "tabular_features": TabularFeatureExtractor(
                columns=["v1", "v2"],
            ),
            "objectives": TabularFeatureExtractor(columns=["objective_input"]),
        },
        feature_preprocessors={"tabular_features": [TabularFeaturePreprocessor()]},
    )


@pytest.fixture()
def mogd() -> MOGD:
    set_deterministic_torch(42)

    params = MOGD.Params(
        learning_rate=0.1,
        max_iters=100,
        patience=30,
        multistart=2,
        objective_stress=10,
    )
    mogd = MOGD(params)

    return mogd


@pytest.fixture()
def data_processor_paper() -> DataProcessor:
    return DataProcessor(
        iterator_cls=DummyUdaoIterator,
        feature_extractors={
            "tabular_features": TabularFeatureExtractor(
                columns=["v1"],
            ),
            "objectives": TabularFeatureExtractor(columns=["objective_input"]),
        },
    )


class PaperModel1(nn.Module):
    def forward(
        self, input_variables: Dict[str, th.Tensor], input_parameters: Any = None
    ) -> th.Tensor:
        return th.reshape(2400 / (input_variables["cores"]), (-1, 1))


class PaperModel2(nn.Module):
    def forward(
        self, input_variables: Dict[str, th.Tensor], input_parameters: Any = None
    ) -> th.Tensor:
        return th.reshape(input_variables["cores"], (-1, 1))


@pytest.fixture()
def paper_mogd() -> MOGD:
    set_deterministic_torch(42)

    params = MOGD.Params(
        learning_rate=0.1,
        max_iters=100,
        patience=30,
        multistart=2,
        objective_stress=10,
    )
    mogd = MOGD(params)

    return mogd


class TestMOGD:
    @pytest.mark.parametrize(
        "gpu, strict_rounding, expected_obj, expected_vars",
        [
            (False, True, 1, {"v1": 1.0, "v2": 3.0}),
            (False, False, 1, {"v1": 1.0, "v2": 3.0}),
            (True, True, 1, {"v1": 1.0, "v2": 3.0}),
            (True, False, 1, {"v1": 1.0, "v2": 3.0}),
        ],
    )
    def test_solve(
        self,
        mogd: MOGD,
        data_processor: DataProcessor,
        gpu: bool,
        strict_rounding: bool,
        expected_obj: float,
        expected_vars: Dict[str, float],
    ) -> None:
        mogd.device = th.device("cuda") if gpu else th.device("cpu")

        if gpu and not th.cuda.is_available():
            pytest.skip("Skip GPU test")
        set_deterministic_torch(0)
        mogd.strict_rounding = strict_rounding

        problem = co.SOProblem(
            data_processor=data_processor,
            objective=co.Objective(
                "obj1",
                minimize=False,
                function=SimpleModel1(),
                lower=0,
                upper=2,
            ),
            variables={"v1": co.FloatVariable(0, 1), "v2": co.IntegerVariable(2, 3)},
            constraints=[
                co.Constraint(
                    lower=0,
                    upper=1,
                    function=SimpleModel2(),
                )
            ],
        )
        optimal_obj, optimal_vars = mogd.solve(problem, seed=0)
        assert optimal_obj is not None
        np.testing.assert_array_almost_equal(optimal_obj, expected_obj, decimal=5)
        assert optimal_vars == expected_vars

    @pytest.mark.parametrize(
        "strict_rounding, variable, batch_size, expected_obj, expected_vars",
        [
            (True, co.IntegerVariable(0, 24), 1, 200, {"cores": 12}),
            # the strict rounding pulls the gradient descent back to integer value
            # at the end of each iteration. So the variable might not be updated
            # properly if the learning rate is not big enough.
            # This can be improved by data normalization or disabling strict_rounding.
            (True, co.IntegerVariable(0, 24), 16, 150, {"cores": 16}),
            (True, co.FloatVariable(0, 24), 1, 148.17662, {"cores": 16.2}),
            (True, co.FloatVariable(0, 24), 16, 148.17662, {"cores": 16.2}),
            (False, co.IntegerVariable(0, 24), 1, 150, {"cores": 16}),
            (False, co.IntegerVariable(0, 24), 16, 150, {"cores": 16}),
            (False, co.FloatVariable(0, 24), 1, 148.17662, {"cores": 16.2}),
            (False, co.FloatVariable(0, 24), 16, 148.17662, {"cores": 16.2}),
        ],
    )
    def test_solve_paper(
        self,
        paper_mogd: MOGD,
        strict_rounding: bool,
        variable: co.Variable,
        batch_size: int,
        expected_obj: float,
        expected_vars: Dict[str, float],
    ) -> None:
        problem = co.SOProblem(
            objective=co.Objective(
                "obj1",
                minimize=True,
                function=PaperModel1(),
                lower=100,
                upper=200,
            ),
            variables={"cores": variable},
            constraints=[
                co.Objective(
                    name="obj2",
                    lower=8,
                    upper=16.2,
                    function=PaperModel2(),
                    minimize=False,
                )
            ],
        )
        paper_mogd.batch_size = batch_size
        paper_mogd.strict_rounding = strict_rounding
        optimal_obj, optimal_vars = paper_mogd.solve(problem, seed=0)
        logger.debug(f"optimal_obj: {optimal_obj}, optimal_vars: {optimal_vars}")
        assert optimal_obj is not None
        np.testing.assert_allclose([optimal_obj], [expected_obj], rtol=1e-3)
        assert optimal_vars is not None
        assert len(optimal_vars) == 1
        np.testing.assert_allclose(
            [optimal_vars["cores"]], [expected_vars["cores"]], rtol=1e-3
        )

    def test_solve_no_constraints(
        self, mogd: MOGD, data_processor: DataProcessor
    ) -> None:
        def objective_function(x: UdaoInput) -> th.Tensor:
            return th.reshape(x.features[:, 0] ** 2 + x.features[:, 1] ** 2, (-1, 1))

        problem = co.SOProblem(
            data_processor=data_processor,
            objective=co.Objective(
                name="obj1",
                minimize=False,
                function=objective_function,
            ),
            variables={"v1": co.FloatVariable(0, 1), "v2": co.IntegerVariable(2, 3)},
            constraints=[],
        )
        optimal_obj, optimal_vars = mogd.solve(problem, seed=0)

        assert optimal_obj is not None
        np.testing.assert_array_equal(optimal_obj, 2)
        assert optimal_vars is not None
        assert optimal_vars == {"v1": 1, "v2": 3}

    def test_get_unprocessed_input_values(
        self, mogd: MOGD, data_processor: DataProcessor
    ) -> None:
        mogd.batch_size = 4

        input_values, _ = mogd._get_unprocessed_input_values(
            numeric_variables={
                "v1": co.FloatVariable(0, 1),
                "v2": co.IntegerVariable(2, 3),
            },
        )
        assert set(input_values.keys()) == {"v1", "v2"}
        assert all(input_values["v1"] <= 1) and all(input_values["v1"] >= 0)
        assert all([v in [2, 3] for v in input_values["v2"]])

    def test_get_processed_input_values(
        self, mogd: MOGD, data_processor: DataProcessor
    ) -> None:
        mogd.batch_size = 4
        (
            input_values,
            input_shape,
            make_tabular_container,
        ) = mogd._get_processed_input_values(
            data_processor=data_processor,
            numeric_variables={
                "v1": co.FloatVariable(0, 1),
                "v2": co.IntegerVariable(2, 3),
            },
        )
        assert input_values.features.shape == (4, 2)
        assert th.all(input_values.features <= 1) and th.all(input_values.features >= 0)
        assert input_shape.output_names == ["objective_input"]
        assert input_shape.feature_names == ["v1", "v2"]
        container = make_tabular_container(input_values.features)
        np.testing.assert_equal(
            container.data["v1"].values, input_values.features[:, 0].cpu().numpy()
        )
        np.testing.assert_equal(
            container.data["v2"].values, input_values.features[:, 1].cpu().numpy()
        )

    def test_get_processed_input_bounds(
        self, mogd: MOGD, data_processor: DataProcessor
    ) -> None:
        input_lower, input_upper = mogd._get_processed_input_bounds(
            data_processor=data_processor,
            numeric_variables={
                "v1": co.FloatVariable(0, 1),
                "v2": co.IntegerVariable(2, 3),
            },
        )
        assert th.equal(input_lower.features[0].cpu(), th.tensor([0, 0]).cpu())
        assert th.equal(input_upper.features[0].cpu(), th.tensor([1, 1]).cpu())

    def test_get_unprocessed_input_bounds(
        self, mogd: MOGD, data_processor: DataProcessor
    ) -> None:
        data_processor.feature_processors = {}
        input_lower, input_upper = mogd._get_unprocessed_input_bounds(
            numeric_variables={
                "v1": co.FloatVariable(0, 1),
                "v2": co.IntegerVariable(2, 3),
            },
        )
        assert input_lower == {"v1": 0, "v2": 2}
        assert input_upper == {"v1": 1, "v2": 3}

    @pytest.mark.parametrize(
        "objective_values, expected_loss",
        [
            # 0.5 /2 (normalized) * direction (-1 for max) = -0.25
            (th.tensor([0.5]), th.tensor([-0.25])),
            # (-0.2 / 2 - 0.5)**2 + stress (0.1) = 0.46
            (th.tensor([-0.2]), th.tensor([0.46])),
            (th.tensor([[0.5], [0.3], [-0.2]]), th.tensor([[-0.25], [-0.15], [0.46]])),
        ],
    )
    def test__objective_loss_with_bounds(
        self,
        mogd: MOGD,
        data_processor: DataProcessor,
        objective_values: th.Tensor,
        expected_loss: th.Tensor,
    ) -> None:
        objective = co.Objective(
            "obj1",
            minimize=False,
            function=SimpleModel1(),
            lower=0,
            upper=2,
        )
        mogd.objective_stress = 0.1
        loss = mogd.objective_loss(
            objective_values.to(mogd.device), objective.to(mogd.device)  # type: ignore
        )
        # 0.5 /2 (normalized) * direction (-1 for max) = -0.25
        assert th.equal(loss.cpu(), expected_loss.cpu())

    @pytest.mark.parametrize(
        "objective_values, expected_loss",
        [
            # direction * 0.5**2
            (th.tensor([0.5]), th.tensor([-0.25])),
            # direction * (-O.2)**2
            (th.tensor([-0.2]), th.tensor([0.04])),
            (th.tensor([[0.5], [0.3], [-0.2]]), th.tensor([[-0.25], [-0.09], [0.04]])),
        ],
    )
    def test__objective_loss_without_bounds(
        self,
        mogd: MOGD,
        data_processor: DataProcessor,
        objective_values: th.Tensor,
        expected_loss: th.Tensor,
    ) -> None:
        objective = co.Objective(
            "obj1",
            minimize=False,
            function=SimpleModel1(),
        )
        loss = mogd.objective_loss(
            objective_values.to(mogd.device), objective.to(mogd.device)  # type: ignore
        )
        # 0.5 /2 (normalized) * direction (-1 for max) = -0.25
        assert th.allclose(loss.cpu(), expected_loss.cpu())

    @pytest.mark.parametrize(
        "constraint_values, expected_loss",
        [
            (
                [th.tensor([1.1]), th.tensor([1.1]), th.tensor([3.5])],
                # 0.6**2 + 10+ 0 + 0.5**2 + 1000
                th.tensor([1010.61]),
            ),
        ],
    )
    def test_constraints_loss(
        self,
        mogd: MOGD,
        data_processor: DataProcessor,
        constraint_values: List[th.Tensor],
        expected_loss: th.Tensor,
    ) -> None:
        mogd.constraint_stress = 1000
        constraints = [
            co.Objective(
                name="objA",
                minimize=False,
                lower=0,
                upper=1,
                function=SimpleModel1(),
            ),
            co.Constraint(
                lower=0,
                upper=2,
                function=SimpleModel2(),
            ),
            co.Constraint(
                upper=3,
                function=SimpleModel2(),
            ),
        ]
        loss = mogd.constraints_loss(
            [c.to(mogd.device) for c in constraint_values],
            [c.to(mogd.device) for c in constraints],  # type: ignore
        )
        assert th.allclose(loss.cpu(), expected_loss.cpu())

    def test_get_meshed_categorical_variables(self, mogd: MOGD) -> None:
        variables = {
            "v1": co.IntegerVariable(2, 3),
            "v2": co.EnumVariable([4, 5]),
            "v3": co.EnumVariable([10, 20]),
        }
        meshed_variables = mogd.get_meshed_categorical_vars(variables=variables)
        assert meshed_variables is not None
        np.testing.assert_array_equal(
            meshed_variables, [[4.0, 10.0], [5.0, 10.0], [4.0, 20.0], [5.0, 20.0]]
        )
