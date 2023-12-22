import pandas as pd
import torch as th

from ...containers.tabular_container import TabularContainer
from ...iterators.tabular_iterator import TabularIterator


def test_iterator_augmentation() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "id": ["1", "2", "3"]})
    df.set_index("id", inplace=True)
    tabular_container = TabularContainer(data=df)
    iterator = TabularIterator(["1", "2", "3"], tabular_container)
    assert th.equal(iterator[0], th.tensor([1, 4]))
    iterator.set_augmentations([lambda x: x + 1, lambda x: x**2])
    assert th.equal(iterator[0], th.tensor([(1 + 1) ** 2, (4 + 1) ** 2]))
