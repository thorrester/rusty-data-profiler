from typing import Union
from rusty_data_profiler import Profiler
import numpy as np
from numpy.typing import NDArray
import polars as pl
import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture
import time


@pytest.mark.parametrize(
    "test_data",
    [
        lazy_fixture("test_polars_dataframe"),
        lazy_fixture("test_pandas_dataframe"),
        lazy_fixture("test_numpy_array"),
    ],
)
def test_parse_array(
    test_data: Union[pl.DataFrame, pd.DataFrame, NDArray],
) -> None:
    profiler = Profiler(data=test_data)

    if isinstance(test_data, (pl.DataFrame, pd.DataFrame)):
        assert profiler.numeric_features == ["int", "float"]
        assert profiler.categorical_features == ["str"]

    else:
        assert profiler.numeric_features == ["dtype"]
        assert profiler.categorical_features == []


def test_array():
    array = np.random.rand(1_000_000, 20)

    start = time.time()

    medians = np.median(array, axis=0)
    stddev = np.std(array, axis=0)
    means = np.mean(array, axis=0)
    inf = np.isinf(array).sum(axis=0)
    min_ = np.min(array, axis=0)
    max_ = np.max(array, axis=0)

    for col in array.T:
        unique = len(np.unique(np.array([1, 1, 0])))
        hist = np.histogram(col)

    print(f"numpy: {time.time() - start}")

    profiler = Profiler(data=array)
    start = time.time()
    profiler.parse()
    print(f"rust: {time.time() - start}")
    a
