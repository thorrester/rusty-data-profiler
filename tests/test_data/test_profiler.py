from typing import Union
from rusty_data_profiler import Profiler
import numpy as np
from numpy.typing import NDArray
import polars as pl
import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture


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
