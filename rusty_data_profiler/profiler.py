from typing import List, Union, Optional
from numpy.typing import NDArray
import polars as pl
import pandas as pd

from .rusty_data_profiler import parse_array, FeatureBin


class Profiler:
    def __init__(
        self,
        data: Union[pl.DataFrame, pd.DataFrame, NDArray],
    ):
        """
        Class used to generate a data profile from a pandas dataframe,
        polars dataframe or numpy array.

        Args:
            feature_names:
                List of feature names.
            num_bins:
                Number of bins to use for the histogram.
        """

        self.types = self._parse_types(data)

    def _parse_types(self, data: Union[pl.DataFrame, pd.DataFrame, NDArray]) -> NDArray:
        if isinstance(data, pl.DataFrame):
            return data.schema
        elif isinstance(data, pd.DataFrame):
            return data.dtypes.apply(lambda x: x.name).to_dict()
        elif isinstance(data, NDArray):
            return {"dtype": data.dtype}
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def parse(self, array: NDArray) -> List[FeatureBin]:
        return parse_array(
            array=array,
            feature_names=self.feature_names,
            num_bins=self.num_bins,
        )
