from typing import List, Union, Optional, Dict
from numpy.typing import NDArray
from functools import cached_property
import polars as pl
import pandas as pd
import numpy as np

from .rusty_data_profiler import parse_array, FeatureStat


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
        self.array = self._convert_data_to_array(data)
        self.types = self._parse_types(data)

    @cached_property
    def feature_names(self) -> List[str]:
        if isinstance(self.array, np.ndarray):
            return [f"feature_{i}" for i in range(self.array.shape[1])]
        return list(self.types.keys())

    @cached_property
    def numeric_features(self) -> List[str]:
        """Retrieve and cache numeric features"""
        return [
            feature_name
            for feature_name, feature_type in self.types.items()
            if any([value in feature_type.lower() for value in ["int", "float"]])
        ]

    @cached_property
    def categorical_features(self) -> List[Optional[str]]:
        """Retrieve and cache categorical features"""
        return [feature_name for feature_name, _ in self.types.items() if feature_name not in self.numeric_features]

    def _convert_data_to_array(self, data: Union[pl.DataFrame, pd.DataFrame, NDArray]) -> NDArray:
        if isinstance(data, pl.DataFrame):
            return data.to_numpy()
        elif isinstance(data, pd.DataFrame):
            return data.to_numpy()
        return data

    def _parse_types(self, data: Union[pl.DataFrame, pd.DataFrame, NDArray]) -> Dict[str, str]:
        """Parse the data types of the input data

        Args:
            data:
                Input data to parse

        Returns:
            Dictionary of feature names and their data types
        """

        if isinstance(data, pl.DataFrame):
            return {key: str(value) for key, value in data.schema.items()}
        elif isinstance(data, pd.DataFrame):
            return data.dtypes.apply(lambda x: x.name).to_dict()
        elif isinstance(data, np.ndarray):
            return {"dtype": str(data.dtype)}
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def parse(self) -> List[FeatureStat]:
        return parse_array(array=self.array, feature_names=self.feature_names)
