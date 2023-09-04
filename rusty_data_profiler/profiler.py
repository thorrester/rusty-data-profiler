from typing import List
from numpy.typing import NDArray

from .rusty_data_profiler import parse_array, FeatureBin


class Profiler:
    def __init__(self, feature_names, num_bins):
        """
        Class used to generate a data profile from a pandas dataframe,
        polars dataframe or numpy array.

        Args:
            feature_names:
                List of feature names.
            num_bins:
                Number of bins to use for the histogram.
        """
        self.feature_names = feature_names
        self.num_bins = num_bins

    def parse(self, array: NDArray) -> List[FeatureBin]:
        return parse_array(
            array=array,
            feature_names=self.feature_names,
            num_bins=self.num_bins,
        )
