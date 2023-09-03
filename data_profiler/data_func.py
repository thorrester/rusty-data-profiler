from rusty_data_profiler import parse_array
from numpy.typing import NDArray


class Parse2DArray:
    def __init__(self, feature_names, num_bins):
        self.feature_names = feature_names
        self.num_bins = num_bins

    def parse(self, array: NDArray):
        return parse_array(
            array=array,
            feature_name=self.feature_names,
            num_bins=self.num_bins,
        )
