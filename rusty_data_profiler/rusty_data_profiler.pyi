from typing import List, Optional
from numpy.typing import NDArray

class FeatureBin:
    """Class that holds feature information related to the histogram bins."""

    def __init__(self, name: str, bins: List[float], bin_counts: List[int]) -> None: ...

def parse_array(
    feature_names: List[str],
    array: NDArray,
    bins: List[float],
    num_bins: Optional[int],
) -> List[FeatureBin]:
    """Parse a numpy array and return a list of FeatureBin objects.

    Args:
        feature_names:
            List of feature names.
        array:
            Numpy array to parse.
        bins:
            List of bin edges to use for the histogram.
        num_bins:
            Number of bins to use for the histogram.

    Returns:
        List of `FeatureBin` objects.
    """
    ...
