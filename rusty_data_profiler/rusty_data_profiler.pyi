from typing import List, Optional, Dict
from numpy.typing import NDArray

class Bin:
    """Class that holds feature information related to the histogram bins"""

    def __init__(self, bins: List[float], bin_counts: List[int]) -> None:
        """Instantiate a FeatureBin object.

        Args:
            bins:
                List of bin edges.
            bin_counts:
                List of bin counts.
        """

        ...

class Distinct:
    """Class that holds feature information related to the number of distinct values"""

    def __init__(self, count: int, percent: float) -> None:
        """Instantiate a Distinct object.

        Args:
            count:
                Number of distinct values.
            percent:
                Percentage of distinct values.
        """

        ...

class Infinity:
    """Class that holds feature information related to the number of infinite values"""

    def __init__(self, count: int, percent: float) -> None:
        """Instantiate a Infinity object.

        Args:
            count:
                Number of distinct values.
            percent:
                Percentage of distinct values.
        """

        ...

class Missing:
    """Class that holds feature information related to the number of missing values"""

    def __init__(self, count: int, percent: float) -> None:
        """Instantiate a Missing object.

        Args:
            count:
                Number of distinct values.
            percent:
                Percentage of distinct values.
        """

        ...

class Stats:
    """Class that holds feature statistics"""

    def __init__(
        self,
        median: float,
        mean: float,
        standard_dev: float,
        min: float,
        max: float,
        distinct: Distinct,
        infinity: Infinity,
        missing: Optional[Missing] = None,
    ):
        """Instantiate a Stats object.

        Args:
            median:
                Median value.
            mean:
                Mean value.
            standard_dev:
                Standard deviation.
            min:
                Minimum value.
            max:
                Maximum value.
            distinct:
                `Distinct`
            infinity:
                `Infinity`
            missing:
               Optional `Missing`
        """
    ...

class FeatureStat:
    """Class that holds feature statistics"""

    def __init__(self, name: str, bins: Bin, stats: Stats):
        """Instantiate a FeatureStat object.

        Args:
            name:
                Name of the feature.
            bins:
                `Bin`
            stats:
                `Stats`
        """
        ...

def parse_array(
    feature_names: List[str],
    array: NDArray,
    bins: List[float],
    num_bins: Optional[int],
) -> Dict[str, FeatureStat]:
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
