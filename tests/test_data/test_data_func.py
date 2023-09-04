from rusty_data_profiler import Parse2DArray
import numpy as np


def test_parse_array():
    array = np.random.rand(1_000, 20)
    feature_names = [f"feature_{i}" for i in range(20)]
    results = Parse2DArray(feature_names=feature_names, num_bins=20).parse(array=array)

    assert len(results[0].bins) == 20
    assert len(results) == 20
