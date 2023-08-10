from rusty_data_profiler import compute_mean, add, sum_as_string


def test_add():
    assert add(1, 2) == 3


def test_sum_as_string():
    assert sum_as_string(1, 2) == "3"


def test_mean():
    import timeit, functools
    import numpy as np

    random_array = np.random.rand(10000000)

    t = timeit.Timer(functools.partial(np.mean, random_array))
    print(f"numpy: {t.timeit(1)}")

    t = timeit.Timer(functools.partial(compute_mean, random_array))
    print(f"rust: {t.timeit(1)}")

    a

    assert np.round(compute_mean(random_array), 1) == 0.5
