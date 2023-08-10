from rusty_data_profiler import add, sum_as_string


def test_add():
    assert add(1, 2) == 3


def test_sum_as_string():
    assert sum_as_string(1, 2) == "3"
