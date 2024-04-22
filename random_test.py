import numpy as np


def is_valid_value(el):
    return ~(np.isnan(el) & ~np.isinf(el))


print(is_valid_value(5))
print(is_valid_value(np.nan))
print(is_valid_value(np.inf))
print(is_valid_value(-np.inf))

