import numpy as np

def is_valid_value(el):
    return ~(np.isnan(el) & ~np.isinf(el))

# Test the function
print(is_valid_value(5))       # True
print(is_valid_value(np.nan))  # False
print(is_valid_value(np.inf))  # True
print(is_valid_value(-np.inf)) # True
