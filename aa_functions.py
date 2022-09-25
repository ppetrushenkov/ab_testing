from scipy.stats import ttest_ind
from typing import Union
from hashlib import md5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_hash_group(id: Union[str, int], salt:Union[str, int]="experiment", num_groups:int=5) -> int:
    """
    Return an int number in the range from 0 to num_groups.
    id: User ID
    salt: additional text / number. May describe the current experiment.
    num_groups: Number of desired groups
    """
    combined_id = str(id) + "_" + str(salt)
    hashed_id = md5(combined_id.encode('ascii')).hexdigest()
    hashed_int = int(hashed_id, 16)
    return hashed_int % num_groups

def get_percent_of_data_lower_thresh(arr:Union[np.array, list], thresh: float=0.05) -> float:
    arr = np.array(arr)
    lower_thresh = arr[arr<=thresh]
    return lower_thresh.shape[0] * 100 / arr.shape[0]

def _run_many_tests(group1:pd.Series, group2:pd.Series, n_tests: int=10000, n_samples: int=500) -> float:
    ttest_pvalue_array = []
    for _ in range(n_tests):
        chunk1 = group1.sample(n_samples)
        chunk2 = group2.sample(n_samples)
        _, p_value_chunk = ttest_ind(chunk1, chunk2, equal_var=False)
        ttest_pvalue_array.append(p_value_chunk)
    
    ttest_pvalue_array = np.array(ttest_pvalue_array)
    return ttest_pvalue_array

def run_aatest(
    group1:Union[np.array, pd.DataFrame], 
    group2:Union[np.array, pd.DataFrame], 
    n_tests:int=10000, n_samples:int=500,
    show_distribution:bool=True) -> pd.DataFrame:
    """Function run A/A test for `n_tests` times with `n_samples` chunk size.

    Args:
        group1 (Union[np.array, pd.DataFrame]): First group of data
        group2 (Union[np.array, pd.DataFrame]): Second group of data
        n_tests (int, optional): Number of tests. Defaults to 10000.
        n_samples (int, optional): Number of values in each chunk. Defaults to 500.

    Returns:
        pd.DataFrame: Description about tests
    """
    _, tp_value = ttest_ind(group1, group2, equal_var=False)

    ttest_pvalue_array = _run_many_tests(group1, group2, n_tests, n_samples)
    lower_10pct = get_percent_of_data_lower_thresh(ttest_pvalue_array, thresh=0.1)
    lower_5pct = get_percent_of_data_lower_thresh(ttest_pvalue_array, thresh=0.05)
    lower_1pct = get_percent_of_data_lower_thresh(ttest_pvalue_array, thresh=0.01)

    idx_names = [
        "Full TTest Pvalue", 
        "Percent of tests lower 0.1 pvalue thresh",
        "Percent of tests lower 0.05 pvalue thresh",
        "Percent of tests lower 0.01 pvalue thresh"
        ]
    data = np.round([tp_value, lower_10pct, lower_5pct, lower_1pct], 2)
    stat = pd.DataFrame(data=data, index=idx_names, columns=["Result"])
    
    if show_distribution:
        sns.histplot(data=ttest_pvalue_array, bins=20)
        plt.title("P value distribution")
        plt.show()

    return stat

