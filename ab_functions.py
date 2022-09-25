from typing import List, Tuple, Sequence
from dataclasses import dataclass
from scipy.stats import (ttest_ind, mannwhitneyu, poisson)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['axes.titlepad'] = 10
sns.set(rc={'figure.figsize':(11.7,28.27)})

@dataclass
class TestResultList:
    """
    ttest_pvalues: T-test P value List
    mannwhit_pvalues: Mann Whitneyu P value List
    test_type: Test description (title)
    """
    ttest_pvalues: Sequence[float]
    mannwhit_pvalues: Sequence[float]
    test_type: List[str]

@dataclass
class TestResult:
    """
    tt_pvalue: T-test P value
    mw_pvalue: Mann Whitneyu P value
    ctr1: CTR1 array
    ctr2: CTR2 array
    """
    tt_pvalue: float
    mw_pvalue: float
    ctr1: Sequence[float]
    ctr2: Sequence[float]

def run_tests(metric1: Sequence, metric2: Sequence) -> Tuple[float, float]:
    """Run tests and return P values for 2 tests: T-test & Mann Whitneyu test

    Args:
        metric1 (Sequence): Metric for 1st group
        metric2 (Sequence): Metric for 2nd group

    Returns:
        Tuple[float, float]: Pvalue for T-test, Pvalue for MW test
    """
    _, ttest_pvalue = ttest_ind(metric1, metric2, equal_var=False)
    _, mw_pvalue = mannwhitneyu(metric1, metric2)
    return (ttest_pvalue, mw_pvalue)

def standard_tests(data:pd.DataFrame, first_group: int, second_group: int) -> TestResult:
    ctr1 = data[data['exp_group']==first_group]['ctr'].values
    ctr2 = data[data['exp_group']==second_group]['ctr'].values

    ttest_pvalue, mw_pvalue = run_tests(ctr1, ctr2)
    res = TestResult(tt_pvalue=ttest_pvalue, mw_pvalue=mw_pvalue, ctr1=ctr1, ctr2=ctr2)
    return res

def get_smoothed_ctr(likes:pd.Series, views:pd.Series, global_ctr: float, alpha: float) -> pd.Series:
    return (likes + alpha * global_ctr) / (views * alpha)

def smoothed_ctr_test(data:pd.DataFrame, first_group: int, second_group: int, alpha: int = 5) -> TestResult:
    global_ctr1 = data[data.exp_group==first_group].likes.sum() / data[data.exp_group==first_group].views.sum()
    global_ctr2 = data[data.exp_group==second_group].likes.sum() / data[data.exp_group==second_group].views.sum()

    ctr1 = data[data.exp_group==first_group].\
                    apply(lambda x: get_smoothed_ctr(x['likes'], x['views'], global_ctr1, alpha), axis=1).values
    ctr2 = data[data.exp_group==second_group].\
                    apply(lambda x: get_smoothed_ctr(x['likes'], x['views'], global_ctr2, alpha), axis=1).values

    ttest_pvalue, mw_pvalue = run_tests(ctr1, ctr2)
    res = TestResult(tt_pvalue=ttest_pvalue, mw_pvalue=mw_pvalue, ctr1=ctr1, ctr2=ctr2)
    return res

def many_views_tests(data:pd.DataFrame, first_group: int, second_group: int, thresh: int = 100) -> TestResult:
    ctr1 = data[(data.exp_group==first_group) & (data.views>=thresh)].ctr.values
    ctr2 = data[(data.exp_group==second_group) & (data.views>=thresh)].ctr.values
    ttest_pvalue, mw_pvalue = run_tests(ctr1, ctr2)
    res = TestResult(tt_pvalue=ttest_pvalue, mw_pvalue=mw_pvalue, ctr1=ctr1, ctr2=ctr2)
    return res

def bootstrap_split(data: pd.DataFrame, 
                    first_group: int, 
                    second_group: int, 
                    n_bootstrap:int = 10000) -> Tuple[List[float], List[float]]:
    likes1 = data[data.exp_group == first_group].likes.values
    views1 = data[data.exp_group == first_group].views.values
    likes2 = data[data.exp_group == second_group].likes.values
    views2 = data[data.exp_group == second_group].views.values

    poison_bootstrap1 = poisson(1).rvs((n_bootstrap, likes1.shape[0])).astype(np.int64)
    poison_bootstrap2 = poisson(1).rvs((n_bootstrap, likes2.shape[0])).astype(np.int64)

    global_ctr1 = (poison_bootstrap1 * likes1).sum(axis=1) / (poison_bootstrap1 * views1).sum(axis=1)
    global_ctr2 = (poison_bootstrap2 * likes2).sum(axis=1) / (poison_bootstrap2 * views2).sum(axis=1)

    return (global_ctr1, global_ctr2)

def bootstrap_tests(data: pd.DataFrame, first_group: int, second_group: int, n_bootstrap:int = 2000) -> TestResult:
    ctr1, ctr2 = bootstrap_split(data, first_group, second_group, n_bootstrap=n_bootstrap)
    ttest_pvalue, mw_pvalue = run_tests(ctr1, ctr2)
    res = TestResult(tt_pvalue=ttest_pvalue, mw_pvalue=mw_pvalue, ctr1=ctr1, ctr2=ctr2)
    return res

def bucket_tests(bucket_data:pd.DataFrame, first_group: int, second_group: int) -> TestResult:
    ctr1 = bucket_data[bucket_data['exp_group']==first_group].bucket_ctr.values
    ctr2 = bucket_data[bucket_data['exp_group']==second_group].bucket_ctr.values
    ttest_pvalue, mw_pvalue = run_tests(ctr1, ctr2)
    res = TestResult(tt_pvalue=ttest_pvalue, mw_pvalue=mw_pvalue, ctr1=ctr1, ctr2=ctr2)
    return res

def bucket_quantile_tests(bucket_data:pd.DataFrame, first_group: int, second_group: int) -> TestResult:
    ctr1 = bucket_data[bucket_data['exp_group']==first_group].ctr9.values
    ctr2 = bucket_data[bucket_data['exp_group']==second_group].ctr9.values
    ttest_pvalue, mw_pvalue = run_tests(ctr1, ctr2)
    res = TestResult(tt_pvalue=ttest_pvalue, mw_pvalue=mw_pvalue, ctr1=ctr1, ctr2=ctr2)
    return res

def form_dataframe(results: TestResultList, names: List[str]) -> pd.DataFrame:
    res_df = {names[0]: results.ttest_pvalues, names[1]: results.mannwhit_pvalues}
    res_df = pd.DataFrame(res_df, index=results.test_type)
    return res_df

def plot_graph(plots: dict):
    fig, ax = plt.subplots(nrows=len(plots.keys()))
    for i, title in enumerate(plots.keys()):
        for hist, color in zip([plots[title][0], plots[title][1]], ['r', 'b']):
            sns.histplot(hist,
                color=color,
                bins=30,
                alpha=0.5,
                ax=ax[i],
                legend=True,
                kde=False).set_title(title)
    fig.tight_layout()
    plt.show()
    return

def run_ab_test_ctr(data:pd.DataFrame,
                    first_group: int,
                    second_group: int,
                    use_standard_tests: bool=True,
                    use_smooth_ctr: bool=True,
                    use_many_views: bool = True,
                    use_bootstrap: bool=True,
                    use_bucket_split: bool=True,
                    bucket_data: pd.DataFrame=None,
                    show_plots: bool = True) -> pd.DataFrame:
    """Run A/B testing on specified two groups. Return Pandas DataFrame with p values for:
    1. TTest
    2. Mann Whitneyu test
    3. TTest for smoothed CTR
    4. Mann Whitneyu test for smoothed CTR
    5. Bootstrap
    6. Bucket split tests

    Args:
        data (pd.DataFrame): Data with metric to test
        first_group (int): Number of the first group
        second_group (int): Number of the second group
        use_smooth_ctr (bool, optional): Use smoothed metric. Defaults to True.

    Returns:
        pd.DataFrame: Return pvalue list for each type of test
    """
    results = TestResultList(
        ttest_pvalues=[], 
        mannwhit_pvalues=[],
        test_type=[]
    )

    plots = {}

    if use_standard_tests:
        print("[INFO] Run standard tests: T-test & Mann Whitneyu")
        result = standard_tests(data, first_group, second_group)
        results.ttest_pvalues.append(result.tt_pvalue)
        results.mannwhit_pvalues.append(result.mw_pvalue)
        results.test_type.append("Standard tests")
        plots["Standard tests"] = (result.ctr1, result.ctr2)
        print("[INFO] Complete")

    if use_smooth_ctr:
        print("[INFO] Run standard tests on Smoothed CTR metric:")
        result = smoothed_ctr_test(data, first_group, second_group, alpha=5)
        results.ttest_pvalues.append(result.tt_pvalue)
        results.mannwhit_pvalues.append(result.mw_pvalue)
        results.test_type.append("Smoothed CTR tests")
        plots["Smoothed CTR tests"] = (result.ctr1, result.ctr2)
        print("[INFO] Complete")

    if use_many_views:
        print("[INFO] Run tests with many views:")
        result = many_views_tests(data, first_group, second_group, thresh=50)
        results.ttest_pvalues.append(result.tt_pvalue)
        results.mannwhit_pvalues.append(result.mw_pvalue)
        results.test_type.append("CTR with many views tests")
        plots["CTR with many views tests"] = (result.ctr1, result.ctr2)
        print("[INFO] Complete")
    
    if use_bootstrap:
        print("[INFO] Run tests on bootstraped CTR metric:")
        result = bootstrap_tests(data, first_group, second_group, n_bootstrap=10000)
        results.ttest_pvalues.append(result.tt_pvalue)
        results.mannwhit_pvalues.append(result.mw_pvalue)
        results.test_type.append("Bootstrap")
        plots["Bootstrap"] = (result.ctr1, result.ctr2)
        print("[INFO] Complete")

    if use_bucket_split:
        print("[INFO] Run tests with bucket split:")
        result = bucket_tests(bucket_data, first_group, second_group)
        results.ttest_pvalues.append(result.tt_pvalue)
        results.mannwhit_pvalues.append(result.mw_pvalue)
        results.test_type.append("Bucket CTR")
        plots["Bucket CTR"] = (result.ctr1, result.ctr2)
        print("[INFO] Complete")

        print("[INFO] Run tests with bucket quantile split:")
        result = bucket_quantile_tests(bucket_data, first_group, second_group)
        results.ttest_pvalues.append(result.tt_pvalue)
        results.mannwhit_pvalues.append(result.mw_pvalue)
        results.test_type.append("Bucket Quantile CTR")
        plots["Bucket Quantile CTR"] = (result.ctr1, result.ctr2)
        print("[INFO] Complete")

    results = form_dataframe(results, names=["T-test P value", "Mann Whitneyu P value"])

    if show_plots:
        plot_graph(plots=plots)

    return results