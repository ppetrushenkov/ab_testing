from scipy.stats import ttest_ind
from pandas import DataFrame, Series
from dataclasses import dataclass
from typing import Sequence

import seaborn as sns
import matplotlib.pyplot as plt

@dataclass
class TestResult:
    """
    ttest_pvalues: T-test P value List
    test_type: Test description (title)
    """
    pvalue: float
    control_metric: Series
    threat_metric: Series


def run_linearized_test(data: DataFrame, first_group: int, second_group: int) -> float:
    """Run T-test and return pvalue, using linearized method"""
    control_group = data[data.exp_group==first_group]
    threat_group = data[data.exp_group==second_group]

    global_ctr1 = control_group.likes.sum() / control_group.views.sum()

    linearized_likes1 = control_group.likes - global_ctr1 * control_group.views
    linearized_likes2 = threat_group.likes - global_ctr1 * threat_group.views

    _, pvalue = ttest_ind(linearized_likes1, linearized_likes2, equal_var=False)
    return TestResult(pvalue=pvalue, control_metric=linearized_likes1, threat_metric=linearized_likes2)

def test_compare(data: DataFrame, first_group: int, second_group: int, show_plot: bool = True) -> DataFrame:
    """Run T-test on simple CTR metric and compare results with Linearized likes metric"""
    _, simple_ttest_pvalue = ttest_ind(
        data[data.exp_group==first_group].ctr,
        data[data.exp_group==second_group].ctr,
        equal_var=False
    )
    if show_plot:
        for hist, color in zip([data[data.exp_group==first_group].ctr,
                                data[data.exp_group==second_group].ctr],
                                ['r', 'b']):
            sns.histplot(hist,
                color=color,
                bins=30,
                alpha=0.5,
                legend=True,
                kde=False).set_title("Standard CTR")
        plt.show()

    linearized_results = run_linearized_test(data, first_group, second_group)

    if show_plot:
        for hist, color in zip([linearized_results.control_metric,
                                linearized_results.threat_metric],
                                ['r', 'b']):
            sns.histplot(hist,
                color=color,
                bins=30,
                alpha=0.5,
                legend=True,
                kde=False).set_title("Linearized likes")
        plt.show()


    return DataFrame({"Simple T-test": [simple_ttest_pvalue],
                    "Linearized T-test": [linearized_results.pvalue]},
                    index=["Comparing"])