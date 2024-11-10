import pytest
import numpy as np

from datetime import datetime, timedelta
from ..squigglepy.utils import (
    _core_cuts,
    _process_weights_values,
    _process_discrete_weights_values,
    event_occurs,
    event_happens,
    event,
    _weighted_percentile,
    get_percentiles,
    get_log_percentiles,
    get_mean_and_ci,
    get_median_and_ci,
    geomean,
    p_to_odds,
    odds_to_p,
    geomean_odds,
    laplace,
    growth_rate_to_doubling_time,
    doubling_time_to_growth_rate,
    roll_die,
    flip_coin,
    kelly,
    full_kelly,
    half_kelly,
    quarter_kelly,
    one_in,
    extremize,
    sharpe_ratio,
    normalize,
)
from ..squigglepy.rng import set_seed
from ..squigglepy.distributions import bernoulli, beta, norm, dist_round, const


def test_process_weights_values_simple_case():
    test = _process_weights_values(weights=[0.1, 0.9], values=[2, 3])
    expected = ([0.1, 0.9], [2, 3])
    assert test == expected


def test_process_weights_values_numpy_arrays():
    test = _process_weights_values(weights=np.array([0.1, 0.9]), values=np.array([2, 3]))
    expected = ([0.1, 0.9], [2, 3])
    assert test == expected


def test_process_weights_values_length_one():
    test = _process_weights_values(weights=[1], values=[2])
    expected = ([1], [2])
    assert test == expected


def test_process_weights_values_alt_format():
    test = _process_weights_values(values=[[0.1, 2], [0.2, 3], [0.3, 4], [0.4, 5]])
    expected = ([0.1, 0.2, 0.3, 0.4], [2, 3, 4, 5])
    assert test == expected


def test_process_weights_values_alt2_format():
    test = _process_weights_values(values={2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4})
    expected = ([0.1, 0.2, 0.3, 0.4], [2, 3, 4, 5])
    assert test == expected


def test_process_weights_values_dict_error():
    with pytest.raises(ValueError) as execinfo:
        _process_weights_values(
            weights=[0.1, 0.2, 0.3, 0.4], values={2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4}
        )
    assert "cannot pass dict and weights separately" in str(execinfo.value)


def test_process_weights_values_weight_inference():
    test = _process_weights_values(weights=[0.9], values=[2, 3])
    expected = ([0.9, 0.1], [2, 3])
    test[0][1] = round(test[0][1], 1)  # fix floating point errors
    assert test == expected


def test_process_weights_values_weight_inference_not_list():
    test = _process_weights_values(weights=0.9, values=[2, 3])
    expected = ([0.9, 0.1], [2, 3])
    test[0][1] = round(test[0][1], 1)  # fix floating point errors
    assert test == expected


def test_process_weights_values_weight_inference_no_weights():
    test = _process_weights_values(values=[2, 3])
    expected = ([0.5, 0.5], [2, 3])
    assert test == expected


def test_process_weights_values_weight_inference_relative_weights():
    test = _process_weights_values(values=[2, 3], relative_weights=[1, 3])
    expected = ([0.25, 0.75], [2, 3])
    assert test == expected


def test_process_weights_values_weight_inference_no_weights_len4():
    test = _process_weights_values(values=[2, 3, 4, 5])
    expected = ([0.25, 0.25, 0.25, 0.25], [2, 3, 4, 5])
    assert test == expected


def test_process_weights_values_weights_must_be_list_error():
    with pytest.raises(ValueError) as excinfo:
        _process_weights_values(weights="error", values=[2, 3])
    assert "passed weights must be an iterable" in str(excinfo.value)


def test_process_weights_values_values_must_be_list_error():
    with pytest.raises(ValueError) as excinfo:
        _process_weights_values(weights=[0.1, 0.9], values="error")
    assert "passed values must be an iterable" in str(excinfo.value)


def test_process_weights_values_weights_must_sum_to_1_error():
    with pytest.raises(ValueError) as excinfo:
        _process_weights_values(weights=[0.2, 0.9], values=[2, 3])
    assert "weights don't sum to 1 - they sum to 1.1" in str(excinfo.value)


def test_process_weights_values_length_mismatch_error():
    with pytest.raises(ValueError) as excinfo:
        _process_weights_values(weights=[0.1, 0.9], values=[2, 3, 4])
    assert "weights and values not same length" in str(excinfo.value)


def test_process_weights_values_negative_weights():
    with pytest.raises(ValueError) as excinfo:
        _process_weights_values(weights=[-0.1, 0.2, 0.9], values=[2, 3, 4])
    assert "weight cannot be negative" in str(excinfo.value)


def test_process_weights_values_remove_zero_weights():
    test = _process_weights_values(weights=[0, 0.3, 0, 0.7, 0], values=[1, 2, 3, 4, 5])
    expected = ([0.3, 0.7], [2, 4])
    assert test == expected


def test_process_weights_values_handle_none():
    test = _process_weights_values(weights=None, values=[1, None, 3, 4, 5])
    expected = ([0.2, 0.2, 0.2, 0.2, 0.2], [1, None, 3, 4, 5])
    assert test == expected


def test_process_weights_values_can_drop_none():
    test = _process_weights_values(weights=None, values=[1, None, 3, 4, 5], drop_na=True)
    expected = ([0.25, 0.25, 0.25, 0.25], [1, 3, 4, 5])
    assert test == expected


def test_process_weights_values_attempt_drop_none_with_weights_error():
    with pytest.raises(ValueError) as execinfo:
        _process_weights_values(
            weights=[0.2, 0.2, 0.2, 0.2, 0.2], values=[1, None, 3, 4, 5], drop_na=True
        )
    assert "cannot drop NA and process weights" in str(execinfo.value)
    with pytest.raises(ValueError) as execinfo:
        _process_weights_values(
            relative_weights=[1, 1, 1, 1, 1], values=[1, None, 3, 4, 5], drop_na=True
        )
    assert "cannot drop NA and process weights" in str(execinfo.value)


def test_process_discrete_weights_values_simple_case():
    test = _process_discrete_weights_values([[0.1, 2], [0.9, 3]])
    expected = ([0.1, 0.9], [2, 3])
    assert test == expected
    test = _process_discrete_weights_values({2: 0.1, 3: 0.9})
    expected = ([0.1, 0.9], [2, 3])
    assert test == expected


def test_process_discrete_weights_values_compress():
    items = [round((x % 10) / 10, 1) for x in range(1000)]
    test = _process_discrete_weights_values(items)
    expected = (
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    assert test == expected


def test_normalize():
    assert normalize([0.1, 0.4]) == [0.2, 0.8]


def test_event_occurs():
    set_seed(42)
    assert event_occurs(0.9)
    set_seed(42)
    assert not event_occurs(0.1)


def test_event_occurs_can_handle_distributions():
    set_seed(42)
    assert event_occurs(bernoulli(0.9))


def test_event_occurs_can_handle_distributions2():
    set_seed(42)
    assert event_occurs(beta(10, 1))


def test_event_occurs_can_handle_distributions_callable():
    def get_p():
        return 0.9

    assert event_occurs(get_p)


def test_event_happens():
    set_seed(42)
    assert event_happens(0.9)
    set_seed(42)
    assert not event_happens(0.1)


def test_event():
    set_seed(42)
    assert event(0.9)
    set_seed(42)
    assert not event(0.1)


def test_one_in():
    assert one_in(0.1) == "1 in 10"
    assert one_in(0.02) == "1 in 50"
    assert one_in(0.00002) == "1 in 50,000"


def test_one_in_w_rounding():
    assert one_in(0.1415) == "1 in 7"
    assert one_in(0.1415, digits=1) == "1 in 7.1"
    assert one_in(0.1415, digits=2) == "1 in 7.07"
    assert one_in(0.1415, digits=3) == "1 in 7.067"


def test_one_in_not_verbose():
    assert one_in(0.1415, digits=3, verbose=False) == 7.067


def test_get_percentiles():
    test = get_percentiles(range(1, 901))
    expected = {
        1: 9.99,
        5: 45.95,
        10: 90.9,
        20: 180.8,
        30: 270.7,
        40: 360.6,
        50: 450.5,
        60: 540.4,
        70: 630.3,
        80: 720.2,
        90: 810.1,
        95: 855.05,
        99: 891.01,
    }
    assert test == expected


def test_get_percentiles_change_percentiles():
    test = get_percentiles(range(1, 901), percentiles=[20, 80])
    expected = {20: 180.8, 80: 720.2}
    assert test == expected

    test = get_percentiles(range(1, 901), percentiles=[25, 75])
    expected = {25: 225.75, 75: 675.25}
    assert test == expected


def test_get_percentiles_reverse():
    test = get_percentiles(range(1, 901), percentiles=[20, 80], reverse=True)
    expected = {20: 720.2, 80: 180.8}
    assert test == expected


def test_get_percentiles_digits():
    test = get_percentiles(range(1, 901), percentiles=[25, 75], digits=1)
    expected = {25: 225.8, 75: 675.2}
    assert test == expected


def test_get_percentiles_length_one():
    test = get_percentiles(range(1, 901), percentiles=[25], digits=1)
    assert test == 225.8
    test = get_percentiles(range(1, 901), percentiles=25, digits=1)
    assert test == 225.8


def test_get_percentiles_zero_digits():
    test = get_percentiles(range(1, 901), percentiles=[25, 75], digits=0)
    expected = {25: 226, 75: 675}
    assert test == expected
    assert isinstance(expected[25], int)
    assert isinstance(expected[75], int)


def test_get_percentiles_negative_one_digits():
    test = get_percentiles(range(1, 901), percentiles=[25, 75], digits=-1)
    expected = {25: 230, 75: 680}
    assert test == expected
    assert isinstance(expected[25], int)
    assert isinstance(expected[75], int)


def test_weighted_percentile():
    test = _weighted_percentile(
        range(1, 901), weights=[1 for _ in range(900)], percentiles=[25, 75]
    )
    assert test[0] == 225.5
    assert test[1] == 675.5
    test = _weighted_percentile(range(1, 901), weights=range(900), percentiles=[25, 75])
    assert round(test[0], 1) == 450.7
    assert round(test[1], 1) == 780.0


def test_weighted_percentile_requires_weights_len_equal_data_len():
    with pytest.raises(ValueError) as execinfo:
        _weighted_percentile([1, 2, 3], [1, 2], [1, 2, 3])
    assert "must be of the same length" in str(execinfo.value)


def test_get_percentiles_weighted():
    test = get_percentiles(range(1, 901), weights=range(900), digits=-1)
    expected = {
        1: 90,
        5: 200,
        10: 290,
        20: 400,
        30: 490,
        40: 570,
        50: 640,
        60: 700,
        70: 750,
        80: 810,
        90: 850,
        95: 880,
        99: 900,
    }
    assert test == expected


def test_get_percentiles_with_weights_requires_weights_len_equal_data_len():
    with pytest.raises(ValueError) as execinfo:
        get_percentiles(range(1, 901), weights=range(800), digits=-1)
    assert "must be of the same length" in str(execinfo.value)


def test_get_log_percentiles():
    test = get_log_percentiles([10**x for x in range(1, 10)])
    expected = {
        1: "1.7e+01",
        5: "4.6e+01",
        10: "8.2e+01",
        20: "6.4e+02",
        30: "4.6e+03",
        40: "2.8e+04",
        50: "1.0e+05",
        60: "8.2e+05",
        70: "6.4e+06",
        80: "4.6e+07",
        90: "2.8e+08",
        95: "6.4e+08",
        99: "9.3e+08",
    }
    assert test == expected


def test_get_log_percentiles_change_percentiles():
    test = get_log_percentiles([10**x for x in range(1, 10)], percentiles=[20, 80])
    expected = {20: "6.4e+02", 80: "4.6e+07"}
    assert test == expected


def test_get_log_percentiles_reverse():
    test = get_log_percentiles([10**x for x in range(1, 10)], percentiles=[20, 80], reverse=True)
    expected = {20: "4.6e+07", 80: "6.4e+02"}
    assert test == expected


def test_get_log_percentiles_no_display():
    test = get_log_percentiles([10**x for x in range(1, 10)], percentiles=[20, 80], display=False)
    expected = {20: 2.8, 80: 7.7}
    assert test == expected


def test_get_log_percentiles_zero_digits():
    test = get_log_percentiles(
        [10**x for x in range(1, 10)], percentiles=[20, 80], display=False, digits=0
    )
    expected = {20: 3, 80: 8}
    assert test == expected


def test_get_log_percentiles_length_one():
    test = get_log_percentiles(
        [10**x for x in range(1, 10)], percentiles=[20], display=False, digits=0
    )
    assert test == 3
    test = get_log_percentiles(
        [10**x for x in range(1, 10)], percentiles=20, display=False, digits=0
    )
    assert test == 3


def test_get_log_percentiles_weighted():
    test = get_log_percentiles([10**x for x in range(1, 10)], weights=range(9))
    expected = {
        1: "7.5e+01",
        5: "8.8e+02",
        10: "6.8e+03",
        20: "7.9e+04",
        30: "6.6e+05",
        40: "4.1e+06",
        50: "1.0e+07",
        60: "6.0e+07",
        70: "1.8e+08",
        80: "6.2e+08",
        90: "1.0e+09",
        95: "1.0e+09",
        99: "1.0e+09",
    }
    assert test == expected


def test_get_mean_and_ci():
    test1 = get_mean_and_ci(range(1, 901), digits=1)
    assert test1 == {"mean": 450.5, "ci_low": 46.0, "ci_high": 855.0}
    test2 = get_mean_and_ci([1, 2, 6], digits=1)
    assert test2 == {"mean": 3, "ci_low": 1.1, "ci_high": 5.6}


def test_get_mean_and_80_pct_ci():
    test = get_mean_and_ci(range(1, 901), digits=1, credibility=80)
    assert test == {"mean": 450.5, "ci_low": 90.9, "ci_high": 810.1}


def test_get_mean_and_ci_weighted():
    test1 = get_mean_and_ci(range(1, 901), weights=range(900), digits=1)
    assert test1 == {"mean": 450.5, "ci_low": 202.1, "ci_high": 877.7}


def test_get_median_and_ci():
    test1 = get_median_and_ci(range(1, 901), digits=1)
    assert test1 == {"median": 450.5, "ci_low": 46.0, "ci_high": 855.0}
    test2 = get_median_and_ci([1, 2, 6], digits=1)
    assert test2 == {"median": 2, "ci_low": 1.1, "ci_high": 5.6}


def test_get_median_and_80_pct_ci():
    test = get_median_and_ci(range(1, 901), digits=1, credibility=80)
    assert test == {"median": 450.5, "ci_low": 90.9, "ci_high": 810.1}


def test_get_median_and_ci_weighted():
    test1 = get_median_and_ci(range(1, 901), weights=range(900), digits=1)
    assert test1 == {"median": 637.0, "ci_low": 202.1, "ci_high": 877.7}


def test_geomean():
    assert round(geomean([0.1, 0.2, 0.3, 0.4, 0.5]), 2) == 0.26


def test_geomean_numpy():
    assert round(geomean(np.array([0.1, 0.2, 0.3, 0.4, 0.5])), 2) == 0.26


def test_weighted_geomean():
    assert round(geomean([0.1, 0.2, 0.3, 0.4, 0.5], weights=[0.5, 0.1, 0.1, 0.1, 0.2]), 2) == 0.19


def test_relative_weighted_geomean():
    assert round(geomean([0.1, 0.2, 0.3, 0.4, 0.5], relative_weights=[5, 1, 1, 1, 2]), 2) == 0.19


def test_geomean_with_none_value():
    assert round(geomean([0.1, 0.2, None, 0.3, 0.4, None, 0.5]), 2) == 0.26


def test_weighted_geomean_alt_format():
    assert round(geomean([[0.5, 0.1], [0.1, 0.2], [0.1, 0.3], [0.1, 0.4], [0.2, 0.5]]), 2) == 0.19


def test_weighted_geomean_alt2_format():
    assert round(geomean({0.1: 0.5, 0.2: 0.1, 0.3: 0.1, 0.4: 0.1, 0.5: 0.2}), 2) == 0.19


def test_weighted_geomean_errors_with_none_value():
    with pytest.raises(ValueError) as execinfo:
        geomean({0.1: 0.5, 0.2: 0.1, 0.3: None, 0.4: 0.1, 0.5: 0.2})
    assert "cannot handle NA-like values in weights" in str(execinfo.value)
    with pytest.raises(ValueError) as execinfo:
        geomean([[0.5, 0.1], [0.1, None], [0.1, 0.3], [0.1, 0.4], [0.2, 0.5]])
    assert "cannot drop NA and process weights" in str(execinfo.value)


def test_p_to_odds():
    assert round(p_to_odds(0.1), 2) == 0.11


def test_odds_to_p():
    assert round(odds_to_p(1 / 9), 2) == 0.1


def test_p_to_odds_handles_none():
    assert p_to_odds(None) is None


def test_odds_to_p_handles_none():
    assert odds_to_p(None) is None


def test_p_to_odds_handles_multiple():
    assert all(np.round(p_to_odds([0.1, 0.2, 0.3]), 2) == np.array([0.11, 0.25, 0.43]))


def test_odds_to_p_handles_multiple():
    assert all(np.round(odds_to_p([0.1, 0.2, 0.3]), 2) == np.array([0.09, 0.17, 0.23]))


def test_geomean_odds():
    assert round(geomean_odds([0.1, 0.2, 0.3, 0.4, 0.5]), 2) == 0.28


def test_geomean_odds_numpy():
    assert round(geomean_odds(np.array([0.1, 0.2, 0.3, 0.4, 0.5])), 2) == 0.28


def test_weighted_geomean_odds():
    assert (
        round(
            geomean_odds([0.1, 0.2, 0.3, 0.4, 0.5], weights=[0.5, 0.1, 0.1, 0.1, 0.2]),
            2,
        )
        == 0.2
    )


def test_weighted_geomean_odds_alt_format():
    assert (
        round(
            geomean_odds([[0.5, 0.1], [0.1, 0.2], [0.1, 0.3], [0.1, 0.4], [0.2, 0.5]]),
            2,
        )
        == 0.2
    )


def test_weighted_geomean_odds_alt2_format():
    assert round(geomean_odds({0.1: 0.5, 0.2: 0.1, 0.3: 0.1, 0.4: 0.1, 0.5: 0.2}), 2) == 0.2


def test_laplace_simple():
    test = laplace(0, 1)
    expected = 1 / 3
    assert test == expected


def test_laplace_s_is_1():
    test = laplace(1, 1)
    expected = 2 / 3
    assert test == expected


def test_laplace_s_gt_n():
    with pytest.raises(ValueError) as excinfo:
        laplace(3, 2)
    assert "`s` cannot be greater than `n`" in str(excinfo.value)


def test_time_invariant_laplace_zero_s():
    assert laplace(s=0, time_passed=2, time_remaining=2) == 0.5


def test_time_invariant_laplace_one_s_time_fixed():
    assert laplace(s=1, time_passed=2, time_remaining=2, time_fixed=True) == 0.75


def test_time_invariant_laplace_one_s_time_variable():
    assert laplace(s=1, time_passed=2, time_remaining=2) == 0.5


def test_time_invariant_laplace_infer_time_remaining():
    assert round(laplace(s=1, time_passed=2), 2) == 0.33


def test_time_invariant_laplace_two_s():
    assert laplace(s=2, time_passed=2, time_remaining=2) == 0.75


def test_laplace_only_s():
    with pytest.raises(ValueError) as excinfo:
        laplace(3)
    assert "Must define `time_passed` or `n`" in str(excinfo.value)


def test_laplace_no_time_passed():
    with pytest.raises(ValueError) as excinfo:
        laplace(3, time_remaining=1)
    assert "Must define `time_passed`" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        laplace(3, n=10, time_remaining=1)
    assert "Must define `time_passed`" in str(excinfo.value)


def test_growth_rate_to_doubling_time_float():
    assert round(growth_rate_to_doubling_time(0.01), 2) == 69.66
    assert round(growth_rate_to_doubling_time(0.5), 2) == 1.71
    assert round(growth_rate_to_doubling_time(1.0), 2) == 1.0


def test_growth_rate_to_doubling_time_nparray():
    result = growth_rate_to_doubling_time(np.array([0.01, 0.5, 1.0]))
    assert round(result[0], 2) == 69.66
    assert round(result[1], 2) == 1.71
    assert round(result[2], 2) == 1


def test_growth_rate_to_doubling_time_dist():
    assert round(growth_rate_to_doubling_time(const(0.01)) @ 1, 2) == 69.66


def test_doubling_time_to_growth_rate_float():
    assert round(doubling_time_to_growth_rate(12), 2) == 0.06
    assert round(doubling_time_to_growth_rate(5.5), 2) == 0.13
    assert round(doubling_time_to_growth_rate(1), 2) == 1


def test_doubling_time_to_growth_rate_nparray():
    result = doubling_time_to_growth_rate(np.array([12, 5.5, 1]))
    assert round(result[0], 2) == 0.06
    assert round(result[1], 2) == 0.13
    assert round(result[2], 2) == 1


def test_doubling_time_to_growth_rate_dist():
    assert round(doubling_time_to_growth_rate(const(12)) @ 1, 2) == 0.06


def test_roll_die():
    set_seed(42)
    assert roll_die(6) == 5


def test_roll_die_different_sides():
    set_seed(42)
    assert roll_die(4) == 4


def test_roll_die_with_distribution():
    set_seed(42)
    assert (norm(2, 6) >> dist_round >> roll_die) == 2


def test_roll_one_sided_die():
    with pytest.raises(ValueError) as excinfo:
        roll_die(1)
    assert "cannot roll less than a 2-sided die" in str(excinfo.value)


def test_roll_nonint_die():
    with pytest.raises(ValueError) as excinfo:
        roll_die(2.5)
    assert "can only roll an integer number of sides" in str(excinfo.value)


def test_roll_nonint_n():
    with pytest.raises(ValueError) as excinfo:
        roll_die(6, 2.5)
    assert "can only roll an integer number of times" in str(excinfo.value)


def test_roll_five_die():
    set_seed(42)
    assert list(roll_die(4, 4)) == [4, 2, 4, 3]


def test_flip_coin():
    set_seed(42)
    assert flip_coin() == "heads"


def test_flip_five_coins():
    set_seed(42)
    assert flip_coin(5) == ["heads", "tails", "heads", "heads", "tails"]


def test_kelly_market_price_error():
    for val in [0, 1, 2, -1]:
        with pytest.raises(ValueError) as execinfo:
            kelly(my_price=0.99, market_price=val)
        assert "market_price must be >0 and <1" in str(execinfo.value)


def test_kelly_my_price_error():
    for val in [0, 1, 2, -1]:
        with pytest.raises(ValueError) as execinfo:
            kelly(my_price=val, market_price=0.99)
        assert "my_price must be >0 and <1" in str(execinfo.value)


def test_kelly_deference_error():
    for val in [-1, 2]:
        with pytest.raises(ValueError) as execinfo:
            kelly(my_price=0.01, market_price=0.99, deference=val)
        assert "deference must be >=0 and <=1" in str(execinfo.value)


def test_kelly_user_below_market_price_error():
    with pytest.raises(ValueError) as execinfo:
        kelly(my_price=0.1, market_price=0.2)
    assert "below the market price" in str(execinfo.value)
    assert "bypass this issue" in str(execinfo.value)


def test_kelly_user_below_market_price_error_can_be_overriden():
    assert isinstance(kelly(my_price=0.1, market_price=0.2, error=False), dict)


def test_kelly_defaults():
    obj = kelly(my_price=0.99, market_price=0.01)
    assert obj["my_price"] == 0.99
    assert obj["market_price"] == 0.01
    assert obj["deference"] == 0
    assert obj["adj_price"] == 0.99
    assert obj["delta_price"] == 0.98
    assert obj["adj_delta_price"] == 0.98
    assert obj["kelly"] == 0.99
    assert obj["target"] == 0.99
    assert obj["current"] == 0
    assert obj["delta"] == 0.99
    assert obj["max_gain"] == 98.99
    assert obj["modeled_gain"] == 97.99
    assert obj["expected_roi"] == 98
    assert obj["expected_arr"] is None
    assert obj["resolve_date"] is None


def test_full_kelly():
    obj = full_kelly(my_price=0.99, market_price=0.01)
    assert obj["my_price"] == 0.99
    assert obj["market_price"] == 0.01
    assert obj["deference"] == 0
    assert obj["adj_price"] == 0.99
    assert obj["delta_price"] == 0.98
    assert obj["adj_delta_price"] == 0.98
    assert obj["kelly"] == 0.99
    assert obj["target"] == 0.99
    assert obj["current"] == 0
    assert obj["delta"] == 0.99
    assert obj["max_gain"] == 98.99
    assert obj["modeled_gain"] == 97.99
    assert obj["expected_roi"] == 98
    assert obj["expected_arr"] is None
    assert obj["resolve_date"] is None


def test_half_kelly():
    obj = half_kelly(my_price=0.99, market_price=0.01)
    assert obj["my_price"] == 0.99
    assert obj["market_price"] == 0.01
    assert obj["deference"] == 0.5
    assert obj["adj_price"] == 0.5
    assert obj["delta_price"] == 0.98
    assert obj["adj_delta_price"] == 0.49
    assert obj["kelly"] == 0.495
    assert obj["target"] == 0.49
    assert obj["current"] == 0
    assert obj["delta"] == 0.49
    assert obj["max_gain"] == 49.49
    assert obj["modeled_gain"] == 24.5
    assert obj["expected_roi"] == 49
    assert obj["expected_arr"] is None
    assert obj["resolve_date"] is None


def test_third_kelly():
    obj = third_kelly(my_price=0.99, market_price=0.01)
    assert obj["my_price"] == 0.99
    assert obj["market_price"] == 0.01
    assert obj["deference"] == 0.66
    assert obj["adj_price"] == 0.26
    assert obj["delta_price"] == 0.98
    assert obj["adj_delta_price"] == 0.24
    assert obj["kelly"] == 0.247
    assert obj["target"] == 0.25
    assert obj["current"] == 0
    assert obj["delta"] == 0.25
    assert obj["max_gain"] == 24.75
    assert obj["modeled_gain"] == 6.13
    assert obj["expected_roi"] == 24.5
    assert obj["expected_arr"] is None
    assert obj["resolve_date"] is None


def test_quarter_kelly():
    obj = quarter_kelly(my_price=0.99, market_price=0.01)
    assert obj["my_price"] == 0.99
    assert obj["market_price"] == 0.01
    assert obj["deference"] == 0.75
    assert obj["adj_price"] == 0.26
    assert obj["delta_price"] == 0.98
    assert obj["adj_delta_price"] == 0.24
    assert obj["kelly"] == 0.247
    assert obj["target"] == 0.25
    assert obj["current"] == 0
    assert obj["delta"] == 0.25
    assert obj["max_gain"] == 24.75
    assert obj["modeled_gain"] == 6.13
    assert obj["expected_roi"] == 24.5
    assert obj["expected_arr"] is None
    assert obj["resolve_date"] is None


def test_kelly_with_bankroll():
    obj = kelly(my_price=0.99, market_price=0.01, bankroll=1000)
    assert obj["my_price"] == 0.99
    assert obj["market_price"] == 0.01
    assert obj["deference"] == 0
    assert obj["adj_price"] == 0.99
    assert obj["delta_price"] == 0.98
    assert obj["adj_delta_price"] == 0.98
    assert obj["kelly"] == 0.99
    assert obj["target"] == 989.9
    assert obj["current"] == 0
    assert obj["delta"] == 989.9
    assert obj["max_gain"] == 98989.9
    assert obj["modeled_gain"] == 97990.1
    assert obj["expected_roi"] == 98
    assert obj["expected_arr"] is None
    assert obj["resolve_date"] is None


def test_kelly_with_current():
    obj = kelly(my_price=0.99, market_price=0.01, bankroll=1000, current=100)
    assert obj["my_price"] == 0.99
    assert obj["market_price"] == 0.01
    assert obj["deference"] == 0
    assert obj["adj_price"] == 0.99
    assert obj["delta_price"] == 0.98
    assert obj["adj_delta_price"] == 0.98
    assert obj["kelly"] == 0.99
    assert obj["target"] == 989.9
    assert obj["current"] == 100
    assert obj["delta"] == 889.9
    assert obj["max_gain"] == 98989.9
    assert obj["modeled_gain"] == 97990.1
    assert obj["expected_roi"] == 98
    assert obj["expected_arr"] is None
    assert obj["resolve_date"] is None


def test_kelly_with_resolve_date():
    one_year_from_today = datetime.now() + timedelta(days=365)
    one_year_from_today_str = one_year_from_today.strftime("%Y-%m-%d")
    obj = kelly(my_price=0.99, market_price=0.01, resolve_date=one_year_from_today_str)
    assert obj["my_price"] == 0.99
    assert obj["market_price"] == 0.01
    assert obj["deference"] == 0
    assert obj["adj_price"] == 0.99
    assert obj["delta_price"] == 0.98
    assert obj["adj_delta_price"] == 0.98
    assert obj["kelly"] == 0.99
    assert obj["target"] == 0.99
    assert obj["current"] == 0
    assert obj["delta"] == 0.99
    assert obj["max_gain"] == 98.99
    assert obj["modeled_gain"] == 97.99
    assert obj["expected_roi"] == 98
    assert obj["expected_arr"] == 99.258
    assert obj["resolve_date"] == datetime(
        one_year_from_today.year,
        one_year_from_today.month,
        one_year_from_today.day,
        0,
        0,
    )


def test_kelly_with_resolve_date2():
    two_years_from_today = datetime.now() + timedelta(days=365 * 2)
    two_years_from_today_str = two_years_from_today.strftime("%Y-%m-%d")
    obj = kelly(my_price=0.99, market_price=0.01, resolve_date=two_years_from_today_str)
    assert obj["my_price"] == 0.99
    assert obj["market_price"] == 0.01
    assert obj["deference"] == 0
    assert obj["adj_price"] == 0.99
    assert obj["delta_price"] == 0.98
    assert obj["adj_delta_price"] == 0.98
    assert obj["kelly"] == 0.99
    assert obj["target"] == 0.99
    assert obj["current"] == 0
    assert obj["delta"] == 0.99
    assert obj["max_gain"] == 98.99
    assert obj["modeled_gain"] == 97.99
    assert obj["expected_roi"] == 98
    assert obj["expected_arr"] == 8.981
    assert obj["resolve_date"] == datetime(
        two_years_from_today.year,
        two_years_from_today.month,
        two_years_from_today.day,
        0,
        0,
    )


def test_kelly_with_resolve_date0pt5():
    half_year_from_today = datetime.now() + timedelta(days=int(round(365 * 0.5)))
    half_year_from_today_str = half_year_from_today.strftime("%Y-%m-%d")
    obj = kelly(my_price=0.99, market_price=0.01, resolve_date=half_year_from_today_str)
    assert obj["my_price"] == 0.99
    assert obj["market_price"] == 0.01
    assert obj["deference"] == 0
    assert obj["adj_price"] == 0.99
    assert obj["delta_price"] == 0.98
    assert obj["adj_delta_price"] == 0.98
    assert obj["kelly"] == 0.99
    assert obj["target"] == 0.99
    assert obj["current"] == 0
    assert obj["delta"] == 0.99
    assert obj["max_gain"] == 98.99
    assert obj["modeled_gain"] == 97.99
    assert obj["expected_roi"] == 98
    assert obj["expected_arr"] == 10575.628
    assert obj["resolve_date"] == datetime(
        half_year_from_today.year,
        half_year_from_today.month,
        half_year_from_today.day,
        0,
        0,
    )


def test_kelly_worked_example():
    half_year_from_today = datetime.now() + timedelta(days=int(round(365 * 0.5)))
    half_year_from_today_str = half_year_from_today.strftime("%Y-%m-%d")
    obj = sq.kelly(my_price=0.6, market_price=0.17, deference=0.66, bankroll=46288, resolve_date=half_year_from_today_str, current=7300)
    assert obj["my_price"] == 0.6
    assert obj["market_price"] == 0.17
    assert obj["deference"] == 0.66
    assert obj["adj_price"] == 0.32
    assert obj["delta_price"] == 0.43
    assert obj["adj_delta_price"] == 0.15
    assert obj["kelly"] == 0.176
    assert obj["target"] == 8153.38
    assert obj["current"] == 7300
    assert obj["delta"] == 853.38
    assert obj["max_gain"] == 6767.31
    assert obj["modeled_gain"] == 1223
    assert obj["expected_roi"] == 0.86
    assert obj["expected_arr"] == 0.86
    assert obj["resolve_date"] == datetime(
        half_year_from_today.year,
        half_year_from_today.month,
        half_year_from_today.day,
        0,
        0,
    )


def test_extremize():
    assert round(extremize(p=0.7, e=1), 3) == 0.7
    assert round(extremize(p=0.7, e=1.73), 3) == 0.875
    assert round(extremize(p=0.2, e=1.73), 3) == 0.062


def test_extremize_out_of_bounds():
    for p in [-1, 0, 1, 2]:
        with pytest.raises(ValueError) as execinfo:
            extremize(p=p, e=1.73)
        assert "must be greater than 0 and less than 1" in str(execinfo.value)


def test_core_cuts():
    assert _core_cuts(10, 2) == [5, 5]
    assert _core_cuts(10, 3) == [3, 3, 4]


def test_sharpe_ratio():
    assert round(sharpe_ratio([0.04, -0.03, 0.05, 0.02, 0.03]), 4) == 0.7898
