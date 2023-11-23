from functools import reduce
from hypothesis import assume, given, settings
import hypothesis.strategies as st
import numpy as np
from pytest import approx
from scipy import integrate, stats

from ..squigglepy.distributions import LognormalDistribution
from ..squigglepy.pdh import ProbabilityMassHistogram, ScaledBinHistogram
from ..squigglepy import samplers


def print_accuracy_ratio(x, y, extra_message=None):
    ratio = max(x / y, y / x) - 1
    if extra_message is not None:
        extra_message += " "
    else:
        extra_message = ""
    direction_off = "small" if x < y else "large"
    if ratio > 1:
        print(f"{extra_message}Ratio: {direction_off} by a factor of {ratio:.1f}")
    else:
        print(f"{extra_message}Ratio: {direction_off} by {100 * ratio:.3f}%")


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=5),
)
def test_pmh_mean(norm_mean, norm_sd):
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = ProbabilityMassHistogram.from_distribution(dist, bin_sizing='mass')
    print("Values:", hist.values)
    assert hist.histogram_mean() == approx(stats.lognorm.mean(dist.norm_sd, scale=np.exp(dist.norm_mean)))


@given(
    # norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    # norm_sd=st.floats(min_value=0.01, max_value=5),
    norm_mean=st.just(0),
    norm_sd=st.just(1),
)
def test_pmh_sd(norm_mean, norm_sd):
    # TODO: The margin of error on the SD estimate is pretty big, mostly
    # because the right tail is underestimating variance. But that might be an
    # acceptable cost. Try to see if there's a way to improve it without compromising the fidelity of the EV estimate.
    #
    # Note: Adding more bins increases accuracy overall, but decreases accuracy
    # on the far right tail.
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = ProbabilityMassHistogram.from_distribution(dist, bin_sizing='mass')

    def true_variance(left, right):
        return integrate.quad(
            lambda x: (x - dist.lognorm_mean) ** 2
            * stats.lognorm.pdf(x, dist.norm_sd, scale=np.exp(dist.norm_mean)),
            left,
            right,
        )[0]

    def observed_variance(left, right):
        return np.sum(hist.masses[left:right] * (hist.values[left:right] - hist.histogram_mean()) ** 2)

    midpoint = hist.values[int(num_bins * 9/10)]
    expected_left_variance = true_variance(0, midpoint)
    expected_right_variance = true_variance(midpoint, np.inf)
    midpoint_index = int(len(hist) * hist.contribution_to_ev(midpoint))
    observed_left_variance = observed_variance(0, midpoint_index)
    observed_right_variance = observed_variance(midpoint_index, len(hist))
    print_accuracy_ratio(observed_left_variance, expected_left_variance,   "Left   ")
    print_accuracy_ratio(observed_right_variance, expected_right_variance, "Right  ")
    print_accuracy_ratio(hist.histogram_sd(), dist.lognorm_sd, "Overall")
    assert hist.histogram_sd() == approx(dist.lognorm_sd)


def relative_error(observed, expected):
    return np.exp(abs(np.log(observed / expected))) - 1


def test_mean_error_propagation(verbose=True):
    dist = LognormalDistribution(norm_mean=0, norm_sd=1)
    hist = ProbabilityMassHistogram.from_distribution(dist, bin_sizing='mass')
    hist_base = ProbabilityMassHistogram.from_distribution(dist, bin_sizing='mass')
    abs_error = []
    rel_error = []

    if verbose:
        print("")
    for i in range(1, 17):
        true_mean = stats.lognorm.mean(np.sqrt(i))
        abs_error.append(abs(hist.histogram_mean() - true_mean))
        rel_error.append(relative_error(hist.histogram_mean(), true_mean))
        if verbose:
            print(f"n = {i:2d}: {abs_error[-1]:7.2f} ({rel_error[-1]*100:7.1f}%) from mean {hist.histogram_mean():6.2f}")
        hist = hist * hist_base


def test_mc_mean_error_propagation():
    dist = LognormalDistribution(norm_mean=0, norm_sd=1)
    rel_error = [0]
    print("")
    for i in [1, 2, 4, 8, 16, 32, 64]:
        true_mean = stats.lognorm.mean(np.sqrt(i))
        curr_rel_errors = []
        for _ in range(10):
            mcs = [samplers.sample(dist, 100**2) for _ in range(i)]
            mc = reduce(lambda acc, mc: acc * mc, mcs)
            curr_rel_errors.append(relative_error(np.mean(mc), true_mean))
        rel_error.append(np.mean(curr_rel_errors))
        print(f"n = {i:2d}: {rel_error[-1]*100:4.1f}% (up {(rel_error[-1] + 1) / (rel_error[-2] + 1):.2f}x)")


def test_sd_error_propagation(verbose=True):
    dist = LognormalDistribution(norm_mean=0, norm_sd=1)
    num_bins = 100
    hist = ProbabilityMassHistogram.from_distribution(dist, num_bins=num_bins, bin_sizing='mass')
    abs_error = []
    rel_error = []

    if verbose:
        print("")
    for i in [1, 2, 4, 8, 16, 32, 64]:
        true_mean = stats.lognorm.mean(np.sqrt(i))
        true_sd = hist.exact_sd
        abs_error.append(abs(hist.histogram_sd() - true_sd))
        rel_error.append(relative_error(hist.histogram_sd(), true_sd))
        if verbose:
            print(f"n = {i:2d}: {rel_error[-1]*100:4.1f}% from SD {hist.histogram_sd():.3f}")
        hist = hist * hist

    expected_error_pcts = [0.9, 2.8, 9.9, 40.7, 211, 2678, 630485]
    for i in range(len(expected_error_pcts)):
        assert rel_error[i] < expected_error_pcts[i] / 100


def test_mc_sd_error_propagation():
    dist = LognormalDistribution(norm_mean=0, norm_sd=1)
    num_bins = 100  # we don't actually care about the histogram, we just use it
                    # to calculate exact_sd
    hist = ProbabilityMassHistogram.from_distribution(dist, num_bins=num_bins)
    hist_base = ProbabilityMassHistogram.from_distribution(dist, num_bins=num_bins)
    abs_error = []
    rel_error = [0]
    print("")
    for i in range(1, 17):
        true_mean = stats.lognorm.mean(np.sqrt(i))
        true_sd = hist.exact_sd
        curr_rel_errors = []
        for _ in range(10):
            mcs = [samplers.sample(dist, 1000**2) for _ in range(i)]
            mc = reduce(lambda acc, mc: acc * mc, mcs)
            mc_sd = np.std(mc)
            curr_rel_errors.append(relative_error(mc_sd, true_sd))
        rel_error.append(np.mean(curr_rel_errors))
        print(f"n = {i:2d}: {rel_error[-1]*100:4.1f}% (up {(rel_error[-1] + 1) / (rel_error[-2] + 1):.2f}x)")
        hist = hist * hist_base


def test_sd_accuracy_vs_monte_carlo():
    num_bins = 100
    num_samples = 100**2
    dists = [LognormalDistribution(norm_mean=i, norm_sd=0.5 + i/4) for i in range(5)]
    hists = [ProbabilityMassHistogram.from_distribution(dist, num_bins=num_bins) for dist in dists]
    hist = reduce(lambda acc, hist: acc * hist, hists)
    true_sd = hist.exact_sd
    dist_abs_error = abs(hist.histogram_sd() - true_sd)

    mc_abs_error = []
    for i in range(10):
        mcs = [samplers.sample(dist, num_samples) for dist in dists]
        mc = reduce(lambda acc, mc: acc * mc, mcs)
        mc_abs_error.append(abs(np.std(mc) - true_sd))

    mc_abs_error.sort()

    # dist should be more accurate than at least 8 out of 10 Monte Carlo runs
    assert dist_abs_error < mc_abs_error[8]



@given(
    norm_mean1=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_mean2=st.floats(min_value=-np.log(1e5), max_value=np.log(1e5)),
    norm_sd1=st.floats(min_value=0.1, max_value=3),
    norm_sd2=st.floats(min_value=0.001, max_value=3),
)
@settings(max_examples=100)
def test_exact_moments(norm_mean1, norm_mean2, norm_sd1, norm_sd2):
    dist1 = LognormalDistribution(norm_mean=norm_mean1, norm_sd=norm_sd1)
    dist2 = LognormalDistribution(norm_mean=norm_mean2, norm_sd=norm_sd2)
    hist1 = ProbabilityMassHistogram.from_distribution(dist1)
    hist2 = ProbabilityMassHistogram.from_distribution(dist2)
    hist_prod = hist1 * hist2
    assert hist_prod.exact_mean == approx(
        stats.lognorm.mean(
            np.sqrt(norm_sd1**2 + norm_sd2**2), scale=np.exp(norm_mean1 + norm_mean2)
        )
    )
    assert hist_prod.exact_sd == approx(
        stats.lognorm.std(
            np.sqrt(norm_sd1**2 + norm_sd2**2), scale=np.exp(norm_mean1 + norm_mean2)
        )
    )


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=4),
    bin_num=st.integers(min_value=1, max_value=999),
)
def test_pmh_contribution_to_ev(norm_mean, norm_sd, bin_num):
    fraction = bin_num / 1000
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = ProbabilityMassHistogram.from_distribution(dist)
    assert hist.contribution_to_ev(dist.inv_contribution_to_ev(fraction)) == approx(fraction)


@given(
    norm_mean=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
    norm_sd=st.floats(min_value=0.001, max_value=4),
    bin_num=st.integers(min_value=2, max_value=998),
)
def test_pmh_inv_contribution_to_ev(norm_mean, norm_sd, bin_num):
    # The nth value stored in the PMH represents a value between the nth and n+1th edges
    dist = LognormalDistribution(norm_mean=norm_mean, norm_sd=norm_sd)
    hist = ProbabilityMassHistogram.from_distribution(dist)
    fraction = bin_num / hist.num_bins
    prev_fraction = fraction - 1 / hist.num_bins
    next_fraction = fraction
    assert hist.inv_contribution_to_ev(fraction) > dist.inv_contribution_to_ev(prev_fraction)
    assert hist.inv_contribution_to_ev(fraction) < dist.inv_contribution_to_ev(next_fraction)


# TODO: uncomment
# @given(
#     norm_mean1=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
#     norm_mean2=st.floats(min_value=-np.log(1e9), max_value=np.log(1e9)),
#     norm_sd1=st.floats(min_value=0.1, max_value=3),
#     norm_sd2=st.floats(min_value=0.1, max_value=3),
# )
# def test_lognorm_product_summary_stats(norm_mean1, norm_sd1, norm_mean2, norm_sd2):
def test_lognorm_product_summary_stats():
    # norm_means = np.repeat([0, 1, 1, 100], 4)
    # norm_sds = np.repeat([1, 0.7, 2, 0.1], 4)
    norm_means = np.repeat([0], 2)
    norm_sds = np.repeat([1], 2)
    dists = [
        LognormalDistribution(norm_mean=norm_means[i], norm_sd=norm_sds[i])
        for i in range(len(norm_means))
    ]
    dist_prod = LognormalDistribution(
        norm_mean=np.sum(norm_means), norm_sd=np.sqrt(np.sum(norm_sds**2))
    )
    pmhs = [ProbabilityMassHistogram.from_distribution(dist) for dist in dists]
    pmh_prod = reduce(lambda acc, hist: acc * hist, pmhs)
    print_accuracy_ratio(pmh_prod.histogram_sd(), dist_prod.lognorm_sd)
    assert pmh_prod.histogram_mean() == approx(dist_prod.lognorm_mean)
    assert pmh_prod.histogram_sd() == approx(dist_prod.lognorm_sd)


def test_lognorm_sample():
    # norm_means = np.repeat([0, 1, -1, 100], 4)
    # norm_sds = np.repeat([1, 0.7, 2, 0.1], 4)
    norm_means = np.repeat([0], 2)
    norm_sds = np.repeat([1], 2)
    dists = [
        LognormalDistribution(norm_mean=norm_means[i], norm_sd=norm_sds[i])
        for i in range(len(norm_means))
    ]
    dist_prod = LognormalDistribution(
        norm_mean=np.sum(norm_means), norm_sd=np.sqrt(np.sum(norm_sds**2))
    )
    num_samples = 1e6
    sample_lists = [samplers.sample(dist, num_samples) for dist in dists]
    samples = np.product(sample_lists, axis=0)
    print_accuracy_ratio(np.std(samples), dist_prod.lognorm_sd)
    assert np.std(samples) == approx(dist_prod.lognorm_sd)


def test_scaled_bin():
    for repetitions in [1, 4, 8, 16]:
        norm_means = np.repeat([0], repetitions)
        norm_sds = np.repeat([1], repetitions)
        dists = [
            LognormalDistribution(norm_mean=norm_means[i], norm_sd=norm_sds[i])
            for i in range(len(norm_means))
        ]
        dist_prod = LognormalDistribution(
            norm_mean=np.sum(norm_means), norm_sd=np.sqrt(np.sum(norm_sds**2))
        )
        hists = [ScaledBinHistogram.from_distribution(dist) for dist in dists]
        hist_prod = reduce(lambda acc, hist: acc * hist, hists)
        print("")
        print_accuracy_ratio(hist_prod.histogram_mean(), dist_prod.lognorm_mean, "Mean")
        print_accuracy_ratio(hist_prod.histogram_sd(), dist_prod.lognorm_sd, "SD  ")


def test_accuracy_scaled_vs_flexible():
    for repetitions in [1, 4, 8, 16]:
        norm_means = np.repeat([0], repetitions)
        norm_sds = np.repeat([1], repetitions)
        dists = [
            LognormalDistribution(norm_mean=norm_means[i], norm_sd=norm_sds[i])
            for i in range(len(norm_means))
        ]
        dist_prod = LognormalDistribution(
            norm_mean=np.sum(norm_means), norm_sd=np.sqrt(np.sum(norm_sds**2))
        )
        scaled_hists = [ScaledBinHistogram.from_distribution(dist) for dist in dists]
        scaled_hist_prod = reduce(lambda acc, hist: acc * hist, scaled_hists)
        flexible_hists = [ProbabilityMassHistogram.from_distribution(dist) for dist in dists]
        flexible_hist_prod = reduce(lambda acc, hist: acc * hist, flexible_hists)
        scaled_mean_error = abs(scaled_hist_prod.histogram_mean() - dist_prod.lognorm_mean)
        flexible_mean_error = abs(flexible_hist_prod.histogram_mean() - dist_prod.lognorm_mean)
        scaled_sd_error = abs(scaled_hist_prod.histogram_sd() - dist_prod.lognorm_sd)
        flexible_sd_error = abs(flexible_hist_prod.histogram_sd() - dist_prod.lognorm_sd)
        assert scaled_mean_error > flexible_mean_error
        assert scaled_sd_error > flexible_sd_error
        print("")
        print(
            f"Mean error: scaled = {scaled_mean_error:.3f}, flexible = {flexible_mean_error:.3f}"
        )
        print(f"SD   error: scaled = {scaled_sd_error:.3f}, flexible = {flexible_sd_error:.3f}")


def test_performance():
    return None  # so we don't accidentally run this while running all tests
    import cProfile
    import pstats
    import io

    dist = LognormalDistribution(norm_mean=0, norm_sd=1)

    pr = cProfile.Profile()
    pr.enable()

    for i in range(100):
        hist = ProbabilityMassHistogram.from_distribution(dist, num_bins=1000)
        for _ in range(4):
            hist = hist * hist

    pr.disable()
    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
