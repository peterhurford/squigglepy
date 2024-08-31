import os
import pytest
import numpy as np
from unittest.mock import patch, Mock

from ..squigglepy.distributions import (
    const,
    uniform,
    norm,
    lognorm,
    binomial,
    beta,
    bernoulli,
    discrete,
    tdist,
    log_tdist,
    triangular,
    pert,
    chisquare,
    poisson,
    exponential,
    gamma,
    pareto,
    mixture,
    zero_inflated,
    inf0,
    geometric,
    dist_min,
    dist_max,
    dist_round,
    dist_ceil,
    dist_floor,
    lclip,
    rclip,
    clip,
    dist_fn,
)
from ..squigglepy import samplers
from ..squigglepy.utils import _is_numpy
from ..squigglepy.samplers import (
    normal_sample,
    lognormal_sample,
    mixture_sample,
    discrete_sample,
    log_t_sample,
    t_sample,
    sample,
)
from ..squigglepy.distributions import NormalDistribution


class FakeRNG:
    def normal(self, mu, sigma, n):
        return round(mu, 2), round(sigma, 2)

    def lognormal(self, mu, sigma, n):
        return round(mu, 2), round(sigma, 2)

    def uniform(self, low, high, n):
        return low, high

    def binomial(self, n, p, nsamp):
        return n, p

    def beta(self, a, b, n):
        return a, b

    def bernoulli(self, p, n):
        return p

    def gamma(self, shape, scale, n):
        return shape, scale

    def pareto(self, shape, n):
        return shape

    def poisson(self, lam, n):
        return lam

    def exponential(self, scale, n):
        return scale

    def triangular(self, left, mode, right, n):
        return left, mode, right

    def standard_t(self, t, n):
        return t

    def chisquare(self, df, n):
        return df

    def geometric(self, p, n):
        return p


def test_noop():
    assert sample() is None


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_norm():
    assert normal_sample(1, 2) == (1, 2)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_norm():
    assert sample(norm(1, 2)) == (1.5, 0.3)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_norm_shorthand():
    assert ~norm(1, 2) == (1.5, 0.3)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_norm_with_credibility():
    assert sample(norm(1, 2, credibility=70)) == (1.5, 0.48)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_norm_with_just_sd_infers_zero_mean():
    assert sample(norm(sd=2)) == (0, 2)


@patch.object(samplers, "normal_sample", Mock(return_value=-100))
def test_sample_norm_passes_lclip():
    assert sample(norm(1, 2)) == -100
    assert sample(norm(1, 2, lclip=1)) == 1


@patch.object(samplers, "normal_sample", Mock(return_value=100))
def test_sample_norm_passes_rclip():
    assert sample(norm(1, 2)) == 100
    assert sample(norm(1, 2, rclip=3)) == 3


@patch.object(samplers, "normal_sample", Mock(return_value=100))
def test_sample_norm_passes_lclip_rclip():
    assert sample(norm(1, 2)) == 100
    assert sample(norm(1, 2, lclip=1, rclip=3)) == 3
    assert ~norm(1, 2, lclip=1, rclip=3) == 3


@patch.object(samplers, "normal_sample", Mock(return_value=100))
def test_sample_norm_competing_clip():
    assert sample(norm(1, 2)) == 100
    assert sample(norm(1, 2, rclip=3)) == 3
    assert sample(norm(1, 2, rclip=3), rclip=2) == 2
    assert sample(norm(1, 2, rclip=2), rclip=3) == 2


@patch.object(samplers, "normal_sample", Mock(return_value=100))
def test_sample_norm_competing_clip_multiple_values():
    assert all(sample(norm(1, 2), n=3) == np.array([100, 100, 100]))
    assert all(sample(norm(1, 2, rclip=3), n=3) == np.array([3, 3, 3]))
    assert all(sample(norm(1, 2, rclip=3), rclip=2, n=3) == np.array([2, 2, 2]))
    assert all(sample(norm(1, 2, rclip=2), rclip=3, n=3) == np.array([2, 2, 2]))


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_lognorm():
    assert lognormal_sample(1, 2) == (1, 2)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_lognorm():
    assert sample(lognorm(1, 2)) == (0.35, 0.21)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_lognorm_with_credibility():
    assert sample(lognorm(1, 2, credibility=70)) == (0.35, 0.33)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_shorthand_lognorm():
    assert ~lognorm(1, 2) == (0.35, 0.21)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_shorthand_lognorm_with_credibility():
    assert ~lognorm(1, 2, credibility=70) == (0.35, 0.33)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_lognorm_with_just_normsd_infers_zero_mean():
    assert sample(lognorm(norm_sd=2)) == (0, 2)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_lognorm_with_just_lognormsd_infers_unit_mean():
    assert sample(lognorm(lognorm_sd=2)) == (-0.8, 1.27)


@patch.object(samplers, "lognormal_sample", Mock(return_value=100))
def test_sample_lognorm_passes_lclip_rclip():
    assert sample(lognorm(1, 2)) == 100
    assert sample(lognorm(1, 2, lclip=1, rclip=3)) == 3


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_uniform():
    assert sample(uniform(1, 2)) == (1, 2)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_binomial():
    assert sample(binomial(10, 0.1)) == (10, 0.1)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_beta():
    assert sample(beta(10, 1)) == (10, 1)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_bernoulli():
    assert sample(bernoulli(0.1)) == 1


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "normal_sample", Mock(return_value=1))
def test_tdist():
    assert round(t_sample(1, 2, 3), 2) == 1


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "normal_sample", Mock(return_value=1))
def test_tdist_t():
    assert round(t_sample(), 2) == 20


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "normal_sample", Mock(return_value=1))
def test_tdist_with_credibility():
    assert round(t_sample(1, 2, 3, credibility=70), 2) == 1


def test_tdist_low_gt_high():
    with pytest.raises(ValueError) as execinfo:
        t_sample(10, 5, 3)
    assert "`high value` cannot be lower than `low value`" in str(execinfo.value)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "normal_sample", Mock(return_value=1))
def test_sample_tdist():
    assert round(sample(tdist(1, 2, 3)), 2) == 1


@patch.object(samplers, "t_sample", Mock(return_value=100))
def test_sample_tdist_passes_lclip_rclip():
    assert sample(tdist(1, 2, 3)) == 100
    assert sample(tdist(1, 2, 3, lclip=1, rclip=3)) == 3


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "normal_sample", Mock(return_value=1))
def test_log_tdist():
    assert round(log_t_sample(1, 2, 3), 2) == 2.72


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "normal_sample", Mock(return_value=1))
def test_log_tdist_with_credibility():
    assert round(log_t_sample(1, 2, 3, credibility=70), 2) == 2.72


def test_log_tdist_low_gt_high():
    with pytest.raises(ValueError) as execinfo:
        log_t_sample(10, 5, 3)
    assert "`high value` cannot be lower than `low value`" in str(execinfo.value)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "normal_sample", Mock(return_value=1))
def teslog_t_sample_log_tdist():
    assert round(sample(log_tdist(1, 2, 3)), 2) == 1 / 3


@patch.object(samplers, "log_t_sample", Mock(return_value=100))
def teslog_t_sample_log_tdist_passes_lclip_rclip():
    assert sample(log_tdist(1, 2, 3)) == 100
    assert sample(log_tdist(1, 2, 3, lclip=1, rclip=3)) == 3


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_triangular():
    assert sample(triangular(10, 20, 30)) == (10, 20, 30)


@patch.object(samplers, "pert_sample", Mock(return_value=100))
def test_sample_pert():
    assert sample(pert(10, 20, 30, 40)) == 100


@patch.object(samplers, "pert_sample", Mock(return_value=100))
def test_sample_pert_passes_lclip_rclip():
    assert sample(pert(1, 2, 3, 4)) == 100
    assert sample(pert(1, 2, 3, 4, lclip=1, rclip=3)) == 3


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_exponential():
    assert sample(exponential(10)) == 10


@patch.object(samplers, "exponential_sample", Mock(return_value=100))
def test_sample_exponential_passes_lclip_rclip():
    assert sample(exponential(1)) == 100
    assert sample(exponential(1, lclip=1, rclip=3)) == 3


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_chisquare():
    assert sample(chisquare(9)) == 9


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_poisson():
    assert sample(poisson(10)) == 10


@patch.object(samplers, "poisson_sample", Mock(return_value=100))
def test_sample_poisson_passes_lclip_rclip():
    assert sample(poisson(1)) == 100
    assert sample(poisson(1, lclip=1, rclip=3)) == 3


def test_sample_const():
    assert sample(const(11)) == 11


def test_sample_const_shorthand():
    assert ~const(11) == 11


def test_nested_const_does_not_resolve():
    assert isinstance((~const(norm(1, 2))), NormalDistribution)
    assert (~const(norm(1, 2))).x == 1
    assert (~const(norm(1, 2))).y == 2


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_nested_const_double_resolve():
    assert ~~const(norm(1, 2)) == (1.5, 0.3)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_gamma_default():
    assert sample(gamma(10)) == (10, 1)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_gamma():
    assert sample(gamma(10, 2)) == (10, 2)


@patch.object(samplers, "gamma_sample", Mock(return_value=100))
def test_sample_gamma_passes_lclip_rclip():
    assert sample(gamma(1, 2)) == 100
    assert sample(gamma(1, 2, lclip=1, rclip=3)) == 3


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_pareto_default():
    assert sample(pareto(10)) == 11


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_discrete():
    assert discrete_sample([0, 1, 2]) == 0


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_discrete_alt_format():
    assert discrete_sample([[0.9, "a"], [0.1, "b"]]) == "a"


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_discrete_alt2_format():
    assert discrete_sample({"a": 0.9, "b": 0.1}) == "a"


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_discrete():
    assert sample(discrete([0, 1, 2])) == 0


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_discrete_alt_format():
    assert sample(discrete([[0.9, "a"], [0.1, "b"]])) == "a"


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_discrete_alt2_format():
    assert sample(discrete({"a": 0.9, "b": 0.1})) == "a"


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_discrete_shorthand():
    assert ~discrete([0, 1, 2]) == 0
    assert ~discrete([[0.9, "a"], [0.1, "b"]]) == "a"
    assert ~discrete({"a": 0.9, "b": 0.1}) == "a"


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_discrete_cannot_mixture():
    obj = ~discrete([norm(1, 2), norm(3, 4)])
    # Instead of sampling `norm(1, 2)`, discrete just returns it unsampled.
    assert isinstance(obj, NormalDistribution)
    assert obj.x == 1
    assert obj.y == 2


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_discrete_indirect_mixture():
    # You would have to double resolve this to get a value.
    assert ~~discrete([norm(1, 2), norm(3, 4)]) == (1.5, 0.3)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_mixture_sample():
    assert mixture_sample([norm(1, 2), norm(3, 4)], [0.2, 0.8]) == (1.5, 0.3)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_mixture_sample_relative_weights():
    assert mixture_sample([norm(1, 2), norm(3, 4)], relative_weights=[1, 1]) == (1.5, 0.3)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_mixture_sample_alt_format():
    assert mixture_sample([[0.2, norm(1, 2)], [0.8, norm(3, 4)]]) == (1.5, 0.3)


@patch.object(samplers, "normal_sample", Mock(return_value=100))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_mixture_sample_rclip_lclip():
    assert mixture_sample([norm(1, 2), norm(3, 4)], [0.2, 0.8]) == 100
    assert mixture_sample([norm(1, 2, rclip=3), norm(3, 4)], [0.2, 0.8]) == 3


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_mixture_sample_no_weights():
    assert mixture_sample([norm(1, 2), norm(3, 4)]) == (1.5, 0.3)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_mixture_sample_different_distributions():
    assert mixture_sample([lognorm(1, 2), norm(3, 4)]) == (0.35, 0.21)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_mixture_sample_with_numbers():
    assert mixture_sample([2, norm(3, 4)]) == 2


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_mixture():
    assert sample(mixture([norm(1, 2), norm(3, 4)], [0.2, 0.8])) == (1.5, 0.3)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_mixture_alt_format():
    assert sample(mixture([[0.2, norm(1, 2)], [0.8, norm(3, 4)]])) == (1.5, 0.3)


@patch.object(samplers, "normal_sample", Mock(return_value=1))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_mixture_lclip():
    assert sample(mixture([norm(1, 2), norm(3, 4)], [0.2, 0.8])) == 1
    assert sample(mixture([norm(1, 2, lclip=3), norm(3, 4)], [0.2, 0.8])) == 3
    assert sample(mixture([norm(1, 2), norm(3, 4)], [0.2, 0.8], lclip=3)) == 3
    assert sample(mixture([norm(1, 2), norm(3, 4)], [0.2, 0.8]), lclip=3) == 3


@patch.object(samplers, "normal_sample", Mock(return_value=1))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_mixture_lclip_multiple_values():
    assert all(sample(mixture([norm(1, 2), norm(3, 4)], [0.2, 0.8]), n=3) == np.array([1, 1, 1]))
    assert all(
        sample(mixture([norm(1, 2, lclip=3), norm(3, 4)], [0.2, 0.8])) == np.array([3, 3, 3])
    )
    assert all(
        sample(mixture([norm(1, 2), norm(3, 4)], [0.2, 0.8], lclip=3)) == np.array([3, 3, 3])
    )
    assert all(
        sample(mixture([norm(1, 2), norm(3, 4)], [0.2, 0.8]), lclip=3) == np.array([3, 3, 3])
    )


@patch.object(samplers, "normal_sample", Mock(return_value=1))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_mixture_lclip_alt_format():
    assert sample(mixture([[0.2, norm(1, 2, lclip=3)], [0.8, norm(3, 4)]])) == 3


@patch.object(samplers, "normal_sample", Mock(return_value=1))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_mixture_sample_lclip_alt_format_multiple_values():
    assert all(
        mixture_sample([[0.2, norm(1, 2, lclip=3)], [0.8, norm(3, 4)]], samples=3)
        == np.array([3, 3, 3])
    )


@patch.object(samplers, "normal_sample", Mock(return_value=1))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_mixture_lclip_alt_format_multiple_values():
    assert all(
        sample(mixture([[0.2, norm(1, 2, lclip=3)], [0.8, norm(3, 4)]]), n=3)
        == np.array([3, 3, 3])
    )


@patch.object(samplers, "normal_sample", Mock(return_value=1))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_mixture_lclip_alt_format2():
    assert ~mixture([[0.2, norm(1, 2, lclip=3)], [0.8, norm(3, 4)]]) == 3


@patch.object(samplers, "normal_sample", Mock(return_value=1))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_mixture_lclip_alt_format2_multiple_values():
    assert all(mixture([[0.2, norm(1, 2, lclip=3)], [0.8, norm(3, 4)]]) @ 3 == np.array([3, 3, 3]))


@patch.object(samplers, "normal_sample", Mock(return_value=100))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_mixture_rclip():
    assert sample(mixture([norm(1, 2), norm(3, 4)], [0.2, 0.8])) == 100
    assert sample(mixture([norm(1, 2, rclip=3), norm(3, 4)], [0.2, 0.8])) == 3
    assert sample(mixture([norm(1, 2), norm(3, 4)], [0.2, 0.8], rclip=3)) == 3
    assert sample(mixture([norm(1, 2), norm(3, 4)], [0.2, 0.8]), rclip=3) == 3


@patch.object(samplers, "normal_sample", Mock(return_value=100))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_mixture_competing_clip():
    assert sample(mixture([norm(1, 2, rclip=3), norm(3, 4)], [0.2, 0.8])) == 3
    assert sample(mixture([norm(1, 2, rclip=2), norm(3, 4)], [0.2, 0.8], rclip=3)) == 2
    assert sample(mixture([norm(1, 2, rclip=3), norm(3, 4)], [0.2, 0.8]), rclip=2) == 2


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_mixture_no_weights():
    assert sample(mixture([norm(1, 2), norm(3, 4)])) == (1.5, 0.3)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_mixture_different_distributions():
    assert sample(mixture([lognorm(1, 2), norm(3, 4)])) == (0.35, 0.21)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_mixture_with_numbers():
    assert sample(mixture([2, norm(3, 4)])) == 2


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_mixture_can_be_discrete():
    assert ~mixture([0, 1, 2]) == 0
    assert ~mixture([[0.9, "a"], [0.1, "b"]]) == "a"
    assert ~mixture({"a": 0.9, "b": 0.1}) == "a"
    assert ~mixture([norm(1, 2), norm(3, 4)]) == (1.5, 0.3)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_mixture_contains_discrete():
    assert sample(mixture([lognorm(1, 2), discrete([3, 4])])) == (0.35, 0.21)


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_mixture_contains_mixture():
    assert sample(mixture([lognorm(1, 2), mixture([1, discrete([3, 4])])])) == (
        0.35,
        0.21,
    )


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_zero_inflated():
    assert ~zero_inflated(0.6, norm(1, 2)) == 0


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
@patch.object(samplers, "uniform_sample", Mock(return_value=0))
def test_sample_inf0():
    assert ~inf0(0.6, norm(1, 2)) == 0


@patch.object(samplers, "_get_rng", Mock(return_value=FakeRNG()))
def test_sample_geometric():
    assert sample(geometric(0.1)) == 0.1


def test_sample_n_gt_1_norm():
    out = sample(norm(1, 2), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_lognorm():
    out = sample(lognorm(1, 2), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_binomial():
    out = sample(binomial(5, 0.1), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_beta():
    out = sample(beta(5, 10), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_bernoulli():
    out = sample(bernoulli(0.1), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_poisson():
    out = sample(poisson(0.1), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_chisquare():
    out = sample(chisquare(10), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_gamma():
    out = sample(gamma(10, 10), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_triangular():
    out = sample(triangular(1, 2, 3), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_tdist():
    out = sample(tdist(1, 2, 3), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_tdist_t():
    out = sample(tdist(), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_log_tdist():
    out = sample(log_tdist(1, 2, 3), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_const():
    out = sample(const(1), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_uniform():
    out = sample(uniform(0, 1), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_discrete():
    out = sample(discrete([1, 2, 3]), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_mixture():
    out = sample(mixture([norm(1, 2), norm(3, 4)]), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_geometric():
    out = sample(geometric(0.1), n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_raw_float():
    out = sample(0.1, n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_raw_int():
    out = sample(1, n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_raw_str():
    out = sample("a", n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_complex():
    out = sample(uniform(0, 1) + 5 >> dist_ceil, n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_gt_1_callable():
    def _fn():
        return norm(1, 2) + norm(3, 4)

    out = sample(_fn, n=5)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_shorthand_n_gt_1():
    out = norm(1, 2) @ 5
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_shorthand_n_is_var():
    n = 2 + 3
    out = norm(1, 2) @ n
    assert _is_numpy(out)
    assert len(out) == n


def test_sample_shorthand_n_is_float():
    out = norm(1, 2) @ 7.0
    assert _is_numpy(out)
    assert len(out) == 7.0


def test_sample_shorthand_n_is_numpy_int():
    out = norm(1, 2) @ np.int64(4)
    assert _is_numpy(out)
    assert len(out) == 4
    out = norm(1, 2) @ np.int32(4)
    assert _is_numpy(out)
    assert len(out) == 4


def test_sample_shorthand_n_gt_1_alt():
    out = 5 @ norm(1, 2)
    assert _is_numpy(out)
    assert len(out) == 5


def test_sample_n_is_0_is_error():
    with pytest.raises(ValueError) as execinfo:
        sample(norm(1, 5), n=0)
    assert "n must be >= 1" in str(execinfo.value)


def test_sample_n_is_0_is_error_shorthand():
    with pytest.raises(ValueError) as execinfo:
        norm(1, 5) @ 0
    assert "n must be >= 1" in str(execinfo.value)


def test_sample_n_is_0_is_error_shorthand_alt():
    with pytest.raises(ValueError) as execinfo:
        0 @ norm(1, 5)
    assert "n must be >= 1" in str(execinfo.value)


def test_sample_int():
    assert sample(4) == 4


def test_sample_float():
    assert sample(3.14) == 3.14


def test_sample_str():
    assert sample("a") == "a"


def test_sample_none():
    assert sample(None) is None


def test_sample_callable():
    def sample_fn():
        return 1

    assert sample(sample_fn) == 1


@patch.object(samplers, "normal_sample", Mock(return_value=1))
@patch.object(samplers, "lognormal_sample", Mock(return_value=4))
def test_sample_more_complex_callable():
    def sample_fn():
        return max(~norm(1, 4), ~lognorm(1, 10))

    assert sample(sample_fn) == 4


@patch.object(samplers, "normal_sample", Mock(return_value=1))
@patch.object(samplers, "lognormal_sample", Mock(return_value=4))
def test_sample_callable_resolves_fully():
    def sample_fn():
        return norm(1, 4) + lognorm(1, 10)

    assert sample(sample_fn) == 5


@patch.object(samplers, "gamma_sample", Mock(return_value=1))
@patch.object(samplers, "normal_sample", Mock(return_value=1))
@patch.object(samplers, "lognormal_sample", Mock(return_value=4))
def test_sample_callable_resolves_fully2():
    def really_inner_sample_fn():
        return gamma(1, 4)

    def inner_sample_fn():
        return norm(1, 4) + lognorm(1, 10) + really_inner_sample_fn()

    def outer_sample_fn():
        return inner_sample_fn

    assert sample(outer_sample_fn) == 6


def test_sample_invalid_input():
    with pytest.raises(ValueError) as execinfo:
        sample([1, 5])
    assert "not a sampleable type" in str(execinfo.value)


@patch.object(samplers, "normal_sample", Mock(return_value=100))
@patch.object(samplers, "lognormal_sample", Mock(return_value=100))
def test_sample_math():
    assert ~(norm(0, 1) + lognorm(1, 2)) == 200


@patch.object(samplers, "normal_sample", Mock(return_value=10))
@patch.object(samplers, "lognormal_sample", Mock(return_value=100))
def test_sample_complex_math():
    obj = (2 ** norm(0, 1)) - (8 * 6) + 2 + (lognorm(10, 100) + 11) / 8
    expected = (2**10) - (8 * 6) + 2 + (100 + 11) / 8
    assert ~obj == expected


@patch.object(samplers, "normal_sample", Mock(return_value=100))
def test_sample_equality():
    assert ~(norm(0, 1) == norm(1, 2))


@patch.object(samplers, "normal_sample", Mock(return_value=10))
def test_pipe():
    assert ~(norm(0, 1) >> rclip(2)) == 2
    assert ~(norm(0, 1) >> lclip(2)) == 10


@patch.object(samplers, "normal_sample", Mock(return_value=1.6))
def test_two_pipes():
    assert ~(norm(0, 1) >> rclip(10) >> dist_round) == 2


@patch.object(samplers, "normal_sample", Mock(return_value=10))
def test_dist_fn():
    def mirror(x):
        return 1 - x if x > 0.5 else x

    assert ~dist_fn(norm(0, 1), mirror) == -9


@patch.object(samplers, "normal_sample", Mock(return_value=10))
def test_dist_fn2():
    def mirror(x, y):
        return 1 - x if x > y else x

    assert ~dist_fn(norm(0, 10), norm(1, 2), mirror) == 10


@patch.object(samplers, "normal_sample", Mock(return_value=10))
def test_dist_fn_list():
    def mirror(x):
        return 1 - x if x > 0.5 else x

    def mirror2(x):
        return 1 + x if x > 0.5 else x

    assert ~dist_fn(norm(0, 1), [mirror, mirror2]) == -9


@patch.object(samplers, "normal_sample", Mock(return_value=10))
@patch.object(samplers, "lognormal_sample", Mock(return_value=20))
def test_max():
    assert ~dist_max(norm(0, 1), lognorm(0.1, 1)) == 20


@patch.object(samplers, "normal_sample", Mock(return_value=10))
@patch.object(samplers, "lognormal_sample", Mock(return_value=20))
def test_min():
    assert ~dist_min(norm(0, 1), lognorm(0.1, 1)) == 10


@patch.object(samplers, "normal_sample", Mock(return_value=3.1415))
def test_round():
    assert ~dist_round(norm(0, 1)) == 3


@patch.object(samplers, "normal_sample", Mock(return_value=3.1415))
def test_round_two_digits():
    assert ~dist_round(norm(0, 1), digits=2) == 3.14


@patch.object(samplers, "normal_sample", Mock(return_value=3.1415))
def test_ceil():
    assert ~dist_ceil(norm(0, 1)) == 4


@patch.object(samplers, "normal_sample", Mock(return_value=3.1415))
def test_floor():
    assert ~dist_floor(norm(0, 1)) == 3


@patch.object(samplers, "normal_sample", Mock(return_value=10))
def test_lclip():
    assert ~lclip(norm(0, 1), 0.5) == 10


@patch.object(samplers, "normal_sample", Mock(return_value=10))
def test_rclip():
    assert ~rclip(norm(0, 1), 0.5) == 0.5


@patch.object(samplers, "normal_sample", Mock(return_value=10))
def test_clip():
    assert ~clip(norm(0, 1), 0.5, 0.9) == 0.9


def test_sample_cache():
    from ..squigglepy.samplers import _squigglepy_internal_sample_caches

    n_caches = len(_squigglepy_internal_sample_caches)

    sample(norm(1, 2), memcache=True)
    from ..squigglepy.samplers import _squigglepy_internal_sample_caches

    n_caches2 = len(_squigglepy_internal_sample_caches)
    assert n_caches < n_caches2

    sample(norm(1, 2), memcache=True)
    from ..squigglepy.samplers import _squigglepy_internal_sample_caches

    n_caches3 = len(_squigglepy_internal_sample_caches)
    assert n_caches2 == n_caches3

    sample(norm(1, 2))
    from ..squigglepy.samplers import _squigglepy_internal_sample_caches

    n_caches4 = len(_squigglepy_internal_sample_caches)
    assert n_caches2 == n_caches4

    sample(norm(3, 4))
    from ..squigglepy.samplers import _squigglepy_internal_sample_caches

    n_caches5 = len(_squigglepy_internal_sample_caches)
    assert n_caches2 == n_caches5

    sample(norm(3, 4), memcache=True)
    from ..squigglepy.samplers import _squigglepy_internal_sample_caches

    n_caches6 = len(_squigglepy_internal_sample_caches)
    assert n_caches6 > n_caches5


def test_sample_reload_cache():
    from ..squigglepy.samplers import _squigglepy_internal_sample_caches

    n_caches = len(_squigglepy_internal_sample_caches)

    out1 = sample(norm(5, 6), memcache=True)
    from ..squigglepy.samplers import _squigglepy_internal_sample_caches

    n_caches2 = len(_squigglepy_internal_sample_caches)
    assert n_caches < n_caches2

    out2 = sample(norm(5, 6), memcache=True)
    from ..squigglepy.samplers import _squigglepy_internal_sample_caches

    n_caches3 = len(_squigglepy_internal_sample_caches)
    assert n_caches2 == n_caches3
    assert out1 == out2

    out3 = sample(norm(5, 6), memcache=True, reload_cache=True)
    from ..squigglepy.samplers import _squigglepy_internal_sample_caches

    n_caches4 = len(_squigglepy_internal_sample_caches)
    assert n_caches3 == n_caches4
    assert out3 != out2


@pytest.fixture
def cachefile():
    cachefile = "testcache"
    yield cachefile
    try:
        os.remove(cachefile + ".sqcache.npy")
    except FileNotFoundError:
        pass


def test_sample_cachefile(cachefile):
    assert not os.path.exists(cachefile + ".sqcache.npy")
    sample(norm(1, 2), dump_cache_file=cachefile)
    assert os.path.exists(cachefile + ".sqcache.npy")


def test_sample_cachefile_primary(cachefile):
    assert not os.path.exists(cachefile + ".sqcache.npy")

    from ..squigglepy.samplers import _squigglepy_internal_sample_caches

    n_caches = len(_squigglepy_internal_sample_caches)

    sample(norm(10, 20), dump_cache_file=cachefile, memcache=True)

    from ..squigglepy.samplers import _squigglepy_internal_sample_caches

    n_caches2 = len(_squigglepy_internal_sample_caches)
    assert n_caches2 == n_caches + 1
    assert os.path.exists(cachefile + ".sqcache.npy")

    o1 = sample(norm(10, 20), load_cache_file=cachefile, memcache=True, cache_file_primary=True)
    o2 = sample(norm(10, 20), load_cache_file=cachefile, memcache=True, cache_file_primary=False)
    assert o1 == o2


def test_sample_load_noop_cachefile(cachefile):
    assert not os.path.exists(cachefile + ".sqcache.npy")

    from ..squigglepy.samplers import _squigglepy_internal_sample_caches

    n_caches = len(_squigglepy_internal_sample_caches)

    o1 = sample(norm(100, 200), dump_cache_file=cachefile, memcache=True)

    from ..squigglepy.samplers import _squigglepy_internal_sample_caches

    n_caches2 = len(_squigglepy_internal_sample_caches)
    assert n_caches2 == n_caches + 1
    assert os.path.exists(cachefile + ".sqcache.npy")

    o2 = sample(load_cache_file=cachefile)
    assert o1 == o2


def test_sample_load_noop_nonexisting_cachefile(cachefile):
    assert not os.path.exists(cachefile + ".sqcache.npy")
    assert sample(load_cache_file=cachefile) is None


def test_sample_multicore():
    sample(norm(100, 200), n=100, cores=2)
    assert not os.path.exists("test-core-0.sqcache")
