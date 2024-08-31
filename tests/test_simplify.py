from hypothesis import example, given
import hypothesis.strategies as st
from numbers import Real
import numpy as np
from pytest import approx

from ..squigglepy.distributions import (
    bernoulli,
    binomial,
    exponential,
    gamma,
    lognorm,
    norm,
    BinomialDistribution,
    ComplexDistribution,
    ExponentialDistribution,
    GammaDistribution,
    LognormalDistribution,
    NormalDistribution,
)


def test_simplify_add_norm():
    x = norm(mean=1, sd=1)
    y = norm(mean=2, sd=2)
    sum2 = x + y
    simplified2 = sum2.simplify()
    assert isinstance(simplified2, NormalDistribution)
    assert simplified2.mean == 3
    assert simplified2.sd == approx(np.sqrt(5))


def test_simplify_add_3_normals():
    x = norm(mean=1, sd=1)
    y = norm(mean=2, sd=2)
    z = norm(mean=-3, sd=2)
    sum3_left = (x + y) + z
    sum3_right = x + (y + z)
    simplified3_left = sum3_left.simplify()
    simplified3_right = sum3_right.simplify()
    assert isinstance(simplified3_left, NormalDistribution)
    assert simplified3_left.mean == 0
    assert simplified3_left.sd == approx(np.sqrt(1 + 4 + 4))
    assert isinstance(simplified3_right, NormalDistribution)
    assert simplified3_right.mean == 0
    assert simplified3_right.sd == approx(np.sqrt(1 + 4 + 4))


def test_simplify_normal_plus_const():
    x = norm(mean=0, sd=1)
    y = 2
    z = norm(mean=1, sd=1)
    simplified = (x + y + z).simplify()
    assert isinstance(simplified, NormalDistribution)
    assert simplified.mean == 3
    assert simplified.sd == approx(np.sqrt(2))


def simplify_scale_norm():
    x = norm(mean=2, sd=4)
    y = 1.5
    product2 = x * y
    simplified = product2.simplify()
    assert isinstance(simplified, NormalDistribution)
    assert simplified.mean == approx(3)
    assert simplified.sd == approx(6)


def test_simplify_mul_3_lognorms():
    x = lognorm(norm_mean=1, norm_sd=1)
    y = lognorm(norm_mean=2, norm_sd=2)
    z = lognorm(norm_mean=-2, norm_sd=3)
    product3_left = (x * y) * z
    product3_right = x * (y * z)
    simplified3_left = product3_left.simplify()
    simplified3_right = product3_right.simplify()
    assert isinstance(simplified3_left, LognormalDistribution)
    assert simplified3_left.norm_mean == 1
    assert simplified3_left.norm_sd == approx(np.sqrt(1 + 4 + 9))
    assert isinstance(simplified3_right, LognormalDistribution)
    assert simplified3_right.norm_mean == 1
    assert simplified3_right.norm_sd == approx(np.sqrt(1 + 4 + 9))


def test_simplify_sub_normals():
    x = norm(mean=1, sd=1)
    y = norm(mean=2, sd=2)
    difference = x - y
    simplified = difference.simplify()
    assert isinstance(simplified, NormalDistribution)
    assert simplified.mean == -1
    assert simplified.sd == approx(np.sqrt(5))


def test_simplify_normal_minus_const():
    x = norm(mean=0, sd=1)
    y = 2
    simplified = (x - y).simplify()
    assert isinstance(simplified, NormalDistribution)
    assert simplified.mean == -2
    assert simplified.sd == approx(1)


@given(
    norm_mean=st.floats(min_value=-10, max_value=10),
    norm_sd=st.floats(min_value=0.1, max_value=3),
    y=st.floats(min_value=0.1, max_value=10),
)
def test_simplify_div_lognorm_by_constant(norm_mean, norm_sd, y):
    x = lognorm(norm_mean=norm_mean, norm_sd=norm_sd)
    simplified = (x / y).simplify()
    assert isinstance(simplified, LognormalDistribution)
    assert simplified.norm_mean == approx(norm_mean - np.log(y))
    assert simplified.norm_sd == approx(norm_sd)


@given(
    x=st.floats(min_value=0.1, max_value=10),
    norm_mean=st.floats(min_value=-10, max_value=10),
    norm_sd=st.floats(min_value=0.1, max_value=3),
)
@example(x=1, norm_mean=0, norm_sd=1)
def test_simplify_div_constant_by_lognorm(x, norm_mean, norm_sd):
    y = lognorm(norm_mean=norm_mean, norm_sd=norm_sd)
    simplified = (x / y).simplify()
    assert isinstance(simplified, LognormalDistribution)
    assert simplified.norm_mean == approx(np.log(x) - norm_mean)
    assert simplified.norm_sd == approx(norm_sd)


def test_simplify_div_lognorms():
    x = lognorm(norm_mean=1, norm_sd=1)
    y = lognorm(norm_mean=2, norm_sd=2)
    quotient = x / y
    simplified = quotient.simplify()
    assert isinstance(simplified, LognormalDistribution)
    assert simplified.norm_mean == -1
    assert simplified.norm_sd == approx(np.sqrt(5))


def test_simplify_div_norm_by_const():
    x = norm(mean=3, sd=1)
    y = 2
    simplified = (x / y).simplify()
    assert isinstance(simplified, NormalDistribution)
    assert simplified.mean == approx(1.5)
    assert simplified.sd == approx(0.5)


def test_simplify_div_norm_by_lognorm():
    x = norm(mean=3, sd=1)
    y = lognorm(norm_mean=2, norm_sd=2)
    simplified = (x / y).simplify()
    assert isinstance(simplified, ComplexDistribution)
    assert simplified.fn_str == "*"
    assert isinstance(simplified.left, NormalDistribution)
    assert simplified.left.mean == 3
    assert simplified.left.sd == 1
    assert isinstance(simplified.right, LognormalDistribution)
    assert simplified.right.norm_mean == -2
    assert simplified.right.norm_sd == approx(2)


def test_simplify_lognorm_pow():
    x = lognorm(norm_mean=3, norm_sd=2)
    y = 2
    product = x**y
    simplified = product.simplify()
    assert isinstance(simplified, LognormalDistribution)
    assert simplified.norm_mean == approx(6)
    assert simplified.norm_sd == approx(4)

    y = -1
    product = x**y
    simplified = product.simplify()
    assert isinstance(simplified, ComplexDistribution)


def test_simplify_clipped_norm():
    x = norm(mean=1, sd=1)
    y = norm(mean=0, sd=1, lclip=1)
    z = norm(mean=3, sd=2)
    simplified = (x + y + z).simplify()
    assert isinstance(simplified, ComplexDistribution)
    assert isinstance(simplified.left, NormalDistribution)
    assert isinstance(simplified.right, NormalDistribution)
    assert simplified.left.mean == 4
    assert simplified.left.sd == approx(np.sqrt(5))
    assert simplified.right.mean == 0
    assert simplified.right.sd == 1
    assert simplified.right.lclip == 1


def test_simplify_bernoulli_sum():
    simplified = (bernoulli(p=0.5) + bernoulli(p=0.5)).simplify()
    assert isinstance(simplified, BinomialDistribution)
    assert simplified.n == 2
    assert simplified.p == 0.5

    # Cannot simplify if the probabilities are different
    simplified = (bernoulli(p=0.5) + bernoulli(p=0.6)).simplify()
    assert isinstance(simplified, ComplexDistribution)


def test_simplify_bernoulli_plus_binomial():
    simplified = (
        bernoulli(p=0.5)
        + binomial(n=10, p=0.2)
        + binomial(n=2, p=0.5)
        + binomial(n=3, p=0.5)
        + bernoulli(p=0.5)
    ).simplify()
    assert isinstance(simplified, ComplexDistribution)
    assert isinstance(simplified.left, BinomialDistribution)
    assert isinstance(simplified.right, BinomialDistribution)
    assert simplified.left.n == 7
    assert simplified.left.p == 0.5
    assert simplified.right.n == 10
    assert simplified.right.p == 0.2


def test_simplify_gamma_sum():
    simplified = (gamma(shape=2, scale=1) + gamma(shape=4, scale=1)).simplify()
    assert isinstance(simplified, GammaDistribution)
    assert simplified.shape == 6
    assert simplified.scale == 1

    # Cannot simplify if the scales are different
    simplified = (gamma(shape=2, scale=1) + gamma(shape=2, scale=2)).simplify()
    assert isinstance(simplified, ComplexDistribution)


def test_simplify_scale_gamma():
    simplified = (2 * gamma(shape=2, scale=3)).simplify()
    assert isinstance(simplified, GammaDistribution)
    assert simplified.shape == 2
    assert simplified.scale == 6


def test_simplify_exponential_sum():
    simplified = (exponential(scale=2) + exponential(scale=2)).simplify()
    assert isinstance(simplified, GammaDistribution)
    assert simplified.shape == 2
    assert simplified.scale == 2

    # Cannot simplify if the rates are different
    simplified = (exponential(scale=2) + exponential(scale=3)).simplify()
    assert isinstance(simplified, ComplexDistribution)


def test_simplify_exponential_gamma_sum():
    simplified = (
        5 * (exponential(scale=3) + gamma(shape=2, scale=3) + exponential(scale=3))
    ).simplify()
    assert isinstance(simplified, GammaDistribution)
    assert simplified.shape == 4
    assert simplified.scale == 15


def test_simplify_big_sum():
    simplified = (
        2 * norm(mean=1, sd=1)
        + lognorm(norm_mean=1, norm_sd=1) * lognorm(norm_mean=3, norm_sd=2)
        + gamma(shape=2, scale=3)
        + exponential(scale=3) / 5
        + exponential(scale=3)
        - norm(mean=2, sd=1)
    ).simplify()

    # simplifies to norm(0, sqrt(5)) + lognorm(4, sqrt(5)) + gamma(3, 3) + exponential(6)
    assert isinstance(simplified, ComplexDistribution)
    assert isinstance(simplified.right, ExponentialDistribution)
    assert simplified.right.scale == approx(3 / 5)
    assert isinstance(simplified.left, ComplexDistribution)
    assert isinstance(simplified.left.right, GammaDistribution)
    assert simplified.left.right.shape == 3
    assert simplified.left.right.scale == 3
    assert isinstance(simplified.left.left, ComplexDistribution)
    assert isinstance(simplified.left.left.right, LognormalDistribution)
    assert simplified.left.left.right.norm_mean == 4
    assert simplified.left.left.right.norm_sd == approx(np.sqrt(5))
    assert isinstance(simplified.left.left.left, NormalDistribution)
    assert simplified.left.left.left.mean == 0
    assert simplified.left.left.left.sd == approx(np.sqrt(5))


def test_preserve_non_commutative_op():
    simplified = (2**3 ** norm(mean=0, sd=1)).simplify()
    assert isinstance(simplified, ComplexDistribution)
    assert simplified.fn_str == "**"
    assert isinstance(simplified.left, Real)
    assert simplified.left == 2
    assert isinstance(simplified.right, ComplexDistribution)
    assert simplified.right.fn_str == "**"
    assert isinstance(simplified.right.left, Real)
    assert simplified.right.left == 3
    assert isinstance(simplified.right.right, NormalDistribution)
