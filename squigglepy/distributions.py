import operator
import math
import numpy as np
import scipy.stats

from typing import Optional, Union

from .utils import _process_weights_values, _is_numpy, is_dist, _round
from .version import __version__
from .correlation import CorrelationGroup

from collections.abc import Iterable

from abc import ABC, abstractmethod
from functools import reduce
from numbers import Real


class BaseDistribution(ABC):
    def __init__(self):
        self.x = None
        self.y = None
        self.n = None
        self.p = None
        self.t = None
        self.a = None
        self.b = None
        self.shape = None
        self.scale = None
        self.credibility = None
        self.mean = None
        self.sd = None
        self.left = None
        self.mode = None
        self.right = None
        self.fn = None
        self.fn_str = None
        self.lclip = None
        self.rclip = None
        self.lam = None
        self.df = None
        self.items = None
        self.dists = None
        self.weights = None
        self._version = __version__

        # Correlation metadata
        self.correlation_group: Optional[CorrelationGroup] = None
        self._correlated_samples: Optional[np.ndarray] = None

    @abstractmethod
    def __str__(self) -> str: ...

    def __repr__(self):
        if self.correlation_group:
            return (
                self.__str__() + f" (version {self._version}, corr_group {self.correlation_group})"
            )
        return self.__str__() + f" (version {self._version})"


class OperableDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()

    def plot(self, num_samples=None, bins=None):
        """
        Plot a histogram of the samples.

        Parameters
        ----------
        num_samples : int
            The number of samples to draw for plotting. Defaults to 1000 if not set.
        bins : int
            The number of bins to plot. Defaults to 200 if not set.

        Examples
        --------
        >>> sq.norm(5, 10).plot()
        """
        from matplotlib import pyplot as plt

        num_samples = 1000 if num_samples is None else num_samples
        bins = 200 if bins is None else bins

        samples = self @ num_samples

        plt.hist(samples, bins=bins)
        plt.show()

    def __invert__(self):
        from .samplers import sample

        return sample(self)

    def __matmul__(self, n):
        try:
            n = int(n)
        except ValueError:
            raise ValueError("number of samples must be an integer")
        from .samplers import sample

        return sample(self, n=n)

    def __rshift__(self, fn):
        if callable(fn):
            return fn(self)
        elif isinstance(fn, ComplexDistribution):
            return ComplexDistribution(self, fn.left, fn.fn, fn.fn_str, infix=False)
        else:
            raise ValueError

    def __rmatmul__(self, n):
        return self.__matmul__(n)

    def __gt__(self, dist):
        return ComplexDistribution(self, dist, operator.gt, ">")

    def __ge__(self, dist):
        return ComplexDistribution(self, dist, operator.ge, ">=")

    def __lt__(self, dist):
        return ComplexDistribution(self, dist, operator.lt, "<")

    def __le__(self, dist):
        return ComplexDistribution(self, dist, operator.le, "<=")

    def __eq__(self, dist):
        return ComplexDistribution(self, dist, operator.le, "==")

    def __ne__(self, dist):
        return ComplexDistribution(self, dist, operator.le, "!=")

    def __neg__(self):
        return ComplexDistribution(self, None, operator.neg, "-")

    def __add__(self, dist):
        return ComplexDistribution(self, dist, operator.add, "+")

    def __radd__(self, dist):
        return ComplexDistribution(dist, self, operator.add, "+")

    def __sub__(self, dist):
        return ComplexDistribution(self, dist, operator.sub, "-")

    def __rsub__(self, dist):
        return ComplexDistribution(dist, self, operator.sub, "-")

    def __mul__(self, dist):
        return ComplexDistribution(self, dist, operator.mul, "*")

    def __rmul__(self, dist):
        return ComplexDistribution(dist, self, operator.mul, "*")

    def __truediv__(self, dist):
        return ComplexDistribution(self, dist, operator.truediv, "/")

    def __rtruediv__(self, dist):
        return ComplexDistribution(dist, self, operator.truediv, "/")

    def __floordiv__(self, dist):
        return ComplexDistribution(self, dist, operator.floordiv, "//")

    def __rfloordiv__(self, dist):
        return ComplexDistribution(dist, self, operator.floordiv, "//")

    def __pow__(self, dist):
        return ComplexDistribution(self, dist, operator.pow, "**")

    def __rpow__(self, dist):
        return ComplexDistribution(dist, self, operator.pow, "**")

    def __hash__(self):
        return hash(repr(self))

    def simplify(self):
        """Simplify a distribution by evaluating all operations that can be
        performed analytically, for example by reducing a sum of normal
        distributions into a single normal distribution. Return a new
        ``OperableDistribution`` that represents the simplified distribution.

        Possible simplifications:
            any - any = any + (-any)
            any / constant = any * (1 / constant)

            -Normal = Normal

            Normal + Normal = Normal
            constant + Normal = Normal
            Bernoulli + Bernoulli = Binomial
            Bernoulli + Binomial = Binomial
            Binomial + Binomial = Binomial
            Chi-Square + Chi-Square = Chi-Square
            Exponential + Exponential = Exponential
            Exponential + Gamma = Gamma
            Gamma + Gamma = Gamma
            Poisson + Poisson = Poisson

            constant * Normal = Normal
            Lognormal * Lognormal = Lognormal
            constant * Lognormal = Lognormal
            constant * Exponential = Exponential
            constant * Gamma = Gamma

            Lognormal / Lognormal = Lognormal
            constant / Lognormal = Lognormal

            Lognormal ** constant = Lognormal

        """
        return self


# Distribution are either discrete, continuous, or composite


class DiscreteDistribution(OperableDistribution, ABC): ...


class ContinuousDistribution(OperableDistribution, ABC): ...


class CompositeDistribution(OperableDistribution):
    def __init__(self):
        super().__init__()
        # Whether this distribution contains any correlated variables
        self.contains_correlated: Optional[bool] = None

    def __post_init__(self):
        assert self.contains_correlated is not None, "contains_correlated must be set"

    def _check_correlated(self, dists: Iterable) -> None:
        for dist in dists:
            if isinstance(dist, BaseDistribution) and dist.correlation_group is not None:
                self.contains_correlated = True
                break
            if isinstance(dist, CompositeDistribution):
                if dist.contains_correlated:
                    self.contains_correlated = True
                    break


class ComplexDistribution(CompositeDistribution):
    def __init__(self, left, right=None, fn=operator.add, fn_str="+", infix=True):
        super().__init__()
        self.left = left
        self.right = right
        self.fn = fn
        self.fn_str = fn_str
        self.infix = infix
        self._check_correlated((left, right))

    def __str__(self):
        if self.right is None and self.infix:
            if self.fn_str == "-":
                out = "<Distribution> {}{}"
            else:
                out = "<Distribution> {} {}"
            out = out.format(self.fn_str, str(self.left).replace("<Distribution> ", ""))
        elif self.right is None and not self.infix:
            out = "<Distribution> {}({})".format(
                self.fn_str, str(self.left).replace("<Distribution> ", "")
            )
        elif self.right is not None and self.infix:
            out = "<Distribution> {} {} {}".format(
                str(self.left).replace("<Distribution> ", ""),
                self.fn_str,
                str(self.right).replace("<Distribution> ", ""),
            )
        elif self.right is not None and not self.infix:
            out = "<Distribution> {}({}, {})"
            out = out.format(
                self.fn_str,
                str(self.left).replace("<Distribution> ", ""),
                str(self.right).replace("<Distribution> ", ""),
            )
        else:
            raise ValueError
        return out

    def simplify(self):
        return FlatTree.build(self).simplify()


def _get_fname(f, name):
    if name is None:
        if isinstance(f, np.vectorize):
            name = f.pyfunc.__name__
        else:
            name = f.__name__
    return name


def dist_fn(dist1, dist2=None, fn=None, name=None):
    """
    Initialize a distribution that has a custom function applied to the result.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution or function or list
        Typically, the distribution to apply the function to. Could also be a function
        or list of functions if ``dist_fn`` is being used in a pipe.
    dist2 : Distribution or function or list or None
        Typically, the second distribution to apply the function to if the function takes
        two arguments. Could also be a function or list of functions if ``dist_fn`` is
        being used in a pipe.
    fn : function or None
        The function to apply to the distribution(s).
    name : str or None
        By default, ``fn.__name__`` will be used to name the function. But you can pass
        a custom name.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated
        when it is sampled.

    Examples
    --------
    >>> def double(x):
    >>>     return x * 2
    >>> dist_fn(norm(0, 1), double)
    <Distribution> double(norm(mean=0.5, sd=0.3))
    >>> norm(0, 1) >> dist_fn(double)
    <Distribution> double(norm(mean=0.5, sd=0.3))
    """
    if isinstance(dist1, list) and callable(dist1[0]) and dist2 is None and fn is None:
        fn = dist1

        def out_fn(d):
            out = d
            for f in fn:
                out = ComplexDistribution(out, None, fn=f, fn_str=_get_fname(f, name), infix=False)
            return out

        return out_fn

    if callable(dist1) and dist2 is None and fn is None:
        return lambda d: dist_fn(d, fn=dist1)

    if isinstance(dist2, list) and callable(dist2[0]) and fn is None:
        fn = dist2
        dist2 = None

    if callable(dist2) and fn is None:
        fn = dist2
        dist2 = None

    if not isinstance(fn, list):
        fn = [fn]

    out = dist1
    for f in fn:
        out = ComplexDistribution(out, dist2, fn=f, fn_str=_get_fname(f, name), infix=False)

    return out


def dist_max(dist1, dist2=None):
    """
    Initialize the calculation of the maximum value of two distributions.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and determine the max of.
    dist2 : Distribution
        The second distribution to sample and determine the max of.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated
        when it is sampled.

    Examples
    --------
    >>> dist_max(norm(0, 1), norm(1, 2))
    <Distribution> max(norm(mean=0.5, sd=0.3), norm(mean=1.5, sd=0.3))
    """
    if is_dist(dist1) and dist2 is None:
        return lambda d: dist_fn(d, dist1, np.maximum, name="max")
    else:
        return dist_fn(dist1, dist2, np.maximum, name="max")


def dist_min(dist1, dist2=None):
    """
    Initialize the calculation of the minimum value of two distributions.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and determine the min of.
    dist2 : Distribution
        The second distribution to sample and determine the min of.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> dist_min(norm(0, 1), norm(1, 2))
    <Distribution> min(norm(mean=0.5, sd=0.3), norm(mean=1.5, sd=0.3))
    """
    if is_dist(dist1) and dist2 is None:
        return lambda d: dist_fn(d, dist1, np.minimum, name="min")
    else:
        return dist_fn(dist1, dist2, np.minimum, name="min")


def dist_round(dist1, digits=0):
    """
    Initialize the rounding of the output of the distribution.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and then round.
    digits : int
        The number of digits to round to.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> dist_round(norm(0, 1))
    <Distribution> round(norm(mean=0.5, sd=0.3), 0)
    """
    if isinstance(dist1, int) and digits == 0:
        return lambda d: dist_round(d, digits=dist1)
    else:
        return dist_fn(dist1, digits, _round, name="round")


def dist_ceil(dist1):
    """
    Initialize the ceiling rounding of the output of the distribution.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and then ceiling round.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> dist_ceil(norm(0, 1))
    <Distribution> ceil(norm(mean=0.5, sd=0.3))
    """
    return dist_fn(dist1, None, np.ceil)


def dist_floor(dist1):
    """
    Initialize the floor rounding of the output of the distribution.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and then floor round.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> dist_floor(norm(0, 1))
    <Distribution> floor(norm(mean=0.5, sd=0.3))
    """
    return dist_fn(dist1, None, np.floor)


def dist_log(dist1, base=math.e):
    """
    Initialize the log of the output of the distribution.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and then take the log of.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> dist_log(norm(0, 1), 10)
    <Distribution> log(norm(mean=0.5, sd=0.3), const(10))
    """
    return dist_fn(dist1, const(base), math.log)


def dist_exp(dist1):
    """
    Initialize the exp of the output of the distribution.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution
        The distribution to sample and then take the exp of.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> dist_exp(norm(0, 1))
    <Distribution> exp(norm(mean=0.5, sd=0.3))
    """
    return dist_fn(dist1, None, math.exp)


@np.vectorize
def _lclip(n, val=None):
    if val is None:
        return n
    else:
        return val if n < val else n


def lclip(dist1, val=None):
    """
    Initialize the clipping/bounding of the output of the distribution by the lower value.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution or function
        The distribution to clip. If this is a funciton, it will return a partial that will
        be suitable for use in piping.
    val : int or float or None
        The value to use as the lower bound for clipping.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> lclip(norm(0, 1), 0.5)
    <Distribution> lclip(norm(mean=0.5, sd=0.3), 0.5)
    """
    if (isinstance(dist1, int) or isinstance(dist1, float)) and val is None:
        return lambda d: lclip(d, dist1)
    elif is_dist(dist1):
        return dist_fn(dist1, val, _lclip, name="lclip")
    else:
        return _lclip(dist1, val)


@np.vectorize
def _rclip(n, val=None):
    if val is None:
        return n
    else:
        return val if n > val else n


def rclip(dist1, val=None):
    """
    Initialize the clipping/bounding of the output of the distribution by the upper value.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution or function
        The distribution to clip. If this is a funciton, it will return a partial that will
        be suitable for use in piping.
    val : int or float or None
        The value to use as the upper bound for clipping.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> rclip(norm(0, 1), 0.5)
    <Distribution> rclip(norm(mean=0.5, sd=0.3), 0.5)
    """
    if (isinstance(dist1, int) or isinstance(dist1, float)) and val is None:
        return lambda d: rclip(d, dist1)
    elif is_dist(dist1):
        return dist_fn(dist1, val, _rclip, name="rclip")
    else:
        return _rclip(dist1, val)


def clip(dist1, left, right=None):
    """
    Initialize the clipping/bounding of the output of the distribution.

    The function won't be applied until the distribution is sampled.

    Parameters
    ----------
    dist1 : Distribution or function
        The distribution to clip. If this is a funciton, it will return a partial that will
        be suitable for use in piping.
    left : int or float or None
        The value to use as the lower bound for clipping.
    right : int or float or None
        The value to use as the upper bound for clipping.

    Returns
    -------
    ComplexDistribution or function
        This will be a lazy evaluation of the desired function that will then be calculated

    Examples
    --------
    >>> clip(norm(0, 1), 0.5, 0.9)
    <Distribution> rclip(lclip(norm(mean=0.5, sd=0.3), 0.5), 0.9)
    """
    if (
        (isinstance(dist1, int) or isinstance(dist1, float))
        and (isinstance(left, int) or isinstance(left, float))
        and right is None
    ):
        return lambda d: rclip(lclip(d, dist1), left)
    else:
        return rclip(lclip(dist1, left), right)


class ConstantDistribution(DiscreteDistribution):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def __str__(self):
        return "<Distribution> const({})".format(self.x)


def const(x):
    """
    Initialize a constant distribution.

    Constant distributions always return the same value no matter what.

    Parameters
    ----------
    x : anything
        The value the constant distribution should always return.

    Returns
    -------
    ConstantDistribution

    Examples
    --------
    >>> const(1)
    <Distribution> const(1)
    """
    return ConstantDistribution(x)


class UniformDistribution(ContinuousDistribution):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        assert x < y, "x must be less than y"

    def __str__(self):
        return "<Distribution> uniform({}, {})".format(self.x, self.y)


def uniform(x, y):
    """
    Initialize a uniform random distribution.

    Parameters
    ----------
    x : float
        The smallest value the uniform distribution will return.
    y : float
        The largest value the uniform distribution will return.

    Returns
    -------
    UniformDistribution

    Examples
    --------
    >>> uniform(0, 1)
    <Distribution> uniform(0, 1)
    """
    return UniformDistribution(x=x, y=y)


class NormalDistribution(ContinuousDistribution):
    def __init__(self, x=None, y=None, mean=None, sd=None, credibility=90, lclip=None, rclip=None):
        super().__init__()
        self.x = x
        self.y = y
        self.credibility = credibility
        self.mean = mean
        self.sd = sd
        self.lclip = lclip
        self.rclip = rclip

        if self.x is not None and self.y is not None and self.x > self.y:
            raise ValueError("`high value` cannot be lower than `low value`")

        if (self.x is None or self.y is None) and self.sd is None:
            raise ValueError("must define either x/y or mean/sd")
        elif (self.x is not None or self.y is not None) and self.sd is not None:
            raise ValueError("must define either x/y or mean/sd -- cannot define both")
        elif self.sd is not None and self.mean is None:
            self.mean = 0

        if self.mean is None and self.sd is None:
            self.mean = (self.x + self.y) / 2
            cdf_value = 0.5 + 0.5 * (self.credibility / 100)
            normed_sigma = scipy.stats.norm.ppf(cdf_value)
            self.sd = (self.y - self.mean) / normed_sigma

    def __str__(self):
        out = "<Distribution> norm(mean={}, sd={}".format(round(self.mean, 2), round(self.sd, 2))
        if self.lclip is not None:
            out += ", lclip={}".format(self.lclip)
        if self.rclip is not None:
            out += ", rclip={}".format(self.rclip)
        out += ")"
        return out


def norm(
    x=None, y=None, credibility=90, mean=None, sd=None, lclip=None, rclip=None
) -> NormalDistribution:
    """
    Initialize a normal distribution.

    Can be defined either via a credible interval from ``x`` to ``y`` (use ``credibility`` or
    it will default to being a 90% CI) or defined via ``mean`` and ``sd``.

    Parameters
    ----------
    x : float
        The low value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    y : float
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    credibility : float
        The range of the credibility interval. Defaults to 90. Ignored if the distribution is
        defined instead by ``mean`` and ``sd``.
    mean : float or None
        The mean of the normal distribution. If not defined, defaults to 0.
    sd : float
        The standard deviation of the normal distribution.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    NormalDistribution

    Examples
    --------
    >>> norm(0, 1)
    <Distribution> norm(mean=0.5, sd=0.3)
    >>> norm(mean=1, sd=2)
    <Distribution> norm(mean=1, sd=2)
    """
    return NormalDistribution(
        x=x, y=y, credibility=credibility, mean=mean, sd=sd, lclip=lclip, rclip=rclip
    )


class LognormalDistribution(ContinuousDistribution):
    def __init__(
        self,
        x=None,
        y=None,
        norm_mean=None,
        norm_sd=None,
        lognorm_mean=None,
        lognorm_sd=None,
        credibility=90,
        lclip=None,
        rclip=None,
    ):
        super().__init__()
        self.x = x
        self.y = y
        self.credibility = credibility
        self.norm_mean = norm_mean
        self.norm_sd = norm_sd
        self.lognorm_mean = lognorm_mean
        self.lognorm_sd = lognorm_sd
        self.lclip = lclip
        self.rclip = rclip

        if self.x is not None and self.y is not None and self.x > self.y:
            raise ValueError("`high value` cannot be lower than `low value`")
        if self.x is not None and self.x <= 0:
            raise ValueError("lognormal distribution must have values > 0")

        if (self.x is None or self.y is None) and self.norm_sd is None and self.lognorm_sd is None:
            raise ValueError(
                ("must define only one of x/y, norm_mean/norm_sd, " "or lognorm_mean/lognorm_sd")
            )
        elif (self.x is not None or self.y is not None) and (
            self.norm_sd is not None or self.lognorm_sd is not None
        ):
            raise ValueError(
                ("must define only one of x/y, norm_mean/norm_sd, " "or lognorm_mean/lognorm_sd")
            )
        elif (self.norm_sd is not None or self.norm_mean is not None) and (
            self.lognorm_sd is not None or self.lognorm_mean is not None
        ):
            raise ValueError(
                ("must define only one of x/y, norm_mean/norm_sd, " "or lognorm_mean/lognorm_sd")
            )
        elif self.norm_sd is not None and self.norm_mean is None:
            self.norm_mean = 0
        elif self.lognorm_sd is not None and self.lognorm_mean is None:
            self.lognorm_mean = 1

        if self.x is not None:
            self.norm_mean = (np.log(self.x) + np.log(self.y)) / 2
            cdf_value = 0.5 + 0.5 * (self.credibility / 100)
            normed_sigma = scipy.stats.norm.ppf(cdf_value)
            self.norm_sd = (np.log(self.y) - self.norm_mean) / normed_sigma

        if self.lognorm_sd is None:
            self.lognorm_mean = np.exp(self.norm_mean + self.norm_sd**2 / 2)
            self.lognorm_sd = (
                (np.exp(self.norm_sd**2) - 1) * np.exp(2 * self.norm_mean + self.norm_sd**2)
            ) ** 0.5
        elif self.norm_sd is None:
            self.norm_mean = np.log(
                (self.lognorm_mean**2 / np.sqrt(self.lognorm_sd**2 + self.lognorm_mean**2))
            )
            self.norm_sd = np.sqrt(np.log(1 + self.lognorm_sd**2 / self.lognorm_mean**2))

    def __str__(self):
        out = "<Distribution> lognorm(lognorm_mean={}, lognorm_sd={}, norm_mean={}, norm_sd={}"
        out = out.format(
            round(self.lognorm_mean, 2),
            round(self.lognorm_sd, 2),
            round(self.norm_mean, 2),
            round(self.norm_sd, 2),
        )
        if self.lclip is not None:
            out += ", lclip={}".format(self.lclip)
        if self.rclip is not None:
            out += ", rclip={}".format(self.rclip)
        out += ")"
        return out


def lognorm(
    x=None,
    y=None,
    credibility=90,
    norm_mean=None,
    norm_sd=None,
    lognorm_mean=None,
    lognorm_sd=None,
    lclip=None,
    rclip=None,
):
    """
    Initialize a lognormal distribution.

    Can be defined either via a credible interval from ``x`` to ``y`` (use ``credibility`` or
    it will default to being a 90% CI) or defined via ``mean`` and ``sd``.

    Parameters
    ----------
    x : float
        The low value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
        Must be a value greater than 0.
    y : float
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
        Must be a value greater than 0.
    credibility : float
        The range of the credibility interval. Defaults to 90. Ignored if the distribution is
        defined instead by ``mean`` and ``sd``.
    norm_mean : float or None
        The mean of the underlying normal distribution. If not defined, defaults to 0.
    norm_sd : float
        The standard deviation of the underlying normal distribution.
    lognorm_mean : float or None
        The mean of the lognormal distribution. If not defined, defaults to 1.
    lognorm_sd : float
        The standard deviation of the lognormal distribution.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    LognormalDistribution

    Examples
    --------
    >>> lognorm(1, 10)
    <Distribution> lognorm(lognorm_mean=4.04, lognorm_sd=3.21, norm_mean=1.15, norm_sd=0.7)
    >>> lognorm(norm_mean=1, norm_sd=2)
    <Distribution> lognorm(lognorm_mean=20.09, lognorm_sd=147.05, norm_mean=1, norm_sd=2)
    >>> lognorm(lognorm_mean=1, lognorm_sd=2)
    <Distribution> lognorm(lognorm_mean=1, lognorm_sd=2, norm_mean=-0.8, norm_sd=1.27)
    """
    return LognormalDistribution(
        x=x,
        y=y,
        credibility=credibility,
        norm_mean=norm_mean,
        norm_sd=norm_sd,
        lognorm_mean=lognorm_mean,
        lognorm_sd=lognorm_sd,
        lclip=lclip,
        rclip=rclip,
    )


def to(
    x, y, credibility=90, lclip=None, rclip=None
) -> Union[LognormalDistribution, NormalDistribution]:
    """
    Initialize a distribution from ``x`` to ``y``.

    The distribution will be lognormal by default, unless ``x`` is less than or equal to 0,
    in which case it will become a normal distribution.

    The distribution will default to be a 90% credible interval between ``x`` and ``y`` unless
    ``credibility`` is passed.

    Parameters
    ----------
    x : float
        The low value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    y : float
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    credibility : float
        The range of the credibility interval. Defaults to 90.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    ``LognormalDistribution`` if ``x`` > 0, otherwise a ``NormalDistribution``

    Examples
    --------
    >>> to(1, 10)
    <Distribution> lognorm(mean=1.15, sd=0.7)
    >>> to(-10, 10)
    <Distribution> norm(mean=0.0, sd=6.08)
    """
    if x > 0:
        return lognorm(x=x, y=y, credibility=credibility, lclip=lclip, rclip=rclip)
    else:
        return norm(x=x, y=y, credibility=credibility, lclip=lclip, rclip=rclip)


class BinomialDistribution(DiscreteDistribution):
    def __init__(self, n, p):
        super().__init__()
        self.n = n
        self.p = p
        if self.p <= 0 or self.p >= 1:
            raise ValueError("p must be between 0 and 1 (exclusive)")

    def __str__(self):
        return "<Distribution> binomial(n={}, p={})".format(self.n, self.p)


def binomial(n, p):
    """
    Initialize a binomial distribution.

    Parameters
    ----------
    n : int
        The number of trials.
    p : float
        The probability of success for each trial. Must be between 0 and 1.

    Returns
    -------
    BinomialDistribution

    Examples
    --------
    >>> binomial(1, 0.1)
    <Distribution> binomial(1, 0.1)
    """
    return BinomialDistribution(n=n, p=p)


class BetaDistribution(ContinuousDistribution):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def __str__(self):
        return "<Distribution> beta(a={}, b={})".format(self.a, self.b)


def beta(a, b):
    """
    Initialize a beta distribution.

    Parameters
    ----------
    a : float
        The alpha shape value of the distribution. Typically takes the value of the
        number of trials that resulted in a success.
    b : float
        The beta shape value of the distribution. Typically takes the value of the
        number of trials that resulted in a failure.

    Returns
    -------
    BetaDistribution

    Examples
    --------
    >>> beta(1, 2)
    <Distribution> beta(1, 2)
    """
    return BetaDistribution(a, b)


class BernoulliDistribution(DiscreteDistribution):
    def __init__(self, p):
        super().__init__()
        if not isinstance(p, float) or isinstance(p, int):
            raise ValueError("bernoulli p must be a float or int")
        if p <= 0 or p >= 1:
            raise ValueError("bernoulli p must be 0-1 (exclusive)")
        self.p = p

    def __str__(self):
        return "<Distribution> bernoulli(p={})".format(self.p)


def bernoulli(p):
    """
    Initialize a Bernoulli distribution.

    Parameters
    ----------
    p : float
        The probability of the binary event. Must be between 0 and 1.

    Returns
    -------
    BernoulliDistribution

    Examples
    --------
    >>> bernoulli(0.1)
    <Distribution> bernoulli(p=0.1)
    """
    return BernoulliDistribution(p)


class CategoricalDistribution(DiscreteDistribution):
    def __init__(self, items):
        super().__init__()
        if not isinstance(items, dict) and not isinstance(items, list) and not _is_numpy(items):
            raise ValueError("inputs to categorical must be a dict or list")
        assert len(items) > 0, "inputs to categorical must be non-empty"
        self.items = list(items) if _is_numpy(items) else items

    def __str__(self):
        return "<Distribution> categorical({})".format(self.items)


def discrete(items):
    """
    Initialize a discrete distribution (aka categorical distribution).

    Parameters
    ----------
    items : list or dict
        The values that the discrete distribution will return and their associated
        weights (or likelihoods of being returned when sampled).

    Returns
    -------
    CategoricalDistribution

    Examples
    --------
    >>> discrete({0: 0.1, 1: 0.9})  # 10% chance of returning 0, 90% chance of returning 1
    <Distribution> categorical({0: 0.1, 1: 0.9})
    >>> discrete([[0.1, 0], [0.9, 1]])  # Different notation for the same thing.
    <Distribution> categorical([[0.1, 0], [0.9, 1]])
    >>> discrete([0, 1, 2])  # When no weights are given, all have equal chance of happening.
    <Distribution> categorical([0, 1, 2])
    >>> discrete({'a': 0.1, 'b': 0.9})  # Values do not have to be numbers.
    <Distribution> categorical({'a': 0.1, 'b': 0.9})
    """
    return CategoricalDistribution(items)


class TDistribution(ContinuousDistribution):
    def __init__(self, x=None, y=None, t=20, credibility=90, lclip=None, rclip=None):
        super().__init__()
        self.x = x
        self.y = y
        self.t = t
        self.df = t
        self.credibility = credibility
        self.lclip = lclip
        self.rclip = rclip

        if (self.x is None or self.y is None) and not (self.x is None and self.y is None):
            raise ValueError("must define either both `x` and `y` or neither.")
        elif self.x is not None and self.y is not None and self.x > self.y:
            raise ValueError("`high value` cannot be lower than `low value`")

        if self.x is None:
            self.credibility = None

    def __str__(self):
        if self.x is not None:
            out = "<Distribution> tdist(x={}, y={}, t={}".format(self.x, self.y, self.t)
        else:
            out = "<Distribution> tdist(t={}".format(self.t)
        if self.credibility != 90 and self.credibility is not None:
            out += ", credibility={}".format(self.credibility)
        if self.lclip is not None:
            out += ", lclip={}".format(self.lclip)
        if self.rclip is not None:
            out += ", rclip={}".format(self.rclip)
        out += ")"
        return out


def tdist(x=None, y=None, t=20, credibility=90, lclip=None, rclip=None):
    """
    Initialize a t-distribution.

    Is defined either via a loose credible interval from ``x`` to ``y`` (use ``credibility`` or
    it will default to being a 90% CI). Unlike the normal and lognormal distributions, this
    credible interval is an approximation and is not precisely defined.

    If ``x`` and ``y`` are not defined, can just return a classic t-distribution defined via
    ``t`` as the number of degrees of freedom.

    Parameters
    ----------
    x : float or None
        The low value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    y : float or None
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    t : float
        The number of degrees of freedom of the t-distribution. Defaults to 20.
    credibility : float
        The range of the credibility interval. Defaults to 90.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    TDistribution

    Examples
    --------
    >>> tdist(0, 1, 2)
    <Distribution> tdist(x=0, y=1, t=2)
    >>> tdist()
    <Distribution> tdist(t=1)
    """
    return TDistribution(x=x, y=y, t=t, credibility=credibility, lclip=lclip, rclip=rclip)


class LogTDistribution(ContinuousDistribution):
    def __init__(self, x=None, y=None, t=1, credibility=90, lclip=None, rclip=None):
        super().__init__()
        self.x = x
        self.y = y
        self.t = t
        self.df = t
        self.credibility = credibility
        self.lclip = lclip
        self.rclip = rclip

        if (self.x is None or self.y is None) and not (self.x is None and self.y is None):
            raise ValueError("must define either both `x` and `y` or neither.")
        if self.x is not None and self.y is not None and self.x > self.y:
            raise ValueError("`high value` cannot be lower than `low value`")
        if self.x is not None and self.x <= 0:
            raise ValueError("`low value` must be greater than 0.")

        if self.x is None:
            self.credibility = None

    def __str__(self):
        if self.x is not None:
            out = "<Distribution> log_tdist(x={}, y={}, t={}".format(self.x, self.y, self.t)
        else:
            out = "<Distribution> log_tdist(t={}".format(self.t)
        if self.credibility != 90 and self.credibility is not None:
            out += ", credibility={}".format(self.credibility)
        if self.lclip is not None:
            out += ", lclip={}".format(self.lclip)
        if self.rclip is not None:
            out += ", rclip={}".format(self.rclip)
        out += ")"
        return out


def log_tdist(x=None, y=None, t=1, credibility=90, lclip=None, rclip=None):
    """
    Initialize a log t-distribution, which is a t-distribution in log-space.

    Is defined either via a loose credible interval from ``x`` to ``y`` (use ``credibility`` or
    it will default to being a 90% CI). Unlike the normal and lognormal distributions, this
    credible interval is an approximation and is not precisely defined.

    If ``x`` and ``y`` are not defined, can just return a classic t-distribution defined via
    ``t`` as the number of degrees of freedom, but in log-space.

    Parameters
    ----------
    x : float or None
        The low value of a credible interval defined by ``credibility``. Must be greater than 0.
        Defaults to a 90% CI.
    y : float or None
        The high value of a credible interval defined by ``credibility``. Defaults to a 90% CI.
    t : float
        The number of degrees of freedom of the t-distribution. Defaults to 1.
    credibility : float
        The range of the credibility interval. Defaults to 90.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    LogTDistribution

    Examples
    --------
    >>> log_tdist(0, 1, 2)
    <Distribution> log_tdist(x=0, y=1, t=2)
    >>> log_tdist()
    <Distribution> log_tdist(t=1)
    """
    return LogTDistribution(x=x, y=y, t=t, credibility=credibility, lclip=lclip, rclip=rclip)


class TriangularDistribution(ContinuousDistribution):
    def __init__(self, left, mode, right):
        super().__init__()
        if left > mode:
            raise ValueError("left must be less than or equal to mode")
        if right < mode:
            raise ValueError("right must be greater than or equal to mode")
        if left == right:
            raise ValueError("left and right must be different")
        self.left = left
        self.mode = mode
        self.right = right

    def __str__(self):
        return "<Distribution> triangular({}, {}, {})".format(self.left, self.mode, self.right)


def triangular(left, mode, right, lclip=None, rclip=None):
    """
    Initialize a triangular distribution.

    Parameters
    ----------
    left : float
        The smallest value of the triangular distribution.
    mode : float
        The most common value of the triangular distribution.
    right : float
        The largest value of the triangular distribution.

    Returns
    -------
    TriangularDistribution

    Examples
    --------
    >>> triangular(1, 2, 3)
    <Distribution> triangular(1, 2, 3)
    """
    return TriangularDistribution(left=left, mode=mode, right=right)


class PERTDistribution(ContinuousDistribution):
    def __init__(self, left, mode, right, lam=4, lclip=None, rclip=None):
        super().__init__()
        if left > mode:
            raise ValueError("left must be less than or equal to mode")
        if right < mode:
            raise ValueError("right must be greater than or equal to mode")
        if lam < 0:
            raise ValueError("the shape parameter must be positive")
        if left == right:
            raise ValueError("left and right must be different")

        self.left = left
        self.mode = mode
        self.right = right
        self.lam = lam
        self.lclip = lclip
        self.rclip = rclip

    def __str__(self):
        out = "<Distribution> PERT({}, {}, {}, lam={}".format(
            self.left, self.mode, self.right, self.lam
        )
        if self.lclip is not None:
            out += ", lclip={}".format(self.lclip)
        if self.rclip is not None:
            out += ", rclip={}".format(self.rclip)
        out += ")"
        return out


def pert(left, mode, right, lam=4, lclip=None, rclip=None):
    """
    Initialize a PERT distribution.

    Parameters
    ----------
    left : float
        The smallest value of the PERT distribution.
    mode : float
        The most common value of the PERT distribution.
    right : float
        The largest value of the PERT distribution.
    lam : float
        The lambda value of the PERT distribution. Defaults to 4.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    PERTDistribution

    Examples
    --------
    >>> pert(1, 2, 3)
    <Distribution> PERT(1, 2, 3)
    """
    return PERTDistribution(left=left, mode=mode, right=right, lam=lam, lclip=lclip, rclip=rclip)


class PoissonDistribution(DiscreteDistribution):
    def __init__(self, lam, lclip=None, rclip=None):
        super().__init__()
        self.lam = lam
        self.lclip = lclip
        self.rclip = rclip

    def __str__(self):
        out = "<Distribution> poisson({}".format(self.lam)
        if self.lclip is not None:
            out += ", lclip={}".format(self.lclip)
        if self.rclip is not None:
            out += ", rclip={}".format(self.rclip)
        out += ")"
        return out


def poisson(lam, lclip=None, rclip=None):
    """
    Initialize a poisson distribution.

    Parameters
    ----------
    lam : float
        The lambda value of the poisson distribution.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    PoissonDistribution

    Examples
    --------
    >>> poisson(1)
    <Distribution> poisson(1)
    """
    return PoissonDistribution(lam=lam, lclip=lclip, rclip=rclip)


class ChiSquareDistribution(ContinuousDistribution):
    def __init__(self, df):
        super().__init__()
        self.df = df
        if self.df <= 0:
            raise ValueError("df must be positive")

    def __str__(self):
        return "<Distribution> chisquare({})".format(self.df)


def chisquare(df):
    """
    Initialize a chi-square distribution.

    Parameters
    ----------
    df : float
        The degrees of freedom. Must be positive.

    Returns
    -------
    ChiSquareDistribution

    Examples
    --------
    >>> chisquare(2)
    <Distribution> chiaquare(2)
    """
    return ChiSquareDistribution(df=df)


class ExponentialDistribution(ContinuousDistribution):
    def __init__(self, scale, lclip=None, rclip=None):
        super().__init__()
        assert scale > 0, "scale must be positive"
        # Prevent numeric overflows
        assert scale < 1e20, "scale must be less than 1e20"
        self.scale = scale
        self.lclip = lclip
        self.rclip = rclip

    def __str__(self):
        out = "<Distribution> exponential({}".format(self.scale)
        if self.lclip is not None:
            out += ", lclip={}".format(self.lclip)
        if self.rclip is not None:
            out += ", rclip={}".format(self.rclip)
        out += ")"
        return out


def exponential(scale, lclip=None, rclip=None):
    """
    Initialize an exponential distribution.

    Parameters
    ----------
    scale : float
        The scale value of the exponential distribution (> 0)
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    ExponentialDistribution

    Examples
    --------
    >>> exponential(1)
    <Distribution> exponential(1)
    """
    return ExponentialDistribution(scale=scale, lclip=lclip, rclip=rclip)


class GammaDistribution(ContinuousDistribution):
    def __init__(self, shape, scale=1, lclip=None, rclip=None):
        super().__init__()
        self.shape = shape
        self.scale = scale
        self.lclip = lclip
        self.rclip = rclip

    def __str__(self):
        out = "<Distribution> gamma(shape={}, scale={}".format(self.shape, self.scale)
        if self.lclip is not None:
            out += ", lclip={}".format(self.lclip)
        if self.rclip is not None:
            out += ", rclip={}".format(self.rclip)
        out += ")"
        return out


def gamma(shape, scale=1, lclip=None, rclip=None):
    """
    Initialize a gamma distribution.

    Parameters
    ----------
    shape : float
        The shape value of the gamma distribution.
    scale : float
        The scale value of the gamma distribution. Defaults to 1.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    GammaDistribution

    Examples
    --------
    >>> gamma(10, 1)
    <Distribution> gamma(shape=10, scale=1)
    """
    return GammaDistribution(shape=shape, scale=scale, lclip=lclip, rclip=rclip)


class ParetoDistribution(ContinuousDistribution):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __str__(self):
        return "<Distribution> pareto({})".format(self.shape)


def pareto(shape):
    """
    Initialize a pareto distribution.

    Parameters
    ----------
    shape : float
        The shape value of the pareto distribution.

    Returns
    -------
    ParetoDistribution

    Examples
    --------
    >>> pareto(1)
    <Distribution> pareto(1)
    """
    return ParetoDistribution(shape=shape)


class MixtureDistribution(CompositeDistribution):
    def __init__(self, dists, weights=None, relative_weights=None, lclip=None, rclip=None):
        super().__init__()
        weights, dists = _process_weights_values(weights, relative_weights, dists)
        self.dists = dists
        self.weights = weights
        self.lclip = lclip
        self.rclip = rclip
        self._check_correlated(dists)

    def __str__(self):
        out = "<Distribution> mixture"
        for i in range(len(self.dists)):
            out += "\n - {} weight on {}".format(self.weights[i], self.dists[i])
        return out


def mixture(dists, weights=None, relative_weights=None, lclip=None, rclip=None):
    """
    Initialize a mixture distribution, which is a combination of different distributions.

    Parameters
    ----------
    dists : list or dict
        The distributions to mix. Can also be defined as a list of weights and distributions.
    weights : list or None
        The weights for each distribution.
    relative_weights : list or None
        Relative weights, which if given will be weights that are normalized
        to sum to 1.
    lclip : float or None
        If not None, any value below ``lclip`` will be coerced to ``lclip``.
    rclip : float or None
        If not None, any value below ``rclip`` will be coerced to ``rclip``.

    Returns
    -------
    MixtureDistribution

    Examples
    --------
    >>> mixture([norm(1, 2), norm(3, 4)], weights=[0.1, 0.9])
    <Distribution> mixture
     - <Distribution> norm(mean=1.5, sd=0.3)
     - <Distribution> norm(mean=3.5, sd=0.3)
    >>> mixture([[0.1, norm(1, 2)], [0.9, norm(3, 4)]])  # Different notation for the same thing.
    <Distribution> mixture
     - <Distribution> norm(mean=1.5, sd=0.3)
     - <Distribution> norm(mean=3.5, sd=0.3)
    >>> mixture([norm(1, 2), norm(3, 4)])  # When no weights are given, all have equal chance
    >>>                                    # of happening.
    <Distribution> mixture
     - <Distribution> norm(mean=1.5, sd=0.3)
     - <Distribution> norm(mean=3.5, sd=0.3)
    """
    return MixtureDistribution(
        dists=dists,
        weights=weights,
        relative_weights=relative_weights,
        lclip=lclip,
        rclip=rclip,
    )


def zero_inflated(p_zero, dist):
    """
    Initialize an arbitrary zero-inflated distribution.

    Parameters
    ----------
    p_zero : float
        The chance of the distribution returning zero
    dist : Distribution
        The distribution to sample from when not zero

    Returns
    -------
    MixtureDistribution

    Examples
    --------
    >>> zero_inflated(0.6, norm(1, 2))
    <Distribution> mixture
     - 0
     - <Distribution> norm(mean=1.5, sd=0.3)
    """
    if p_zero > 1 or p_zero < 0 or not isinstance(p_zero, float):
        raise ValueError("`p_zero` must be between 0 and 1")
    return MixtureDistribution(dists=[0, dist], weights=p_zero)


def inf0(p_zero, dist):
    """
    Initialize an arbitrary zero-inflated distribution.

    Alias for ``zero_inflated``.

    Parameters
    ----------
    p_zero : float
        The chance of the distribution returning zero
    dist : Distribution
        The distribution to sample from when not zero

    Returns
    -------
    MixtureDistribution

    Examples
    --------
    >>> inf0(0.6, norm(1, 2))
    <Distribution> mixture
     - 0
     - <Distribution> norm(mean=1.5, sd=0.3)
    """
    return zero_inflated(p_zero=p_zero, dist=dist)


class GeometricDistribution(OperableDistribution):
    def __init__(self, p):
        super().__init__()
        self.p = p
        if self.p < 0 or self.p > 1:
            raise ValueError("p must be between 0 and 1")

    def __str__(self):
        return "<Distribution> geometric(p={})".format(self.p)


def geometric(p):
    """
    Initialize a geometric distribution.

    Parameters
    ----------
    p : float
        The probability of success of an individual trial. Must be between 0 and 1.

    Returns
    -------
    GeometricDistribution

    Examples
    --------
    >>> geometric(0.1)
    <Distribution> geometric(0.1)
    """
    return GeometricDistribution(p=p)


class FlatTree:
    """Helper class for simplifying analytic expressions. A ``FlatTree`` is
    sort of like a ``ComplexDistribution`` except that it flattens
    commutative/associative operations onto a single object instead of having
    one object per binary operation.

    This class operates in two phases.

    Phase 1: Generate a ``FlatTree`` object from a :ref:``BaseDistribution`` by
    calling ``FlatTree.build(dist)``. This generates a tree where any series of
    a single commutative/associative operation done repeatedly is flattened
    onto a single ``FlatTree`` node. It also converts operations into a
    normalized form, for example converting ``a - b`` into ``a + (-b)``.

    Phase 2: Generate a simplified ``Distribution`` by calling
    :ref:``simplify``. This works by combing through each flat list of
    distributions to find which ones can be analytically simplified (for
    example, converting a sum of normal distributions into a single normal
    distribution).

    """

    COMMUTABLE_OPERATIONS = set([operator.add, operator.mul])

    def __init__(self, dist=None, fn=None, fn_str=None, children=None, is_unary=False, infix=None):
        self.dist = dist
        self.fn = fn
        self.fn_str = fn_str
        self.children = children
        self.is_unary = is_unary
        self.infix = infix
        if dist is not None:
            self.is_leaf = True
        elif fn is not None and children is not None:
            self.is_leaf = False
        else:
            raise ValueError("Missing arguments to FlatTree constructor")

    def __str__(self):
        if self.is_leaf:
            return f"FlatTree({self.dist})"
        else:
            return "FlatTree({})[{}]".format(self.fn_str, ", ".join(map(str, self.children)))

    def __repr__(self):
        return str(self)

    @classmethod
    def build(cls, dist):
        if dist is None:
            return None
        if isinstance(dist, Real):
            return cls(dist=dist)
        if not isinstance(dist, BaseDistribution):
            raise ValueError(f"dist must be a BaseDistribution or numeric type, not {type(dist)}")
        if not isinstance(dist, ComplexDistribution):
            return cls(dist=dist)

        is_unary = dist.right is None
        if is_unary and dist.right is not None:
            raise ValueError(f"Multiple arguments provided for unary operator {dist.fn}")

        # Convert x - y into x + (-y)
        if dist.fn == operator.sub:
            return cls.build(
                ComplexDistribution(
                    dist.left,
                    ComplexDistribution(dist.right, right=None, fn=operator.neg, fn_str="-"),
                    fn=operator.add,
                    fn_str="+",
                )
            )

        # If the denominator is a constant, replace division by constant
        # with multiplication by the reciprocal of the constant
        if dist.fn == operator.truediv and isinstance(dist.right, Real):
            if dist.right == 0:
                raise ZeroDivisionError("Division by zero in ComplexDistribution: {dist}")
            return cls.build(
                ComplexDistribution(
                    dist.left,
                    1 / dist.right,
                    fn=operator.mul,
                    fn_str="*",
                )
            )
        if dist.fn == operator.truediv and isinstance(dist.right, LognormalDistribution):
            return cls.build(
                ComplexDistribution(
                    dist.left,
                    LognormalDistribution(
                        norm_mean=-dist.right.norm_mean, norm_sd=dist.right.norm_sd
                    ),
                    fn=operator.mul,
                    fn_str="*",
                )
            )

        left_tree = cls.build(dist.left)
        right_tree = cls.build(dist.right)

        # Make a list of possibly-joinable distributions, plus a list of
        # children as trees who could not be simplified at this level
        children = []

        # If the child nodes use the same commutable operation as ``dist``, add
        # their flattened ``children`` lists to ``children``. Otherwise, put
        # the whole node in ``children``.
        if left_tree.is_leaf:
            children.append(left_tree.dist)
        elif left_tree.fn == dist.fn and dist.fn in cls.COMMUTABLE_OPERATIONS:
            children.extend(left_tree.children)
        else:
            children.append(left_tree)
        if right_tree is not None:
            if right_tree.is_leaf:
                children.append(right_tree.dist)
            elif right_tree.fn == dist.fn and dist.fn in cls.COMMUTABLE_OPERATIONS:
                children.extend(right_tree.children)
            else:
                children.append(right_tree)

        return cls(
            fn=dist.fn, fn_str=dist.fn_str, children=children, is_unary=is_unary, infix=dist.infix
        )

    def _join_dists(self, left_type, right_type, join_fn, commutative=True, condition=None):
        simplified_dists = []
        acc = None
        acc_index = None
        acc_is_left = True
        for i, x in enumerate(self.children):
            if isinstance(x, BaseDistribution) and (x.lclip is not None or x.rclip is not None):
                # We can't simplify a clipped distribution
                simplified_dists.append(x)
            elif acc is None and isinstance(x, left_type):
                acc = x
                acc_index = i
            elif (
                acc is not None
                and isinstance(x, right_type)
                and acc_is_left
                and (condition is None or condition(acc, x))
            ):
                acc = join_fn(acc, x)
            elif commutative and acc is None and isinstance(x, right_type):
                acc = x
                acc_index = i
                acc_is_left = False
            elif (
                commutative
                and acc is not None
                and isinstance(x, left_type)
                and not acc_is_left
                and (condition is None or condition(x, acc))
            ):
                acc = join_fn(x, acc)
            else:
                simplified_dists.append(x)

        if acc is not None:
            simplified_dists.insert(acc_index, acc)
        self.children = simplified_dists

    @classmethod
    def _lognormal_times_const(cls, norm_mean, norm_sd, k):
        if k == 0:
            return 0
        elif k > 0:
            return LognormalDistribution(norm_mean=norm_mean + np.log(k), norm_sd=norm_sd)
        else:
            return -LognormalDistribution(norm_mean=norm_mean + np.log(-k), norm_sd=norm_sd)

    def simplify(self):
        """Convert a FlatTree back into a Distribution, simplifying as much as
        possible."""
        if self.is_leaf:
            return self.dist

        for i in range(len(self.children)):
            if isinstance(self.children[i], FlatTree):
                self.children[i] = self.children[i].simplify()

        # Simplify unary operations
        if len(self.children) == 1:
            child = self.children[0]
            if self.fn == operator.neg:
                if isinstance(child, Real):
                    return -child
                if isinstance(child, NormalDistribution):
                    return NormalDistribution(mean=-child.mean, sd=child.sd)

            return ComplexDistribution(
                child, right=None, fn=self.fn, fn_str=self.fn_str, infix=self.infix
            )

        if self.fn == operator.add:
            self._join_dists(
                NormalDistribution,
                NormalDistribution,
                lambda x, y: NormalDistribution(
                    mean=x.mean + y.mean, sd=np.sqrt(x.sd**2 + y.sd**2)
                ),
            )
            self._join_dists(
                NormalDistribution, Real, lambda x, y: NormalDistribution(mean=x.mean + y, sd=x.sd)
            )
            self._join_dists(
                BernoulliDistribution,
                BernoulliDistribution,
                lambda x, y: BinomialDistribution(n=2, p=x.p),
                condition=lambda x, y: x.p == y.p,
            )
            self._join_dists(
                BinomialDistribution,
                BernoulliDistribution,
                lambda x, y: BinomialDistribution(n=x.n + 1, p=x.p),
                condition=lambda x, y: x.p == y.p,
            )
            self._join_dists(
                BinomialDistribution,
                BinomialDistribution,
                lambda x, y: BinomialDistribution(n=x.n + y.n, p=x.p),
                condition=lambda x, y: x.p == y.p,
            )
            self._join_dists(
                ChiSquareDistribution,
                ChiSquareDistribution,
                lambda x, y: ChiSquareDistribution(df=x.df + y.df),
            )
            self._join_dists(
                ExponentialDistribution,
                ExponentialDistribution,
                lambda x, y: GammaDistribution(shape=2, scale=x.scale),
                condition=lambda x, y: x.scale == y.scale,
            )
            self._join_dists(
                ExponentialDistribution,
                GammaDistribution,
                lambda x, y: GammaDistribution(shape=y.shape + 1, scale=x.scale),
                condition=lambda x, y: x.scale == y.scale,
            )
            self._join_dists(
                GammaDistribution,
                GammaDistribution,
                lambda x, y: GammaDistribution(shape=x.shape + y.shape, scale=x.scale),
                condition=lambda x, y: x.scale == y.scale,
            )
            self._join_dists(
                PoissonDistribution,
                PoissonDistribution,
                lambda x, y: PoissonDistribution(lam=x.lam + y.lam),
            )

        elif self.fn == operator.mul:
            self._join_dists(
                NormalDistribution,
                Real,
                lambda x, y: NormalDistribution(mean=x.mean * y, sd=x.sd * y),
            )
            self._join_dists(
                LognormalDistribution,
                LognormalDistribution,
                lambda x, y: LognormalDistribution(
                    norm_mean=x.norm_mean + y.norm_mean,
                    norm_sd=np.sqrt(x.norm_sd**2 + y.norm_sd**2),
                ),
            )
            self._join_dists(
                LognormalDistribution,
                Real,
                lambda x, y: self._lognormal_times_const(x.norm_mean, x.norm_sd, y),
            )
            self._join_dists(
                ExponentialDistribution,
                Real,
                lambda x, y: ExponentialDistribution(scale=x.scale * y),
            )
            self._join_dists(
                GammaDistribution,
                Real,
                lambda x, y: GammaDistribution(shape=x.shape, scale=x.scale * y),
            )

        elif self.fn == operator.truediv:
            self._join_dists(
                LognormalDistribution,
                LognormalDistribution,
                lambda x, y: LognormalDistribution(
                    norm_mean=x.norm_mean - y.norm_mean,
                    norm_sd=np.sqrt(x.norm_sd**2 + y.norm_sd**2),
                ),
                commutative=False,
            )
            self._join_dists(
                Real,
                LognormalDistribution,
                lambda x, y: self._lognormal_times_const(-y.norm_mean, y.norm_sd, x),
                commutative=False,
            )

        elif self.fn == operator.pow:
            self._join_dists(
                LognormalDistribution,
                Real,
                lambda x, y: LognormalDistribution(
                    norm_mean=x.norm_mean * y, norm_sd=x.norm_sd * y
                ),
                commutative=False,
                condition=lambda x, y: y > 0,
            )

        return reduce(
            lambda acc, x: ComplexDistribution(acc, x, fn=self.fn, fn_str=self.fn_str),
            self.children,
        )
