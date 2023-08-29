from __future__ import annotations

import os
import time
import math
import msgspec

import numpy as np
import pathos.multiprocessing as mp

from datetime import datetime
from typing import Callable, Optional, Union

from .distributions import (
    BaseDistribution,
    MixtureDistribution,
    BetaDistribution,
    NormalDistribution,
    norm,
    beta,
    mixture,
)
from .utils import Weights, _core_cuts, _init_tqdm, _tick_tqdm, _flush_tqdm


_squigglepy_internal_bayesnet_caches = {}


def simple_bayes(likelihood_h: float, likelihood_not_h: float, prior: float) -> float:
    """
    Calculate Bayes rule.

    p(h|e) = (p(e|h)*p(h)) / (p(e|h)*p(h) + p(e|~h)*(1-p(h)))
    p(h|e) is called posterior
    p(e|h) is called likelihood
    p(h) is called prior

    Parameters
    ----------
    likelihood_h : float
        The likelihood (given that the hypothesis is true), aka p(e|h)
    likelihood_not_h : float
        The likelihood given the hypothesis is not true, aka p(e|~h)
    prior : float
        The prior probability, aka p(h)

    Returns
    -------
    float
        The result of Bayes rule, aka p(h|e)

    Examples
    --------
    # Cancer example: prior of having cancer is 1%, the likelihood of a positive
    # mammography given cancer is 80% (true positive rate), and the likelihood of
    # a positive mammography given no cancer is 9.6% (false positive rate).
    # Given this, what is the probability of cancer given a positive mammography?
    >>> simple_bayes(prior=0.01, likelihood_h=0.8, likelihood_not_h=0.096)
    0.07763975155279504
    """
    return (likelihood_h * prior) / (likelihood_h * prior + likelihood_not_h * (1 - prior))


# TODO: output type for bayesnet
def bayesnet(
    event_fn: Optional[Callable] = None,
    n: int = 1,
    find: Optional[Callable] = None,
    conditional_on: Optional[Callable] = None,
    reduce_fn: Optional[Callable] = None,
    raw: bool = False,
    memcache: bool = True,
    memcache_load: bool = True,
    memcache_save: bool = True,
    reload_cache: bool = False,
    dump_cache_file: str = "",
    load_cache_file: str = "",
    cache_file_primary: bool = False,
    verbose: bool = False,
    cores: int = 1,
):
    """
    Calculate a Bayesian network.

    Allows you to find conditional probabilities of custom events based on
    rejection sampling.

    Parameters
    ----------
    event_fn : function
        A function that defines the bayesian network
    n : int
        The number of samples to generate
    find : a function or None
        What do we want to know the probability of?
    conditional_on : a function or None
        When finding the probability, what do we want to condition on?
    reduce_fn : a function or None
        When taking all the results of the simulations, how do we aggregate them
        into a final answer? Defaults to ``np.mean``.
    raw : bool
        If True, just return the results of each simulation without aggregating.
    memcache : bool
        If True, cache the results in-memory for future calculations. Each cache
        will be matched based on the ``event_fn``. Default ``True``.
    memcache_load : bool
        If True, load cache from the in-memory. This will be true if ``memcache``
        is True. Cache will be matched based on the ``event_fn``. Default ``True``.
    memcache_save : bool
        If True, save results to an in-memory cache. This will be true if ``memcache``
        is True. Cache will be matched based on the ``event_fn``. Default ``True``.
    reload_cache : bool
        If True, any existing cache will be ignored and recalculated. Default ``False``.
    dump_cache_file : str
        If present, will write out the cache to a binary file with this path with
        ``.sqlcache`` appended to the file name.
    load_cache_file : str
        If present, will first attempt to load and use a cache from a file with this
        path with ``.sqlcache`` appended to the file name.
    cache_file_primary : bool
        If both an in-memory cache and file cache are present, the file
        cache will be used for the cache if this is True, and the in-memory cache
        will be used otherwise. Defaults to False.
    verbose : bool
        If True, will print out statements on computational progress.
    cores : int
        If 1, runs on a single core / process. If greater than 1, will run on a multiprocessing
        pool with that many cores / processes.

    Returns
    -------
    various
        The result of ``reduce_fn`` on ``n`` simulations of ``event_fn``.

    Examples
    --------
    # Cancer example: prior of having cancer is 1%, the likelihood of a positive
    # mammography given cancer is 80% (true positive rate), and the likelihood of
    # a positive mammography given no cancer is 9.6% (false positive rate).
    # Given this, what is the probability of cancer given a positive mammography?
    >> def mammography(has_cancer):
    >>    p = 0.8 if has_cancer else 0.096
    >>    return bool(sq.sample(sq.bernoulli(p)))
    >>
    >> def define_event():
    >>    cancer = sq.sample(sq.bernoulli(0.01))
    >>    return({'mammography': mammography(cancer),
    >>            'cancer': cancer})
    >>
    >> bayes.bayesnet(define_event,
    >>                find=lambda e: e['cancer'],
    >>                conditional_on=lambda e: e['mammography'],
    >>                n=1*M)
    0.07723995880535531
    """
    events = {}
    if memcache is True:
        memcache_load = True
        memcache_save = True
    elif memcache is False:
        memcache_load = False
        memcache_save = False
    has_in_mem_cache = event_fn in _squigglepy_internal_bayesnet_caches
    cache_path = load_cache_file + ".sqcache" if load_cache_file != "" else ""
    has_file_cache = os.path.exists(cache_path) if load_cache_file != "" else False
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder()

    if load_cache_file and not has_file_cache and verbose:
        print("Warning: cache file `{}.sqcache` not found.".format(load_cache_file))

    if not reload_cache:
        if load_cache_file and has_file_cache and (not has_in_mem_cache or cache_file_primary):
            if verbose:
                print("Loading from cache file (`{}`)...".format(cache_path))
            with open(cache_path, "rb") as f:
                events = decoder.decode(f.read())

        elif memcache_load and has_in_mem_cache:
            if verbose:
                print("Loading from in-memory cache...")
            events = _squigglepy_internal_bayesnet_caches.get(event_fn)

        if events:
            n_ = events.get("metadata")
            if n_ is not None:
                n_ = n_.get("n")
                if n_ is None:
                    raise ValueError("events is malformed")
                elif n_ < n:
                    raise ValueError(
                        ("insufficient samples - {} results cached but " + "requested {}").format(
                            events["metadata"]["n"], n
                        )
                    )
            else:
                raise ValueError("events is malformed")

            events = events.get("events", [])
            if verbose:
                print("...Loaded")

    elif verbose:
        print("Reloading cache...")

    assert events is not None
    if len(events) < 1:
        if event_fn is None:
            return None

        def run_event_fn(pbar=None, total_cores=1):
            _tick_tqdm(pbar, total_cores)
            return event_fn()

        if cores == 1:
            if verbose:
                print("Generating Bayes net...")
            r_ = range(n)
            pbar = _init_tqdm(verbose=verbose, total=n)
            events = [run_event_fn(pbar=pbar, total_cores=1) for _ in r_]
            _flush_tqdm(pbar)
        else:
            if verbose:
                print("Generating Bayes net with {} cores...".format(cores))
            with mp.ProcessingPool(cores) as pool:
                cuts = _core_cuts(n, cores)

                def multicore_event_fn(core, total_cores=1, verbose=False):
                    r_ = range(cuts[core])
                    pbar = _init_tqdm(verbose=verbose, total=n)
                    batch = [run_event_fn(pbar=pbar, total_cores=total_cores) for _ in r_]
                    _flush_tqdm(pbar)

                    if verbose:
                        print("Shuffling data...")

                    while not os.path.exists("test-core-{}.sqcache".format(core)):
                        with open("test-core-{}.sqcache".format(core), "wb") as outfile:
                            encoder = msgspec.msgpack.Encoder()
                            outfile.write(encoder.encode(batch))
                        if verbose:
                            print("Writing data...")
                            time.sleep(1)

                pool_results = pool.amap(multicore_event_fn, list(range(cores - 1)))
                multicore_event_fn(cores - 1, total_cores=cores, verbose=verbose)
                if verbose:
                    print("Waiting for other cores...")
                while not pool_results.ready():
                    if verbose:
                        print(".", end="", flush=True)
                    time.sleep(1)

        if cores > 1:
            if verbose:
                print("Collecting data...")
            events = []
            pbar = _init_tqdm(verbose=verbose, total=cores)
            for c in range(cores):
                _tick_tqdm(pbar, 1)
                with open("test-core-{}.sqcache".format(c), "rb") as infile:
                    events += decoder.decode(infile.read())
                os.remove("test-core-{}.sqcache".format(c))
            _flush_tqdm(pbar)
            if verbose:
                print("...Collected!")

    metadata = {"n": n, "last_generated": datetime.now()}
    cache_data = {"events": events, "metadata": metadata}
    if memcache_save and (not has_in_mem_cache or reload_cache):
        if verbose:
            print("Caching in-memory...")
        _squigglepy_internal_bayesnet_caches[event_fn] = cache_data
        if verbose:
            print("...Cached!")

    if dump_cache_file:
        cache_path = dump_cache_file + ".sqcache"
        if verbose:
            print("Writing cache to file `{}`...".format(cache_path))
        with open(cache_path, "wb") as f:
            f.write(encoder.encode(cache_data))
        if verbose:
            print("...Cached!")

    assert events is not None
    if conditional_on is not None:
        if verbose:
            print("Filtering conditional...")
        events = [e for e in events if conditional_on(e)]

    if len(events) < 1:
        raise ValueError("insufficient samples for condition")

    if conditional_on and verbose:
        print("...Filtered!")

    if find is None:
        if verbose:
            print("...Reducing")
        events = events if reduce_fn is None else reduce_fn(events)
        if verbose:
            print("...Reduced!")
    else:
        if verbose:
            print("...Finding")
        events = [find(e) for e in events]
        if verbose:
            print("...Found!")
        if not raw:
            if verbose:
                print("...Reducing")
            reduce_fn = np.mean if reduce_fn is None else reduce_fn
            events = reduce_fn(events)
            if verbose:
                print("...Reduced!")
    if verbose:
        print("...All done!")
    return events


def update(
    prior: Union[NormalDistribution, BetaDistribution],
    evidence: Union[NormalDistribution, BetaDistribution],
    evidence_weight: float = 1,
) -> BaseDistribution:
    """
    Update a distribution.

    Starting with a prior distribution, use Bayesian inference to perform an update,
    producing a posterior distribution from the evidence distribution.

    Parameters
    ----------
    prior : Distribution
        The prior distribution. Currently must either be normal or beta type. Other
        types are not yet supported.
    evidence : Distribution
        The distribution used to update the prior. Currently must either be normal
        or beta type. Other types are not yet supported.
    evidence_weight : float
        How much weight to put on the evidence distribution? Currently this only matters
        for normal distributions, where this should be equivalent to the sample weight.

    Returns
    -------
    Distribution
        The posterior distribution

    Examples
    --------
    >> prior = sq.norm(1,5)
    >> evidence = sq.norm(2,3)
    >> bayes.update(prior, evidence)
    <Distribution> norm(mean=2.53, sd=0.29)
    """
    if isinstance(prior, NormalDistribution) and isinstance(evidence, NormalDistribution):
        prior_mean = prior.mean
        prior_var = prior.sd**2
        evidence_mean = evidence.mean
        evidence_var = evidence.sd**2
        return norm(
            mean=(
                (evidence_var * prior_mean + evidence_weight * (prior_var * evidence_mean))
                / (evidence_weight * prior_var + evidence_var)
            ),
            sd=math.sqrt(
                (evidence_var * prior_var) / (evidence_weight * prior_var + evidence_var)
            ),
        )
    elif isinstance(prior, BetaDistribution) and isinstance(evidence, BetaDistribution):
        prior_a = prior.a
        prior_b = prior.b
        evidence_a = evidence.a
        evidence_b = evidence.b
        return beta(prior_a + evidence_a, prior_b + evidence_b)
    elif type(prior) != type(evidence):
        print(type(prior), type(evidence))
        raise ValueError("can only update distributions of the same type.")
    else:
        raise ValueError("type `{}` not supported.".format(prior.__class__.__name__))


def average(
    prior: BaseDistribution,
    evidence: BaseDistribution,
    weights: Optional[Weights] = [0.5, 0.5],
    relative_weights: Optional[Weights] = None,
) -> MixtureDistribution:
    """
    Average two distributions.

    Parameters
    ----------
    prior : Distribution
        The prior distribution.
    evidence : Distribution
        The distribution used to average with the prior.
    weights : list or np.array or float
        How much weight to put on ``prior`` versus ``evidence`` when averaging? If
        only one weight is passed, the other weight will be inferred to make the
        total weights sum to 1. Defaults to 50-50 weights.
    relative_weights : list or None
        Relative weights, which if given will be weights that are normalized
        to sum to 1.

    Returns
    -------
    MixtureDistribution
        A mixture distribution that accords weights to ``prior`` and ``evidence``.

    Examples
    --------
    >> prior = sq.norm(1, 5)
    >> evidence = sq.norm(2, 3)
    >> bayes.average(prior, evidence)
    <Distribution> mixture
    - 0.5 weight on <Distribution> norm(mean=3.0, sd=1.22)
    - 0.5 weight on <Distribution> norm(mean=2.5, sd=0.3)
    """
    return mixture(dists=[prior, evidence], weights=weights, relative_weights=relative_weights)
