## v0.27

* **[Breaking change]** This package now only supports Python 3.9 and higher.
* **[Breaking change]** `get_percentiles` and `get_log_percentiles` now always return a dictionary, even if there's only one element.
* **[Breaking change]** `.type` is now removed from distribution objects.
* **[Breaking change]** `sq.flip_coin` and `sq.roll_die` as functions are now functionally replaced by sampling from `sq.coin` or `sq.die` distribution objects.
* Package load time is now ~2x faster.
* Pandas and matplotlib as removed as required dependencies, but their related features are lazily enabled when the modules are available. These packages are still available for install  as extras, installable with `pip install squigglepy[plots]` (for plotting-related functionality, matplotlib for now), `pip install squigglepy[ecosystem]` (for pandas, and in the future other related packages), or `pip install squigglepy[all]` (for all extras).
* Multicore distribution now does extra checks to avoid crashing from race conditions.
* Distribution objects now have the version of squigglepy they were created with, which can be accessed via `obj._version`. This should be helpful for debugging and noticing stale objects, especially when squigglepy distributions are stored in caches.
* Using black now for formatting.
* Switched from `flake8` to `ruff`.

## v0.26

* **[Breaking change]** `lognorm` can now be defined either referencing the mean and sd of the underlying normal distribution via `norm_mean` / `norm_sd` or via the mean and sd of the lognormal distribution itself via `lognorm_mean` / `lognorm_sd`. To further disambiguate, `mean` and `sd` are no longer variables that can be passed to `lognorm`.


## v0.25

* Added `plot` as a method to more easily plot distributions.
* Added `dist_log` and `dist_exp` operators on distributions.
* Added `growth_rate_to_doubling_time` and `doubling_time_to_growth_rate` convenience functions. These take numbers, numpy arrays or distributions. 
* Mixture distributions now print with weights in addition to distributions.
* Changes `get_log_percentiles` to report in scientific notation.
* `bayes` now supports separate arguments for `memcache_load` and `memcache_save` to better customize how memcache behavior works. `memcache` remains a parameter that sets both `memcache_load` and `memcache_save` to True.


## v0.24

* Distributions can now be negated with `-` (e.g., `-lognorm(0.1, 1)`).
* Numpy ints and floats can now be used for determining the number of samples.
* Fixed some typos in the documentation.


## v0.23

* Added `pareto` distribution.
* Added `get_median_and_ci` to return the median and a given confidence interval for data.
* `discrete` and `mixture` distributions now give more detail when printed.
* Fixed some typos in the documentation.


## v0.22

* Added `extremize` to extremize predictions.
* Added `normalize` to normalize a list of numbers to sum to 1.
* Added `get_mean_and_ci` to return the mean and a given confidence interval for data.
* Added `is_dist` to determine if an object is a Squigglepy distribution.
* Added `is_sampleable` to determine if an object can be sampled using `sample`.
* Support for working within Pandas is now explicitly added. `pandas` has been added as a requirement.
* `discrete` sampling now will compress a large array if possible for more efficient sampling.
* `clip`, `lclip`, and `rclip` can now be used without needing distributions.
* Some functions (e.g, `geomean`) previously only supported lists, dictionaries, and numpy arrays. They have been expanded to support all iterables.
* `dist_max` and `dist_min` now support pipes (`>>`)
* `get_percentiles` now coerces output to integer if `digits` is less than or equal to 0, instead of just exactly 0.


## v0.21

* Mixture sampling is now 4-23x faster.
* You can now get the version of squigglepy via `sq.__version__`.
* Fixes a bug where the tqdm was displayed with the incorrect count when collecting cores during a multicore `sample`.


## v0.20

* Fixes how package dependencies are handled in `setup.py` an specifies Python >= 3.7 must be used. This should fix install errors.


## v0.19

#### Bugfixes

* Fixes a bug where `lclip` and/or `rclip` on `mixture` distribution were not working correctly.
* Fixes a bug where `dist_fn` did not work with `np.vectorize` functions.
* Fixes a bug where in-memory caching was invoked for `bayesnet` when not desired.

#### Caching and Multicore

* **[Breaking change]** `bayesnet` caching is now based on binary files instead of pickle files (uses `msgspec` as the underlying library).
* **[Breaking change]** `sample` caching is now based on numpy files instead of pickle files.
* A cache can now be loaded via `sample(load_cache=cachefile)` or `bayesnet(load_cache=cachefile)`, without needing to pass the distribution / function.
* `bayesnet` and `sample` now take an argument `cores` (default 1). If greater than 1, will run the calculations on multiple cores using the pathos package.

#### Other

* Functions that take `weights` now can instead take a parameter `relative_weights` where waits are automatically normalized to sum to 1 (instead of erroring, which is still the behavior if using `weights`).
* Verbose output for `bayesnet` and `sample` is now clearer (and slightly more verbose).


## v0.18

* **[Breaking change]** The default `t` for t-distributions has changed from 1 to 20.
* `sample` results can now be cached in-memory using `memcache=True`. They can also be cached to a file -- use `dump_cache_file` to write the file and `load_cache_file` to load from the file.
* _(Non-visible backend change)_ Weights that are set to 0 are now dropped entirely.


## v0.17

* When `verbose=True` is used in `sample`, the progress bar now pops up in more relevant places and is much more likely to get triggered when relevant.
* `discrete_sample` and `mixture_sample` now can take `verbose` parameter.


## v0.16

* `zero_inflated` can create an arbitrary zero-inflated distribution.
* Individual sampling functions (`normal_sample`, `lognormal_sample`, etc.) can now take an argument `samples` to generate multiple samples.
* A large speedup has been achieved to sampling from the same distribution multiple times.
* `requirements.txt` has been updated.


## v0.15

* **[Breaking change]** `bayesnet` function now refers to parameter `memcache` where previously this parameter was called `cache`.
* **[Breaking change]** If `get_percentiles` or `get_log_percentiles` is called with just one elemement for `percentiles`, it will return that value instead of a dict.
* Fixed a bug where `get_percentiles` would not round correctly.
* `bayesnet` results can now be cached to a file. Use `dump_cache_file` to write the file and `load_cache_file` to load from the file.
* `discrete` now works with numpy arrays in addition to lists.
* Added `one_in` as a shorthand to convert percentages into "1 in X" notation.
* Distributions can now be compared with `==` and `!=`.


## v0.14

* Nested sampling now works as intended.
* You can now use `>>` for pipes for distributions. For example, `sq.norm(1, 2) >> dist_ceil`
* Distributions can now be compared with `>`, `<`, `>=`, and `<=`.
* `dist_max` can be used to get the maximum value between two distributions. This family of functions are not evaluated until the distribution is sampled and they work with pipes.
* `dist_min` can be used to get the minimum value between two distributions.
* `dist_round` can be used to round the final output of a distribution. This makes the distribution discrete.
* `dist_ceil` can be used to ceiling round the final output of a distribution. This makes the distribution discrete.
* `dist_floor` can be used to floor round the final output of a distribution. This makes the distribution discrete.
* `lclip` can be used to clip a distribution to a lower bound. This is the same functionality that is available within the distribution and the `sample` method.
* `rclip` can be used to clip a distribution to an upper bound. This is the same functionality that is available within the distribution and the `sample` method.
* `clip` can be used to clip a distribution to both an upper bound and a lower bound. This is the same functionality that is available within the distribution and the `sample` method.
* `sample` can now be used directly on numbers. This makes `const` functionally obsolete, but `const` is maintained for backwards compatibility and in case it is useful.
* `sample(None)` now returns `None` instead of an error.


## v0.13

* Sample shorthand notation can go in either order. That is, `100 @ sq.norm(1, 2)` now works and is the same as `sq.norm(1, 2) @ 100`, which is the same as `sq.sample(sq.norm(1, 2), n=100)`.


## v0.12

* Distributions now implement math directly. That is, you can do things like `sq.norm(2, 3) + sq.norm(4, 5)`, whereas previously this would not work. Thanks to Dawn Drescher for helping me implement this.
* `~sq.norm(1, 2)` is now a shorthand for `sq.sample(sq.norm(1, 2))`. Thanks to Dawn Drescher for helping me implement this shorthand.
* `sq.norm(1, 2) @ 100` is now a shorthand for `sq.sample(sq.norm(1, 2), n=100)`


## v0.11

#### Distributions

* **[Breaking change]** `tdist` and `log_tdist` have been modified to better approximate the desired credible intervals.
* `tdist` now can be defined by just `t`, producing a classic t-distribution.
* `tdist` now has a default value for `t`: 1.
* Added `chisquare` distribution.
* `lognormal` now returns an error if it is defined with a zero or negative value.

#### Other

* All functions now have docstrings.
* Added `kelly` to calculate Kelly criterion for bet sizing with probabilities.
* Added `full_kelly`, `half_kelly`, `quarter_kelly` as helpful aliases.


## v0.10

* **[Breaking change]** `credibility` is now defined using a number out of 100 (e.g., `credibility=80` to define an 80% CI) rather than a decimal out of 1 (e.g., `credibility=0.8` to define an 80% CI).
* Distribution objects now print their parameters.


## v0.9

* `goemean` and `geomean_odds` now can take the nested-list-based and dictionary-based formats for passing weights.


## v0.8

#### Bayesian library updates

* **[Breaking change]** `bayes.update` now updates normal distributions from the distribution rather than from samples.
* **[Breaking change]** `bayes.update` no longer takes a `type` parameter but can now infer the type from the passed distribution.
* **[Breaking change]** Corrected a bug in how `bayes.update` implemented `evidence_weight` when updating normal distributions.

#### Non-visible backend changes

* Distributions are now implemented as classes (rather than lists).


## v0.7

#### Bugfixes

* Fixes an issue with sampling from the `bernoulli` distribution.
* Fixes a bug with the implementation of `lclip` and `rclip`.

#### New distributions

* Adds `discrete` to calculate a discrete distribution. Example: `discrete({'A': 0.3, 'B': 0.3, 'C': 0.4})` will return A 30% of the time, B 30% of the time, and C 40% of the time.
* Adds `poisson(lam)` to calculate a poisson distribution.
* Adds `gamma(size, scale)` to calculate a gamma distribution.

#### Bayesian library updates

* Adds `bayes.bayesnet` to do bayesian inferece (see README).
* `bayes.update` now can take an `evidence_weight` parameter. Typically this would be equal to the number of samples.
* **[Breaking change]** `bayes.bayes` has been renamed `bayes.simple_bayes`.

#### Other

* **[Breaking change]** `credibility`, which defines the size of the interval (e.g., `credibility=0.8` for an 80% CI), is now a property of the distribution rather than the sampler. That is, you should now call `sample(norm(1, 3, credibility=0.8))` whereas previously it was `sample(norm(1, 3), credibility=0.8)`. This will allow mixing of distributions with different credible ranges.
* **[Breaking change]** Numbers have been changed from functions to global variables. Use `thousand` or `K` instead of `thousand()` (old/deprecated).
* `sample` now has a nice progress reporter if `verbose=True`.
* The `exponential` distribution now implements `lclip` and `rclip`.
* The `mixture` distribution can infer equal weights if no weights are given.
* The `mixture` distribution can infer the last weight if the last weight is not given.
* `geomean` and `geomean_odds` can infer the last weight if the last weight is not given.
* You can use `flip_coin` and `roll_die(sides)` to flip a coin or roll a die.
* `event_happens` and `event` are aliases for `event_occurs`.
* `get_percentiles` will now cast output to `int` if `digits=0`.
* `get_log_percentiles` now has a default value for `percentiles`.
* You can now set the seed for the RNG using `sq.set_seed`.

#### Non-visible backend changes

* Now has tests via pytest.
* The random numbers now come from a numpy generator as opposed to the previous deprecated `np.random` methods.
* The `sample` module (containing the `sample` function) has been renamed `samplers`.


## v0.6

#### New distributions

* Add `triangular(left, mode, right)` to calculate a triangular distribution.
* Add `binomial(n, p)` to calculate a binomial distribution.
* Add `beta(a, b)` to calculate a beta distribution.
* Add `bernoulli(p)` to calculate a bernoulli distribution.
* Add `exponential(scale)` to calculate an exponential distribution.

#### New Bayesian library

* Add `bayes.update` to get a posterior distribution from a prior distribution and an evidence distribution.
* Add `bayes.average` to average distributions (via a mixture).

#### New utility functions

* Add `laplace` to calculate Laplace's Law of Succession. If `s` and `n` are passed, it will calculate `(s+1)/(n+2)`. If `s`, `time_passed`, and `time_remaining` are passed, it will use the [time invariant version](https://www.lesswrong.com/posts/wE7SK8w8AixqknArs/a-time-invariant-version-of-laplace-s-rule). Use `time_fixed=True` for fixed time periods and `time_fixed=False` (default) otherwise.
* Add `geomean` to calculate the geometric mean.
* Add `p_to_odds` to convert probability to odds. Also `odds_to_p` to convert odds to probability.
* Add `geomean_odds` to calculate the geometric mean of odds, converted to and from probabilities.

#### Other

* If a distribution is defined with `sd` but not `mean`, `mean` will be inferred to be 0.
* `sample` can now take `lclip` and `rclip` directly, in addition to defining `lclip` and `rclip` on the distribution itself. If both are defined, the most restrictive of the two bounds will be used.


## v0.5

* Fix critical bug to `tdist` and `log_tdist` introduced in v0.3.


## v0.4

* Fix critical bug introduced in v0.3.


## v0.3

* Be able to define distributions using `mean` and `sd` instead of defining the interval.


## v0.2

* **[Breaking change]** Change `distributed_log` to `mixture` (to follow Squiggle) and allow it to implement any sub-distribution.
* **[Breaking change]** Changed library to single import.
* **[Breaking change]** Remove `weighted_log` as a distribution.


## v0.1

* Initial library
