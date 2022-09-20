## Squigglepy: Implementation of Squiggle in Python

[Squiggle](https://www.squiggle-language.com/) is a "simple programming language for intuitive probabilistic estimation". It serves as its own standalone programming language with its own syntax, but it is implemented in JavaScript. I like the features of Squiggle and intend to use it frequently, but I also sometimes want to use similar functionalities in Python, especially alongside other Python statistical programming packages like Numpy, Pandas, and Matplotlib. The **squigglepy** package here implements many Squiggle-like functionalities in Python.


## Installation

`pip3 install squigglepy`


## Usage

### Core Features

Here's the Squigglepy implementation of [the example from Squiggle Docs](https://www.squiggle-language.com/docs/Overview):

```Python
import squigglepy as sq
M = sq.million(); K = sq.thousand()

populationOfNewYork2022 = sq.to(8.1*M, 8.4*M) # This means that you're 90% confident the value is between 8.1 and 8.4 Million.

def proportionOfPopulationWithPianos():
    percentage = sq.to(.2, 1)
    return sq.sample(percentage) * 0.01 # We assume there are almost no people with multiple pianos

def pianoTunersPerPiano():
    pianosPerPianoTuner = sq.to(2*K, 50*K)
    return 1 / sq.sample(pianosPerPianoTuner)

def totalTunersIn2022():
    return (sq.sample(populationOfNewYork2022) *
            proportionOfPopulationWithPianos() *
            pianoTunersPerPiano())

sq.get_percentiles(sq.sample(totalTunersIn2022, n=1000))
```

And the version from the Squiggle doc that incorporates time:

```Python
import squigglepy as sq
K = sq.thousand(); M = sq.million()

populationOfNewYork2022 = sq.to(8.1*M, 8.4*M)

def proportionOfPopulationWithPianos():
    percentage = sq.to(.2, 1)
    return sq.sample(percentage) * 0.01

def proportionOfPopulationWithPianos():
    percentage = sq.to(.2, 1)
    return sq.sample(percentage) * 0.01

def pianoTunersPerPiano():
    pianosPerPianoTuner = sq.to(2*K, 50*K)
    return 1 / sq.sample(pianosPerPianoTuner)

# Time in years after 2022
def populationAtTime(t):
    averageYearlyPercentageChange = sq.to(-0.01, 0.05) # We're expecting NYC to continuously grow with an mean of roughly between -1% and +4% per year
    return sq.sample(populationOfNewYork2022) * ((sq.sample(averageYearlyPercentageChange) + 1) ** t)

def totalTunersAtTime(t):
    return (populationAtTime(t) *
            proportionOfPopulationWithPianos() *
            pianoTunersPerPiano())

sq.get_percentiles(sq.sample(lambda: totalTunersAtTime(2030-2022), n=1000))
```

### Additional Features

```Python
import squigglepy as sq

# Normal distribution
sq.sample(sq.norm(1, 3))  # 90% interval from 1 to 3

# Distribution can be sampled with mean and sd too
sq.sample(sq.norm(mean=0, sd=1))
sq.sample(sq.norm(-1.67, 1.67))  # This is equivalent to mean=0, sd=1

# Get more than one sample
sq.sample(sq.norm(1, 3), n=100)

# Other distributions exist
sq.sample(sq.lognorm(1, 10))
sq.sample(sq.tdist(1, 10, t=5))
sq.sample(sq.triangular(1, 2, 3))
sq.sample(sq.binomial(p=0.5, n=5))
sq.sample(sq.beta(a=1, b=2))
sq.sample(sq.bernoulli(p=0.5))
sq.sample(sq.exponential(scale=1))

# Discrete sampling
sq.sample(sq.discrete({'A': 0.1, 'B': 0.9}))

# Can return integers
sq.sample(sq.discrete({0: 0.1, 1: 0.3, 2: 0.3, 3: 0.15, 4: 0.15}))

# Alternate format (also can be used to return more complex objects)
sq.sample(sq.discrete([[0.1,  0],
                       [0.3,  1],
                       [0.3,  2],
                       [0.15, 3],
                       [0.15, 4]]))

sq.sample(sq.discrete([0, 1, 2])) # No weights assumes equal weights

# You can mix distributions together
sq.sample(sq.mixture([sq.norm(1, 3),
                      sq.norm(4, 10),
                      sq.lognorm(1, 10)],  # Distributions to mix
                     [0.3, 0.3, 0.4]))     # These are the weights on each distribution

# This is equivalent to the above, just a different way of doing the notation
sq.sample(sq.mixture([[0.3, sq.norm(1,3)],
                      [0.3, sq.norm(4,10)],
                      [0.4, sq.lognorm(1,10)]]))

# You can add and subtract distributions (a little less cool compared to native Squiggle unfortunately):
sq.sample(lambda: sq.sample(sq.norm(1,3)) + sq.sample(sq.norm(4,5))), n=100)
sq.sample(lambda: sq.sample(sq.norm(1,3)) - sq.sample(sq.norm(4,5))), n=100)
sq.sample(lambda: sq.sample(sq.norm(1,3)) * sq.sample(sq.norm(4,5))), n=100)
sq.sample(lambda: sq.sample(sq.norm(1,3)) / sq.sample(sq.norm(4,5))), n=100)

# You can change the CI from 90% (default) to 80%
sq.sample(sq.norm(1, 3), credibility=0.8)

# You can clip
sq.sample(sq.norm(0, 3, lclip=0, rclip=5)) # Sample norm with a 90% CI from 0-3, but anything lower than 0 gets clipped to 0 and anything higher than 5 gets clipped to 5.

# You can specify a constant (which can be useful for passing things into functions or mixtures)
sq.sample(sq.const(4)) # Always returns 4
```

### Bayesian inference

1% of women at age forty who participate in routine screening have breast cancer.
80% of women with breast cancer will get positive mammographies.
9.6% of women without breast cancer will also get positive mammographies.

A woman in this age group had a positive mammography in a routine screening.
What is the probability that she actually has breast cancer?

We can approximate the answer with a Bayesian network (uses rejection sampling):

```Python
def mammography(has_cancer):
    p = 0.8 if has_cancer else 0.096
    return bool(sq.sample(sq.bernoulli(p)))

def define_event():
    cancer = sq.sample(sq.bernoulli(0.01))    
    return({'mammography': mammography(cancer),
            'cancer': cancer})

bayesnet(define_event,
         n=1000000,
         find=lambda e: e['cancer'],
         conditional_on=lambda e: e['mammography'])
# 0.07723995880535531
```

Or if we have the information immediately on hand, we can directly calculate it:

```Python
from squigglepy import bayes
bayes.simple_bayes(prior=0.01, likelihood_h=0.8, likelihood_not_h=0.096)
# 0.07763975155279504
```

You can also make distributions and update them:

```Python
import matplotlib.pyplot as plt
from squigglepy import bayes

print('Prior')
prior = sq.norm(1,5)
prior_samples = sq.sample(prior, n=10000)
plt.hist(prior_samples, bins = 200)
plt.show()
print(sq.get_percentiles(prior_samples))
print('Prior Mean: {} SD: {}'.format(np.mean(prior_samples), np.std(prior_samples)))
print('-')

print('Evidence')
evidence = sq.norm(2,3)
evidence_samples = sq.sample(evidence, n=10000)
plt.hist(evidence_samples, bins = 200)
plt.show()
print(sq.get_percentiles(evidence_samples))
print('Evidence Mean: {} SD: {}'.format(np.mean(evidence_samples), np.std(evidence_samples)))
print('-')

print('Posterior')
posterior = bayes.update(prior_samples, evidence_samples)
posterior_samples = sq.sample(posterior, n=10000)
plt.hist(posterior_samples, bins = 200)
plt.show()
print(sq.get_percentiles(posterior_samples))
print('Posterior Mean: {} SD: {}'.format(np.mean(posterior_samples), np.std(posterior_samples)))

print('Average')
average = bayes.average(prior, evidence)
average_samples = sq.sample(average, n=10000)
plt.hist(average_samples, bins = 200)
plt.show()
print(sq.get_percentiles(average_samples))
print('Average Mean: {} SD: {}'.format(np.mean(average_samples), np.std(average_samples)))
```


### Rolling a Die

An example of how to use distributions to build tools:

```Python
def roll_die(sides):
    return sq.sample(sq.discrete(list(range(1, sides + 1))))

roll_die(6)
# [2, 6, 5, 2, 6, 2, 3, 1, 5, 2]
```

This is already included standard in the utils of this package. Use `sq.roll_die`.


### A Demonstration of the Monte Hall Problem

```Python
import random
import squigglepy as sq

def monte_hall(door_picked, switch=False, n=1):
    if n > 1:
        return [monte_hall(door_picked=door_picked, switch=switch, interactive=False, n=1) for _ in range(n)]
    
    doors = ['A', 'B', 'C']
    car_is_behind_door = sq.sample(sq.discrete({'A': 1/3, 'B': 1/3, 'C': 1/3}))    
    reveal_door = random.choice([d for d in doors if d != door_picked and d != car_is_behind_door])
    
    if switch:
        old_door_picked = door_picked
        door_picked = [d for d in doors if d != old_door_picked and d != reveal_door][0]
        
    won_car = (car_is_behind_door == door_picked)
    return won_car 


def percent_win(n, switch):
    return sum(monte_hall_(door_picked=door, switch=switch, n=n)) / n


for initial_door in ['A', 'B', 'C']:
    print('{} (No switch): {}'.format(initial_door, percent_win(n=10000, switch=False)))
    
for initial_door in ['A', 'B', 'C']:
    print('{} (switch): {}'.format(initial_door, percent_win(n=10000, switch=True)))

# Output:
# A (No switch): 0.3327
# B (No switch): 0.3403
# C (No switch): 0.3301
# A (switch): 0.6728
# B (switch): 0.6679
# C (switch): 0.6626
```

