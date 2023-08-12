from .utils import is_dist
from .distributions import OperableDistribution


class Die(OperableDistribution):
    def __init__(self, sides=None, explode_on=None):
        super().__init__()

        if is_dist(sides) or callable(sides):
            from .samplers import sample

            sides = sample(sides)
        elif sides < 2:
            raise ValueError("cannot roll less than a 2-sided die.")
        elif not isinstance(sides, int):
            raise ValueError("can only roll an integer number of sides")
        elif explode_on is not None:
            if not isinstance(explode_on, list):
                explode_on = [explode_on]
            if len(explode_on) >= sides:
                raise ValueError("cannot explode on every value")
        self.sides = sides
        self.explode_on = explode_on

    def __str__(self):
        explode_out = (
            "" if self.explode_on is None else ", explodes on {}".format(str(self.explode_on))
        )
        out = "<Distribution> Die({}{})".format(self.sides, explode_out)
        return out


def die(sides, explode_on=None):
    """
    Create a distribution for a die.

    Parameters
    ----------
    sides : int
        The number of sides of the die that is rolled.

    explode_on : list or int
        An additional die will be rolled if the initial die rolls any of these values.
        Implements "exploding dice".

    Returns
    -------
    Distribution
        A distribution that models a coin flip, returning either "heads" or "tails"

    Examples
    --------
    >>> die(6)
    <Distribution> Die(6)
    """
    return Die(sides=sides, explode_on=explode_on)


class Coin(OperableDistribution):
    def __init__(self):
        super().__init__()

    def __str__(self):
        out = "<Distribution> Coin"
        return out


def coin():
    """
    Create a distribution for a coin

    Returns
    -------
    Distribution
        A distribution that models a coin flip, returning either "heads" or "tails"

    Examples
    --------
    >>> coin()
    <Distribution> Coin
    """
    return Coin()
