
import math

from Generators.Discrete_Generators.DiscreteGenerator import DiscreteGenerator
from Generators.Uniform_Generators.EcuyerCombined import EcuyerCombined
from Generators.Continuous_Generators.Exponential import ExponentialInverseDistribution


class Poisson(DiscreteGenerator):
    """
    Class representing a Poisson random variable generator.

    Attributes:

    - lamb (float): the rate of the Poisson distribution
    - seeds (Tuple[int, int]): the seeds for the generator (default: (123456789, 987654321))

    Methods:

    - generate(self) -> int: generate a realisation of a Poisson distribution with parameter lambda
    """

    def __init__(self, input_lambda: float, seeds=(123456789, 987654321)):
        super().__init__(seeds=seeds)
        assert (
            input_lambda > 0
        ), "The rate must be strictly positive for Poisson distribution"
        self.lamb = input_lambda

    def factorial(self, n: int) -> int:
        """
        Recursive function to compute the factorial of a number.

        :param n: the number to compute the factorial of (type: int)
        :return:  the factorial of n (type: int)
        """
        if n == 0:
            return 1
        else:
            return n * self.factorial(n - 1)

    def generate(self) -> None:
        pass


class PoissonFirstAlgo(Poisson):
    """
    Class representing a Poisson generator that uses the inverse distribution method to generate the Poisson distribution.

    Attributes:

    - lamb (float): the rate of the Poisson distribution
    - seeds (Tuple[int, int]): the seeds for the generator (default: (123456789, 987654321))

    Methods:

    - generate(self) -> int: generate a realisation of a Poisson distribution with parameter lambda
    """

    def __init__(self, input_lambda: float, seeds=(123456789, 987654321)):
        super().__init__(input_lambda, seeds=seeds)
        self.generator = EcuyerCombined(seeds=seeds)

    def generate(self) -> int:
        """
        Overrides the generate method from RandomGenerator.
        Uses the Ecuyer Combined generator to generate a realisation from the Poisson distribution.

        :return: A realisation from the Poisson distribution (type: int).
        """
        u = self.generator.generate()
        cumulative_prob = 0
        k = 0
        while True:
            pk = math.exp(-self.lamb) * self.lamb**k / self.factorial(k)
            cumulative_prob += pk
            if u < cumulative_prob:
                return k
            k += 1


class PoissonSecondAlgo(Poisson):
    """
    Class representing a Poisson random variable generator that uses the exponential distribution to generate the Poisson distribution.

    Attributes:

    - lamb (float): the rate of the Poisson distribution
    - seeds (Tuple[int, int]): the seeds for the generator (default: (123456789, 987654321))

    Methods:

    - generate(self) -> int: generate a realisation of a Poisson distribution with parameter lambda
    """

    def __init__(self, input_lambda, seeds=(123456789, 987654321)):
        super().__init__(input_lambda, seeds=seeds)
        self.generator = ExponentialInverseDistribution(input_lambda, seeds=seeds)

    def generate(self) -> int:
        """
        Overrides the generate method from RandomGenerator.
        Uses the ExponentialInverseDistribution generator to generate a realisation from the Poisson distribution.

        :return: A realisation from the Poisson distribution (type: int).
        """
        k = 0
        s = 0
        while True:
            s += self.generator.generate()
            if s >= 1:
                return k
            k += 1
