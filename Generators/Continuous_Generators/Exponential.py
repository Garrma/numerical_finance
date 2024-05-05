from math import log, exp

from Generators.Continuous_Generators.ContinuousGenerator import ContinuousGenerator
from Generators.Uniform_Generators.EcuyerCombined import EcuyerCombined

class Exponential(ContinuousGenerator):
    """
    Class representing the exponential r.v. generators.

    Attributes:

    - lamb (float): the rate of the exponential distribution (> 0)
    """

    def __init__(self, input_lambda: float):
        assert (
            input_lambda > 0
        ), "The rate must be strictly positive for Exponential distribution"
        self.lamb = input_lambda


class ExponentialInverseDistribution(Exponential):
    """
    Class representing an exponential random variable generator using the inverse distribution method.

    Attributes:

    - lamb (float): the rate of the exponential distribution (> 0)
    - seeds (Tuple[int, int]): the seeds for the generator (default: (123456789, 987654321))

    Properties:

    - seeds (Tuple[int, int]): get the seeds of the generator

    Methods:

    - generate(self) -> float: generate a realisation of the exponential distribution of rate lambda
    """

    def __init__(self, input_lambda: float, seeds=(123456789, 987654321)):
        super().__init__(input_lambda)
        self.generator = EcuyerCombined(seeds=seeds)

    @property
    def seeds(self) -> tuple[int, int]:
        return self.generator.seeds

    def generate(self) -> float:
        """
        Overrides the generate method of the RandomGenerator class
        It uses the Ecuyer Combined generator and the inverse distribution method to generate the realisation of an exponential r.v.

        :return: A realisation of the r.v. (type: float).
        """
        u = self.generator.generate()
        return -log(u) / self.lamb


class ExponentialRejectionSampling(Exponential):
    """
    Class representing an exponential random variable generator using the rejection sampling method.

    Attributes:

    - lamb (float): the rate of the exponential distribution (> 0)
    - seeds_1 (Tuple[int, int]): the seeds for the first generator (default: (1234, 5678))
    - seeds_2 (Tuple[int, int]): the seeds for the second generator (default: (4321, 8765))

    Properties:

    - seeds_1 (Tuple[int, int]): get the seeds of the first generator
    - seeds_2 (Tuple[int, int]): get the seeds of the second generator

    Methods:

    - generate(self) -> float: generate a realisation of the exponential distribution of rate lambda
    """

    def __init__(self, input_lambda: float, seeds_1=(1234, 5678), seeds_2=(4321, 8765)):
        super().__init__(input_lambda)
        self.generator_1 = EcuyerCombined(seeds=seeds_1)
        self.generator_2 = EcuyerCombined(seeds=seeds_2)

    @property
    def seeds_1(self) -> tuple[int, int]:
        return self.generator_1.seeds

    @property
    def seeds_2(self) -> tuple[int, int]:
        return self.generator_2.seeds

    def generate(self) -> float:
        """
        Overrides the generate method of the RandomGenerator class
        It uses the Ecuyer Combined generator and the rejection sampling method to generate the realisation of an exponential r.v.

        :return: A realisation of the r.v. (type: float).
        """
        u_1 = self.generator_1.generate()
        u_2 = self.generator_2.generate()
        z = -log(u_1) / self.lamb

        while u_2 > self.lamb * exp(-self.lamb * z):
            u_1 = self.generator_1.generate()
            u_2 = self.generator_2.generate()
            z = -log(u_1) / self.lamb

        return -log(u_2) / self.lamb
