
from Generators.Discrete_Generators.DiscreteGenerator import DiscreteGenerator
from Generators.Uniform_Generators.EcuyerCombined import EcuyerCombined


class Bernoulli(DiscreteGenerator):
    """
    Class representing a Bernoulli generator. Inherits from DiscreteGenerator.

    Attributes:

    - p (float): the probability of success
    - seeds (Tuple[int, int]): the seeds for the generator (default: (123456789, 987654321))

    Methods:

    - generate(self) -> int: generate a realisation a Bernoulli distribution with parameter p
    """

    def __init__(self, p: float, seeds=(123456789, 987654321)):
        assert (
            0 <= p <= 1
        ), "The probability must be between 0 and 1 for Bernoulli distribution"
        super().__init__(seeds=seeds)
        self.p = p
        self.generator = EcuyerCombined(seeds=seeds)

    def generate(self) -> int:
        """
        Overrides the generate method from RandomGenerator.
        Generates a 1 with a probability p and 0 with (1-p)

        :return: A realisation of the r.v. (type: int).
        """
        return 1 if self.generator.generate() < self.p else 0
