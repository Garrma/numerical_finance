
from Generators.Discrete_Generators.DiscreteGenerator import DiscreteGenerator
from Generators.Uniform_Generators.EcuyerCombined import EcuyerCombined


class HeadTail(DiscreteGenerator):
    """
    Class representing a Bernoulli generator with p = 1/2. Inherits from DiscreteGenerator.

    Attributes:

    - proba (float): the probability of success (default = 0.5)
    - seeds (Tuple[int, int]): the seeds for the generator (default: (123456789, 987654321))

    Methods:

    - generate(self) -> int: generate a realisation a Bernoulli distribution with parameter p
    """

    def __init__(self, proba=0.5, seeds=(123456789, 987654321)):
        super().__init__(seeds=seeds)
        assert 0 <= proba <= 1, "Probability must be between [0, 1]"
        self.proba = proba
        self.generator = EcuyerCombined(seeds=seeds)

    def generate(self) -> int:
        """
        Overrides the generate method from RandomGenerator.
        Uses the Ecuyer Combined generator to generate a realisation from the Bernoulli distribution.

        :return: A realisation from the Bernoulli distribution (type: int).
        """
        return 1 if self.generator.generate() < self.proba else 0
