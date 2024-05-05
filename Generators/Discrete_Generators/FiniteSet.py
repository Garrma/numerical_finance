
from Generators.Discrete_Generators.DiscreteGenerator import DiscreteGenerator
from Generators.Uniform_Generators.EcuyerCombined import EcuyerCombined


class FiniteSet(DiscreteGenerator):
    """
    Class representing a discrete random number generators on a finite set.

    Attributes:

    - prob (list[float]): the probabilities of each element in the finite set
    - seeds (Tuple[int, int]): the seeds for the generator (default: (123456789, 987654321))

    Methods:

    - generate(self) -> int: generate a realisation from the finite set
    """

    def __init__(self, prob: list[float], seeds=(123456789, 987654321)):
        super().__init__(seeds=seeds)
        assert len(prob) > 0, "The probabilities list must not be empty"
        assert all(0 <= p <= 1 for p in prob), "Probabilities must be between [0, 1]"
        assert sum(prob) == 1, "The sum of probabilities must be equal to 1"

        self.prob = prob
        self.generator = EcuyerCombined(seeds=seeds)

    def generate(self) -> int:
        """
        Overrides the generate method from RandomGenerator.
        Uses the Ecuyer Combined generator to generate a realisation from the finite set.

        :return: A realisation from the finite set (type: int).
        """
        u = self.generator.generate()
        cumulative_prob = 0
        for i, p in enumerate(self.prob):
            cumulative_prob += p
            if u < cumulative_prob:
                return i
        return len(self.prob) - 1
