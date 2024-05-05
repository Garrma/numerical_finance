from Generators.Uniform_Generators.PseudoGenerator import PseudoGenerator
from Generators.Uniform_Generators.LinearCongruential import LinearCongruential


class EcuyerCombined(PseudoGenerator):
    """
    Class representing the Ecuyer Combined Generator. Uses two Linear Congruential Generators to generate a uniform random number.

    Attributes:

    - seeds (Tuple[int, int]): the seeds for the two Linear Congruential Generators (default: (123456789, 987654321))
    - first_linear (LinearCongruential): the first Linear Congruential Generator
    - second_linear (LinearCongruential): the second Linear Congruential Generator

    Properties:

    - seeds (Tuple[int, int]): get the seeds of the two Linear Congruential Generators

    Methods:

    - generate(self) -> float: generate a realisation of the uniform distribution
    """

    def __init__(self, seeds=(123456789, 987654321)):
        assert seeds[0] != seeds[1], "The seeds must be different"
        assert (
            1 <= seeds[0] <= 2147483562
        ), "The seeds must be in the range [1, 2147483562] for the first generator"
        assert (
            1 <= seeds[1] <= 2147483398
        ), "The seeds must be in the range [1, 2147483398] for the second generator"
        self.seed_1 = seeds[0]
        self.seed_2 = seeds[1]
        self.first_linear = LinearCongruential(self.seed_1, 40014, 0, 2147483563)
        self.second_linear = LinearCongruential(self.seed_2, 40692, 0, 2147483399)

    @property
    def seeds(self) -> tuple[int, int]:
        return self.seed_1, self.seed_2

    def generate(self) -> float:
        """
        Overrides the generate method of the RandomGenerator class
        It uses the two Linear Congruential Generators to generate a uniform random number.

        :return: A realisation of the uniform r.v. (type: float).
        """
        u_1 = self.first_linear.generate()
        u_2 = self.second_linear.generate()

        x_1 = self.first_linear.next_seed
        x_2 = self.second_linear.next_seed

        m_1 = 2147483563

        current = (x_1 - x_2) % (m_1 - 1)

        if current > 0:
            result = current / m_1
        elif current < 0:
            result = current / m_1 + 1
        else:
            result = (m_1 - 1) / m_1

        return result
