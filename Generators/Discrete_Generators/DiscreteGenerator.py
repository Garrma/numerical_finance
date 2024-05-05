
from Generators.Uniform_Generators.UniformGenerator import UniformGenerator


class DiscreteGenerator(UniformGenerator):
    """
    Class representing the Discrete random number generators.
    """

    def __init__(self, seeds=(123456789, 987654321)):
        self._seeds = seeds

    @property
    def seeds(self) -> tuple[int, int]:
        return self._seeds

    def generate(self) -> None:
        pass
