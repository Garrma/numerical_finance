from Generators.Uniform_Generators.UniformGenerator import UniformGenerator


class PseudoGenerator(UniformGenerator):
    """
    Class representing the Pseudo random number generators.

    Attributes:

    - seed (int): the seed for the generator

    Properties:

    - seed (int): return the current seed of the generator
    """

    def __init__(self, seed: int):
        self._seed = seed

    @property
    def seed(self) -> int:
        return self._seed

    def generate(self) -> None:
        pass
