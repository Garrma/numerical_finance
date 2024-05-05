from Generators.Uniform_Generators.UniformGenerator import UniformGenerator


class QuasiGenerator(UniformGenerator):
    """
    Class representing the Quasi random number generators.

    Attributes:

    - base (int): the base for the quasi random number generator

    Properties:

    - base (int): return the current base of the generator

    Methods:

    - generate(self) ->  float: generate a quasi random number
    """

    def __init__(self, base: int):
        self._base = base

    @property
    def base(self) -> int:
        return self._base

    def generate(self) -> None:
        pass
