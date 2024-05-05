from Generators.Uniform_Generators.PseudoGenerator import PseudoGenerator


class LinearCongruential(PseudoGenerator):
    """
    Class representing the Linear Congruential Generator. Generates a uniform random number.

    Attributes:

    - seed (int): the seed for the generator (default: 27)
    - multiplier (int): the multiplier for the generator (default: 17)
    - increment (int): the increment for the generator (default: 43)
    - modulus (int): the modulus for the generator (default: 100)
    - next_seed (int): the next seed for the generator

    Methods:

    - generate(self) -> float: generate a realisation of the uniform distribution
    """

    def __init__(self, seed=27, multiplier=17, increment=43, modulus=100):
        super().__init__(seed)
        self.multiplier = multiplier
        self.increment = increment
        self.modulus = modulus
        self.next_seed = self.seed

    @property
    def next(self) -> int:
        return self.next_seed

    def __str__(self) -> str:
        return f"Linear Congruential Generator: \nSeed: {self.seed}\nMultiplier: {self.multiplier}\nIncrement: {self.increment}\nModulus: {self.modulus}"

    def generate(self) -> float:
        """
        Overrides the generate method of the RandomGenerator class
        It generates a uniform random number.

        :return: A realisation of the r.v. (type: float).
        """
        current = int(
            (self.multiplier * self.next_seed + self.increment) % self.modulus
        )
        self.next_seed = current
        return current / self.modulus
