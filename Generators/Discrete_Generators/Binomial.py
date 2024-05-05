from Generators.Discrete_Generators.Bernoulli import Bernoulli


class Binomial(Bernoulli):
    """
    Class representing a Binomial generator. Inherits from Bernoulli.

    Attributes:

    - n (int): the number of trials
    - p (float): the probability of success
    - seeds (Tuple[int, int]): the seeds for the generator (default: (123456789, 987654321))

    Methods:

    - generate(self) -> int: generate a realisation a Binomial distribution with parameters (n,p)
    """

    def __init__(self, n: int, p: float, seeds=(123456789, 987654321)):
        super().__init__(p, seeds=seeds)
        self.n = n
        self.generator = Bernoulli(p, seeds=seeds)

    def generate(self) -> int:
        """
        Overrides the generate method from RandomGenerator.
        Generates a realisation of the Binomial distribution.

        :return: A realisation of the r.v. (type: int).
        """
        return sum([self.generator.generate() for _ in range(self.n)])
