from Generators.Uniform_Generators.QuasiGenerator import QuasiGenerator


class VanDerCorput(QuasiGenerator):
    """
    Class representing the VanderCorput quasi random number generator.

    Attributes:

    - base (int): the base for the quasi random number generator

    Methods:

    - is_prime(self, n) -> bool: check if a number is prime
    - generate(self) -> float: generate a quasi random number
    """

    def __init__(self, base: int):
        assert self.is_prime(
            base
        ), "Base must be a prime number for the VanDerCorput sequence."
        super().__init__(base)
        self.sequence = [0.1]

    def __str__(self) -> str:
        return f"VanDerCorput Generator: \nBase: {self.base}"

    @staticmethod
    def is_prime(n) -> bool:
        """
        Check if a number is prime.

        :param n: the number to check
        :return: True if n is prime, False otherwise (type: bool).
        """
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def generate(self) -> float:
        """
        Generate a quasi random number using the Van der Corput sequence.

        :return: A quasi random number (type: float).
        """
        num = 0
        f = 1 / self.base
        value = len(self.sequence)
        while value > 0:
            num += (value % self.base) * f
            value //= self.base
            f /= self.base
        num = min(max(num, 1e-10), 1 - 1e-10)
        self.sequence.append(num)
        return num
