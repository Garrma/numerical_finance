
from math import log, cos, sin

from Generators.Continuous_Generators.ContinuousGenerator import ContinuousGenerator
from Generators.Uniform_Generators.EcuyerCombined import EcuyerCombined
from Generators.Uniform_Generators.VanDerCorput import VanDerCorput

class Normal(ContinuousGenerator):
    """
    Class representing the normal random variable generators.

    Attributes:

    - mu (float): the mean of the normal distribution
    - sigma (float): the standard deviation of the normal distribution
    """

    PI = 3.14159265358979323846

    def __init__(self, input_mu: float, input_sigma: float):
        assert (
            input_sigma > 0
        ), "The standard deviation must be strictly positive for Normal distribution"
        self.mu = input_mu
        self.sigma = input_sigma


class NormalBoxMuller(Normal):
    """
    Class representing a normal random variable generator using the Box-Muller method.

    Attributes:

    - mu (float): the mean of the normal distribution
    - sigma (float): the standard deviation of the normal distribution
    - seeds_1 (Tuple[int, int]): the seeds for the first Ecuyer Combined generator (default: (1234, 5678))
    - seeds_2 (Tuple[int, int]): the seeds for the second Ecuyer Combined generator (default: (4321, 8765))

    Properties:

    - seeds_1 (Tuple[int, int]): get the seeds of the first Ecuyer Combined generator
    - seeds_2 (Tuple[int, int]): get the seeds of the second Ecuyer Combined generator

    Methods:

    - generate(self) -> float: generate a realisation of the normal distribution of mean mu and standard deviation sigma
    """

    def __init__(
        self,
        input_mu: float,
        input_sigma: float,
        seeds_1=(1234, 5678),
        seeds_2=(4321, 8765),
    ):
        super().__init__(input_mu, input_sigma)
        self.generator_1 = EcuyerCombined(seeds=seeds_1)
        self.generator_2 = EcuyerCombined(seeds=seeds_2)

    @property
    def seeds_1(self) -> tuple[int, int]:
        return self.generator_1.seeds

    @property
    def seeds_2(self) -> tuple[int, int]:
        return self.generator_2.seeds

    def generate(self) -> float:
        """
        Overrides the generate method of the RandomGenerator class
        It uses the Ecuyer Combined generator and the Box-Muller method to generate the realisation of a normal r.v.

        :return: A realisation of the r.v. (type: float).
        """
        u_1 = self.generator_1.generate()
        u_2 = self.generator_2.generate()
        r = (-2 * log(u_1)) ** 0.5
        theta = 2 * self.PI * u_2
        x = r * cos(theta)
        z = self.mu + x * self.sigma
        return z


class QuasiNormalBoxMuller(Normal):
    """
    Class representing a normal quasi-random variable generator using the Box-Muller method.

    Attributes:

    - mu (float): the mean of the normal distribution
    - sigma (float): the standard deviation of the normal distribution
    - bases (Tuple[int, int]): the bases of the Van Der Corput sequences (default: (3, 5))

    Properties:

    - bases (Tuple[int, int]): get the bases of the Van Der Corput sequences

    Methods:

    - generate(self) -> float: generate a realisation of the normal distribution of mean mu and standard deviation sigma
    """

    def __init__(self, input_mu: float, input_sigma: float, bases=(3, 5)):
        super().__init__(input_mu, input_sigma)
        self.generator_1 = VanDerCorput(bases[0])
        self.generator_2 = VanDerCorput(bases[1])

    @property
    def bases(self) -> tuple[int, int]:
        return self.generator_1.base, self.generator_2.base

    def generate(self) -> float:
        """
        Overrides the generate method of the RandomGenerator class
        It uses the Van Der Corput generator and the Box-Muller method to generate the realisation of a normal r.v.

        :return: A realisation of the r.v. (type: float).
        """
        u_1 = self.generator_1.generate()
        u_2 = self.generator_2.generate()
        r = (-2 * log(u_1)) ** 0.5
        theta = 2 * self.PI * u_2
        x = r * cos(theta)
        z = self.mu + x * self.sigma
        return z


class NormalCLT(Normal):
    """
    Class representing a normal random variable generator using the central limit theorem.

    Attributes:

    - mu (float): the mean of the normal distribution
    - sigma (float): the standard deviation of the normal distribution
    - seeds (Tuple[int, int]): the seeds for the first Ecuyer Combined generator (default: (123456789, 987654321))

    Properties:

    - seeds (Tuple[int, int]): get the seeds of the Ecuyer Combined generator

    Methods:

    - generate(self) -> float: generate a realisation of the normal distribution of mean mu and standard deviation sigma
    """

    def __init__(
        self, input_mu: float, input_sigma: float, seeds=(123456789, 987654321)
    ):
        super().__init__(input_mu, input_sigma)
        self.generator = EcuyerCombined(seeds=seeds)

    @property
    def seeds(self) -> tuple[int, int]:
        return self.generator.seeds

    def generate(self) -> float:
        """
        Overrides the generate method of the RandomGenerator base class.
        It uses the Ecuyer Combined generator and the central limit theorem to generate the realisation of a normal r.v.

        :return: A realisation of the r.v. (type: float).
        """
        x = sum([self.generator.generate() for _ in range(20)]) - 10
        z = self.mu + x * self.sigma
        return z


class QuasiNormalCLT(Normal):
    """
    Class representing a normal quasi-random variable generator using the central limit theorem.

    Attributes:

    - mu (float): the mean of the normal distribution
    - sigma (float): the standard deviation of the normal distribution
    - base (int): the base of the Van Der Corput sequence

    Properties:

    - base (int): get the base of the Van Der Corput sequence

    Methods:

    - generate(self) -> float: generate a realisation of the normal distribution of mean mu and standard deviation sigma
    """

    def __init__(self, input_mu: float, input_sigma: float, base=3):
        super().__init__(input_mu, input_sigma)
        self.generator = VanDerCorput(base)

    @property
    def base(self) -> int:
        return self.generator.base

    def generate(self) -> float:
        """
        Overrides the generate method of the RandomGenerator base class.
        It uses the Van Der Corput generator and the central limit theorem to generate the realisation of a normal r.v.

        :return:  A realisation of the r.v. (type: float).
        """
        x = sum([self.generator.generate() for _ in range(20)]) - 10
        z = self.mu + x * self.sigma
        return z


class NormalRejectionSampling(Normal):
    """
    Class representing a normal generator using the rejection sampling method.

    Attributes:

    - mu (float): the mean of the normal distribution
    - sigma (float): the standard deviation of the normal distribution
    - seeds_1 (Tuple[int, int]): the seeds for the first Ecuyer Combined generator (default: (1234, 5678))
    - seeds_2 (Tuple[int, int]): the seeds for the second Ecuyer Combined generator (default: (4321, 8765))
    - seeds_3 (Tuple[int, int]): the seeds for the third Ecuyer Combined generator (default: (123456789, 987654321))

    Properties:

    - seeds_1 (Tuple[int, int]): get the seeds of the first Ecuyer Combined generator
    - seeds_2 (Tuple[int, int]): get the seeds of the second Ecuyer Combined generator
    - seeds_3 (Tuple[int, int]): get the seeds of the third Ecuyer Combined generator

    Methods:

    - generate(self) -> float: generate a realisation of the normal distribution of mean mu and standard deviation sigma
    """

    def __init__(
        self,
        input_mu: float,
        input_sigma: float,
        seeds_1=(1234, 5678),
        seeds_2=(4321, 8765),
        seeds_3=(123456789, 987654321),
    ):
        super().__init__(input_mu, input_sigma)
        self.generator_1 = EcuyerCombined(seeds=seeds_1)
        self.generator_2 = EcuyerCombined(seeds=seeds_2)
        self.generator = EcuyerCombined(seeds=seeds_3)

    @property
    def seeds_1(self) -> tuple[int, int]:
        return self.generator_1.seeds

    @property
    def seeds_2(self) -> tuple[int, int]:
        return self.generator_2.seeds

    @property
    def seeds_3(self) -> tuple[int, int]:
        return self.generator.seeds

    def generate(self) -> float:
        """
        Overrides the generate method of the RandomGenerator class
        It uses the Ecuyer Combined generator and the rejection sampling method to generate the realisation of a normal r.v.

        :return: A realisation of the r.v. (type: float).
        """
        u_1 = self.generator_1.generate()
        u_2 = self.generator_2.generate()
        y_1 = -log(u_1)
        y_2 = -log(u_2)
        while y_2 >= ((y_1 - 1) ** 2) / 2:
            u_1 = self.generator_1.generate()
            u_2 = self.generator_2.generate()
            y_1 = -log(u_1)
            y_2 = -log(u_2)

        u = self.generator.generate()
        z = y_1 if u < 0.5 else -y_1
        n = self.mu + z * self.sigma
        return n


class QuasiNormalRejectionSampling(Normal):
    """
    Class representing a normal quasi-random variable generator using the rejection sampling method.

    Attributes:

    - mu (float): the mean of the normal distribution
    - sigma (float): the standard deviation of the normal distribution
    - bases (Tuple[int, int, int]): the bases of the Van Der Corput sequences (default: (3, 5, 7))

    Properties:

    - bases (Tuple[int, int, int]): get the bases of the Van Der Corput sequences

    Methods:

    - generate(self) -> float: generate a realisation of the normal distribution of mean mu and standard deviation sigma
    """

    def __init__(self, input_mu: float, input_sigma: float, bases=(3, 5, 7)):
        super().__init__(input_mu, input_sigma)
        self.generator_1 = VanDerCorput(bases[0])
        self.generator_2 = VanDerCorput(bases[1])
        self.generator = VanDerCorput(bases[2])

    @property
    def bases(self) -> tuple[int, int, int]:
        return self.generator_1.base, self.generator_2.base, self.generator.base

    def generate(self) -> float:
        """
        Overrides the generate method of the RandomGenerator class
        It uses the Van Der Corput generator and the rejection sampling method to generate the realisation of a normal quasi r.v.

        :return:  A realisation of the quasi r.v. (type: float).
        """
        u_1 = self.generator_1.generate()
        u_2 = self.generator_2.generate()
        y_1 = -log(u_1)
        y_2 = -log(u_2)
        while y_2 >= ((y_1 - 1) ** 2) / 2:
            u_1 = self.generator_1.generate()
            u_2 = self.generator_2.generate()
            y_1 = -log(u_1)
            y_2 = -log(u_2)

        u = self.generator.generate()
        z = y_1 if u < 0.5 else -y_1
        n = self.mu + z * self.sigma
        return n


class NormalBivariate(ContinuousGenerator):
    """
    Class representing a bivariate normal random variable generator.

    Attributes:

    - mu_1 (float): the mean of the first normal distribution
    - mu_2 (float): the mean of the second normal distribution
    - sigma_1 (float): the standard deviation of the first normal distribution
    - sigma_2 (float): the standard deviation of the second normal distribution
    - rho (float): the correlation coefficient

    Methods:

    - generate(self) -> Tuple[float, float]: generate a couple X, Y of random normal variables
    - mean(self, nb_sim: int, variable: int = 0) -> float: calculate the mean of the simulated variables
    - variance(self, nb_sim: int, variable: int = 0) -> float: calculate the variance of the simulated variables

    """

    PI = 3.14159265358979323846

    def __init__(
        self,
        input_mu_1: float,
        input_mu_2: float,
        input_sigma_1: float,
        input_sigma_2: float,
        input_rho: float,
    ):
        assert (
            input_sigma_1 > 0
        ), "The standard deviation must be strictly positive for Normal distribution"
        assert (
            input_sigma_2 > 0
        ), "The standard deviation must be strictly positive for Normal distribution"
        assert (
            -1 <= input_rho <= 1
        ), "The correlation coefficient must be between -1 and 1"
        self.mu_1 = input_mu_1
        self.mu_2 = input_mu_2
        self.sigma_1 = input_sigma_1
        self.sigma_2 = input_sigma_2
        self.rho = input_rho

    def mean(self, nb_sim: int, variable=0) -> None:
        """
        Overrides the mean method of the RandomGenerator class

        :param nb_sim: The number of simulations (type: int).
        :param variable: The type of variable (type: int, default: 0).
        :return: The mean of the generated numbers (type: float).
        """
        assert nb_sim > 0, "The number of simulations must be greater than 0"
        assert variable == 0 or variable == 1, "Variable must be 0 or 1"
        if variable == 0 or variable == 1:
            return sum([self.generate()[variable] for _ in range(nb_sim)]) / nb_sim  # type: ignore

    def variance(self, nb_sim: int, variable=0) -> float:
        """
        Overrides the variance method of the RandomGenerator class.

        :param nb_sim: The number of simulations (type: int).
        :param variable: The type of variable (type: int, default: 0).
        :return: The variance of the generated numbers (type: float).
        """
        assert nb_sim > 0, "The number of simulations must be greater than 0"
        assert variable == 0 or variable == 1, "Variable must be 0 or 1"
        generated_numbers = [self.generate()[variable] for _ in range(nb_sim)]  # type: ignore
        mean = sum(generated_numbers) / nb_sim
        return sum([(sim - mean) ** 2 for sim in generated_numbers]) / nb_sim


class NormalBivariateBoxMuller(NormalBivariate):
    """
    Class representing a bivariate normal pseudo-random variable generator using the Box-Muller method.

     Attributes:

     - mu_1 (float): the mean of the first normal distribution
     - mu_2 (float): the mean of the second normal distribution
     - sigma_1 (float): the standard deviation of the first normal distribution
     - sigma_2 (float): the standard deviation of the second normal distribution
     - rho (float): the correlation coefficient
     - seeds_1 (Tuple[int, int]): the seeds for the first Ecuyer Combined generator (default: (1234, 5678))
     - seeds_2 (Tuple[int, int]): the seeds for the second Ecuyer Combined generator (default: (4321, 8765))

     Properties:

     - seeds_1 (Tuple[int, int]): get the seeds of the first Ecuyer Combined generator
     - seeds_2 (Tuple[int, int]): get the seeds of the second Ecuyer Combined generator

     Methods:

     - generate(self) -> Tuple[float, float]: generate a couple X, Y of pseudo random normal variables
    """

    def __init__(
        self,
        input_mu_1: float,
        input_mu_2: float,
        input_sigma_1: float,
        input_sigma_2: float,
        input_rho: float,
        seeds_1=(1234, 5678),
        seeds_2=(4321, 8765),
    ):
        super().__init__(
            input_mu_1, input_mu_2, input_sigma_1, input_sigma_2, input_rho
        )
        self.generator_1 = EcuyerCombined(seeds=seeds_1)
        self.generator_2 = EcuyerCombined(seeds=seeds_2)

    @property
    def seeds_1(self) -> tuple[int, int]:
        return self.generator_1.seeds

    @property
    def seeds_2(self) -> tuple[int, int]:
        return self.generator_2.seeds

    def generate(self) -> tuple[float, float]:
        """
        Overrides the generate method of the RandomGenerator class
        It uses the Ecuyer Combined generator and the Box-Muller method to generate a couple of normal variables

        :return: A realisation of the couple of normal pseudo r.v. (type: Tuple[float, float]).
        """
        u_1 = self.generator_1.generate()
        u_2 = self.generator_2.generate()
        r = (-2 * log(u_1)) ** 0.5
        theta = 2 * self.PI * u_2
        z_1 = r * cos(theta)
        z_2 = r * sin(theta)
        x = self.mu_1 + z_1 * self.sigma_1
        y = self.sigma_2 * (self.rho * z_1 + (1 - self.rho**2) ** 0.5 * z_2) + self.mu_2
        return x, y


class QuasiNormalBivariateBoxMuller(NormalBivariate):
    """
    Class representing a bivariate normal quasi-random variable generator using the Box-Muller method.

    Attributes:

    - mu_1 (float): the mean of the first normal distribution
    - mu_2 (float): the mean of the second normal distribution
    - sigma_1 (float): the standard deviation of the first normal distribution
    - sigma_2 (float): the standard deviation of the second normal distribution
    - rho (float): the correlation coefficient
    - bases (Tuple[int, int]): the bases of the Van Der Corput sequences (default: (3, 5))

    Properties:

    - bases (Tuple[int, int]): get the bases of the Van Der Corput sequences

    Methods:

    - generate(self) -> Tuple[float, float]: generate a couple X, Y of quasi random normal variables
    """

    def __init__(
        self,
        input_mu_1: float,
        input_mu_2: float,
        input_sigma_1: float,
        input_sigma_2: float,
        input_rho: float,
        bases=(3, 5),
    ):
        super().__init__(
            input_mu_1, input_mu_2, input_sigma_1, input_sigma_2, input_rho
        )
        self.generator_1 = VanDerCorput(bases[0])
        self.generator_2 = VanDerCorput(bases[1])

    @property
    def bases(self) -> tuple[int, int]:
        return self.generator_1.base, self.generator_2.base

    def generate(self) -> tuple[float, float]:
        """
        Overrides the generate method of the RandomGenerator class
        It uses the Van Der Corput generator and the Box-Muller method to generate a couple of normal variables

        :return: A realisation of the couple of normal quasi r.v. (type: Tuple[float, float]).
        """

        u_1 = self.generator_1.generate()
        u_2 = self.generator_2.generate()
        r = (-2 * log(u_1)) ** 0.5
        theta = 2 * self.PI * u_2
        z_1 = r * cos(theta)
        z_2 = r * sin(theta)
        x = self.mu_1 + z_1 * self.sigma_1
        y = self.sigma_2 * (self.rho * z_1 + (1 - self.rho**2) ** 0.5 * z_2) + self.mu_2
        return x, y
