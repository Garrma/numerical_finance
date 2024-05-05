class RandomGenerator:
    """
    Base class for random number generators.

    Methods:

    - generate(self) -> float: generate a realisation of the distribution of the generator
    - generate_sim(self, nb_sim) -> list[float]: generate a list of random numbers
    - mean(self, nb_sim) -> float: compute the mean of the generated numbers
    - variance(self, nb_sim) -> float: compute the variance of the generated numbers
    """

    def generate(self) -> None:
        pass

    def generate_sim(self, nb_sim: int) -> list[float]:
        """
        This method generates a list of random numbers.

        :param nb_sim: the number of simulations (type: int)
        :return: a list of random numbers (type: list[float])
        """
        return [self.generate() for _ in range(nb_sim)]  # type: ignore

    def mean(self, nb_sim: float) -> float:
        """
        This method computes the mean of the generated numbers.

        :param nb_sim: the number of simulations (type: int)
        :return: the mean of the generated numbers (type: float)
        """
        assert nb_sim > 0, "The number of simulations must be strictly positive"
        return sum(self.generate_sim(nb_sim)) / nb_sim  # type: ignore

    def variance(self, nb_sim: int) -> float:
        """
        This method computes the variance of the generated numbers.

        :param nb_sim: the number of simulations (type: int)
        :return:  the variance of the generated numbers (type: float)
        """
        assert nb_sim > 0, "The number of simulations must be strictly positive"
        generated_numbers = self.generate_sim(nb_sim)
        mean = sum(generated_numbers) / nb_sim  # type: ignore
        return sum([(sim - mean) ** 2 for sim in generated_numbers]) / nb_sim  # type: ignore
