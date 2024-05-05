## Generation of random numbers TEST file ##
from Discrete_Generators.HeadTail import HeadTail
from Discrete_Generators.Bernoulli import Bernoulli
from Discrete_Generators.Binomial import Binomial
from Discrete_Generators.FiniteSet import FiniteSet
from Discrete_Generators.Poisson import PoissonFirstAlgo, PoissonSecondAlgo
from Continuous_Generators.Exponential import ExponentialInverseDistribution, \
    ExponentialRejectionSampling
from Continuous_Generators.Normal import NormalBoxMuller, NormalRejectionSampling, NormalCLT, \
    NormalBivariateBoxMuller, QuasiNormalBoxMuller, QuasiNormalRejectionSampling, QuasiNormalCLT, \
    QuasiNormalBivariateBoxMuller
from Uniform_Generators.VanDerCorput import VanDerCorput


# def main():
#     try:
#         mean_ht = HeadTail().mean(10000)
#         print(f"Mean of HeadTail: {mean_ht} ")
#         mean_ber = Bernoulli(0.3).mean(10000)
#         print(f"Mean of Bernoulli: {mean_ber}  ")
#         mean_bin = Binomial(100, 0.3).mean(10000)
#         print(f"Mean of Binomial: {mean_bin}  ")
#         prob = [0.25, 0.25, 0.25, 0.25]
#         mean_fs = FiniteSet(prob).mean(10000)
#         print(f"Mean of FiniteSet: {mean_fs}  ")
#         mean_poisson = PoissonFirstAlgo(2.3).mean(10000)
#         print(f"Mean of Poisson First Algo: {mean_poisson}  ")
#         mean_poisson2 = PoissonSecondAlgo(2.3).mean(10000)
#         print(f"Mean of Poisson Second Algo: {mean_poisson2}  ")
#         mean_exp_inv = ExponentialInverseDistribution(2.3).mean(10000)
#         print(f"Mean of Exponential Inverse Distribution: {mean_exp_inv}  ")
#         mean_exp_rej = ExponentialRejectionSampling(2.3).mean(100000)
#         print(f"Mean of Exponential Rejection Sampling: {mean_exp_rej}  ")
#         mean_normal = NormalBoxMuller(2, 0.05).mean(10000)
#         print(f"Mean of Normal Box Muller: {mean_normal}  ")
#         mean_normal2 = NormalRejectionSampling(2, 0.05).mean(10000)
#         print(f"Mean of Normal Rejection Sampling: {mean_normal2}  ")
#         mean_normal3 = NormalCLT(2, 0.05).mean(10000)
#         print(f"Mean of Normal CLT: {mean_normal3}  ")
#         value_1 = NormalBivariateBoxMuller(0, 1, 1, 1, 0).generate()[0]
#         value_2 = NormalBivariateBoxMuller(0, 1, 1, 1, 0).generate()[1]
#         print(f"Value 1 of Normal Bivariate Box Muller: {value_1}  ")
#         print(f"Value 2 of Normal Bivariate Box Muller: {value_2}  ")
#         mean_normal4 = NormalBivariateBoxMuller(0, 1, 1, 1, 0).mean(10000)
#         print(f"Mean of Normal Bivariate Box Muller: {mean_normal4}  ")
#         mean_vc = VanDerCorput(3).mean(10000)
#         print(f"Mean of VanDerCorput: {mean_vc}  ")
#         variance_vc = VanDerCorput(3).variance(10000)
#         print(f"Variance of VanDerCorput: {variance_vc}  ")
#         mean_qnm = QuasiNormalBoxMuller(2, 0.05).mean(10000)
#         print(f"Mean of Quasi Normal Box Muller: {mean_qnm}  ")
#         variance_qnm = QuasiNormalBoxMuller(2, 0.05).variance(10000)
#         print(f"Variance of Quasi Normal Box Muller: {variance_qnm}  ")
#         mean_qnm2 = QuasiNormalRejectionSampling(2, 0.05).mean(10000)
#         print(f"Mean of Quasi Normal Rejection Sampling: {mean_qnm2}  ")
#         variance_qnm = QuasiNormalRejectionSampling(2, 0.05).variance(10000)
#         print(f"Variance of Quasi Normal Rejection Sampling: {variance_qnm}  ")
#         mean_qnm3 = QuasiNormalCLT(2, 0.05).mean(10000)
#         print(f"Mean of Quasi Normal CLT: {mean_qnm3}  ")
#         variance_qnm = QuasiNormalCLT(2, 0.05).variance(10000)
#         print(f"Variance of Quasi Normal CLT: {variance_qnm}  ")
#         value_3 = QuasiNormalBivariateBoxMuller(0, 1, 1, 1, 0).generate()[0]
#         print(f"Value 1 of Quasi Normal Bivariate Box Muller: {value_3}  ")

#     except Exception as e:
#         print(e)


if __name__ == "__main__":
    #main()
    print(HeadTail().mean(10000))