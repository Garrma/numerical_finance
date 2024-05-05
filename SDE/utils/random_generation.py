import numpy as np  # type: ignore
from scipy.stats import norm  # type: ignore
import time 

from SDE.utils.matrix import Matrix

# keep following lines to import NormalBoxMuller
import sys
sys.path.append("../../") 
from Generators.Continuous_Generators.Normal import NormalBoxMuller

################################################################
##################### SIMULATION ENVIRONMENT ###################
################################################################

def generate_n_gaussian(n,live_seed=False):
    """
    Simulates n values following Normal(0,1)

    n (int): number of simulations
    Returns: : Array of n random values following the Gaussian(0,1) distribution.
    """

    seed1,seed2 = (203,222),(268,104)                       # ranking of Dauphine Msc finance
    if live_seed : 
        seed_incr = int(time.time()*1000%100)                # set seed to first 3 decimals of time
        seed1 = (seed1[0]+seed_incr,seed1[1]+seed_incr) 
        seed2 = (seed2[0]+seed_incr,seed2[1]+seed_incr) 

    gaussian_values = NormalBoxMuller(0, 1,seed1,seed2).generate_sim(n)
    #gaussian_values = np.random.randn(n)

    if n == 1: gaussian_values = gaussian_values[0]
    return gaussian_values


def build_gaussian_vector(correlation_matrix: Matrix, normal_vec):
    """
    Given a vector of N(0,1) and a correlation matrix, returns the correlated gaussian vector
    
    correlation_matrix (Matrix) : correlation matrix of assets 
    normal_vec (list): list of normal values N(0,1) with len(normal_vec)==nb assets
    """

    assert len(normal_vec) == correlation_matrix.get_shape()[0], f"give normal simulations with size = number of assets {len(normal_vec)}vs{correlation_matrix.get_shape()[0]}"

    if correlation_matrix.is_invertible():
        matrix_decomposition = correlation_matrix.cholesky_decomposition()
    else:
        matrix_decomposition = correlation_matrix.orthogonal_decomposition()

    multivariate_gaussian_values = np.dot(matrix_decomposition, normal_vec)
    return multivariate_gaussian_values


def is_prime(num):
    """
    Check if a number is prime.
    """
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True


def get_first_n_primes(N):
    """
    Generate the first N prime numbers.
    """
    primes = []
    number = 2
    while len(primes) < N:
        if is_prime(number):
            primes.append(number)
        number += 1
    return primes


def van_der_corput(n, base=2):
    """
    Generate the first n terms of the Van der Corput sequence for a given base.
    """
    sequence = []
    for i in range(n):
        num = 0
        f = 1 / base
        value = i
        while value > 0:
            num += (value % base) * f
            value //= base
            f /= base
        num = min(max(num, 1e-10), 1 - 1e-10)  # Avoid exact 0 and 1
        sequence.append(num)
    return np.array(sequence)


def van_der_corput_custom_start(n, base=2, start_value=0.1):
    sequence = [start_value]
    for i in range(1, n):
        num = 0
        f = 1 / base
        value = i
        while value > 0:
            num += (value % base) * f
            value //= base
            f /= base
        sequence.append(num)
    return sequence


def generate_n_gaussian_quasi(M, N):
    """
    Generate an array of N quasi-random points for M assets.

    M (int): number of assets
    N (int): number of simulations
    """
    # Get the first M prime numbers
    prime_bases = get_first_n_primes(M)
    # Generate quasi-random sequences for each asset based on a unique prime base
    sequences = [van_der_corput_custom_start(N, base=prime, start_value=0.1) for prime in prime_bases]
    # Transform each sequence from uniform [0,1] to normal distribution
    normal_sequences = [norm.ppf(seq) for seq in sequences]
    # Stack all sequences to create an N x M matrix of quasi-random normals
    stacked_sequences = list(np.stack(normal_sequences, axis=-1))
    # Return results
    return stacked_sequences


def generate_n_gaussian_quasi_paths(M, N, nb_periods):
    """
    Generate a structured output of quasi-random Gaussian values for M assets over N simulations and nb_periods time steps.
    
    M: Number of assets
    N: Number of simulations
    nb_periods: Number of time steps
    """
    # Get the first M prime numbers
    prime_bases = get_first_n_primes(M * N)
    # Initialize lists
    sequences_assets = []
    all_paths = []
    # Loop over all assets and simulations
    base_index = 0
    for asset in range(M):
        sequences_simulations = []
        for simulation in range(N):
            base = prime_bases[base_index]
            sequence_simulation = van_der_corput_custom_start(nb_periods, base=base, start_value=0.5)
            normal_sequence = [norm.ppf(seq) for seq in sequence_simulation]
            base_index += 1
            sequences_simulations.append(normal_sequence)
        sequences_assets.append(sequences_simulations)
    # Stack all sequences
    stacked_sequences = list(np.stack(sequences_assets, axis=-1))
    # Modify format
    for simulation in range(N):
        all_paths.append([np.array(stacked_sequences[simulation][n_p]) for n_p in range(nb_periods)])
    # Return results
    return all_paths

if __name__ =="__main__":

    print("Here comes the main")