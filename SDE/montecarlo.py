"""
document containing all Monte-Carlo methods relations functions and plot functions
"""

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from tqdm import tqdm  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from typing import Callable

from SDE.underlying import Underlying
from SDE.utils.matrix import Matrix
from SDE.utils.montecarlo_utils import print_log_info,apply_transformation_recursive

def display_single_boxplot(data, title):
    """
    Display a box plot for given data.

    Parameters:
        data (list or array-like): The data to be plotted.
        label (str): Label for the box plot.

    Returns:
        None
    """
    plt.figure(figsize=(5, 6))
    plt.boxplot(data)  #, labels=[""])

    plt.xlabel('Price estimation value')
    #plt.ylabel('Estimations')
    plt.title(title)

    # Disable the grid
    plt.grid(False)

    # Add annotations for mean, standard deviation, and quantiles
    mean = np.mean(data)
    std_dev = np.std(data)
    quantiles = np.percentile(data, [25, 50, 75])
    text = f'Mean: {mean:.2f}\nStd Dev: {std_dev:.2f}\nQ1: {quantiles[0]:.2f}\nMedian: {quantiles[1]:.2f}\nQ3: {quantiles[2]:.2f}'
    plt.text(1.2, mean, text, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))


################################################################
########### GENERAL WRITING OF ESTIMATORS FUNCTIONS ############
################################################################

def mc_pricing(underlying: Underlying,
               actualisation_rate: float,
               terminal_payoff_function: Callable,
               maturity: float,
               nb_simulations: int,
               verbose: bool = False,
               A_transformation: Callable = None,
               h0_function: Callable = None,
               m_value: float = None,
               quasi_random: bool = False,
               display_performance: bool = False,
               **kwargs):  # kwargs used to allow to call function with more arguments which will not be use
    ########## ASSERTIONS & PARAMETERS ###########
    
    ############## PATH SIMULATIONS ##############
    print_log_info("MC path simulation", verbose)

    # simulate n brownians
    random_simulations = underlying.generate_random_values_in_required_format(nb_periods=1)(nb_simulations) #[[generate_n_gaussian(underlying.dimension) for _ in exercise_times] for i in range(nb_simulations)]

    if quasi_random:
        random_simulations = underlying.generate_random_values_in_required_format(nb_periods=1,is_quasi = True)(nb_simulations) #generate_n_gaussian_quasi_paths(nb_asset, nb_simulations, len(exercise_times))

    ############## APPLY ALGORITHM ##############
    actualised_payoff = lambda x: np.exp(-actualisation_rate * maturity) * terminal_payoff_function(x)
    h_function = underlying.h(maturity, actualised_payoff)

    result_vec = [h_function(simulation) for simulation in random_simulations]

    ############# VARIANCE REDUCTION #############
    if A_transformation:
        print_log_info("Antithetic method", verbose)

        random_simulations_A_shift = apply_transformation_recursive(random_simulations,A_transformation)
        result_vec_A = [h_function(simulation_A) for simulation_A in random_simulations_A_shift]
        result_vec = [0.5 * (p + p_a) for p, p_a in zip(result_vec, result_vec_A)]

    if h0_function:
        assert m_value != None, "if h0 is given, m = E[h0(X)] must be given too"

        print_log_info("Control variate method", verbose)

        ##### DEFINE FUNCTION NEEDED FOR CONTROL VAR. ESTIMATOR #####
        def compute_almost_control_estimator(d_gaussian_vec, h_vector, m):
            """
            d_gaussian_vec (array): array of size N (d,n) with N = number of simulations, d=number of asset and n=nuber of periods 
            h_vector (array): array of size N corresponding to final american vector iterated to t0

            Returns : 
                compute -b_star*(h0(X)-m)
                this result in "almost" control variate estimator contrary to  h(X) - b_star*(h0(X)-m)
            """

            ### computing b which minimize variance ###
            # current version of b is biased
            def b_function(h0_vector, h_vector):
                """
                computes b* the value of coefficient b that minimize the variance BIASED VERSION
                """
                h_bar = np.mean(h_vector)
                up_vec = [(i_h0 - m) * (i_h - h_bar) for i_h0, i_h in zip(h0_vector, h_vector)]
                bottom_vec = [(i_h0 - m) ** 2 for i_h0 in h0_vector]
                b_star = np.sum(up_vec) / np.sum(bottom_vec)
                return b_star

            h0_vector = np.array([h0_function(d_gaussian) for d_gaussian in d_gaussian_vec])

            b_star = -b_function(h0_vector, h_vector)

            almost_controle_var_vec = -b_star * (h0_vector - m)

            return almost_controle_var_vec

        almost_controle_var_vec = compute_almost_control_estimator(random_simulations, result_vec, m_value)

        if A_transformation:  # applying antithetic to control
            almost_controle_var_vec_A = compute_almost_control_estimator(random_simulations_A_shift, result_vec_A,m_value)
            almost_controle_var_vec = 0.5 * (almost_controle_var_vec + almost_controle_var_vec_A)

        ###### build control variate estimator 
        # result_vec <=> h(X)
        # almost_controle_var <=> -b(h0(X)-m)
        # result_vec + almost_controle_var <=> h(X)-b(h0(X)-m)
        result_vec = result_vec + almost_controle_var_vec

    ############# ESTIMATOR PERFORMANCE ############

    if display_performance:
        extra_info = ""
        if A_transformation: extra_info += "Antithetic|"
        if h0_function: extra_info += "Controle var.|"
        if quasi_random: extra_info += "Quasi.|"
        display_single_boxplot(result_vec, f"LS {extra_info} estimator performance")

    #################### RESULT ##################
    result = np.mean(result_vec)

    print_log_info(f"MC giving result: {round(result, 2)}", verbose)

    return result


################################################################
###################### MCLS PRICING FUNCTIONS ##################
################################################################

def mcls_pricing(underlying: Underlying,
                 actualisation_rate: float,
                 terminal_payoff_function: Callable,
                 exercise_times,
                 nb_simulations: int,
                 L: int = 2,
                 polynomial_function: Callable = lambda x, i: x ** i,
                 verbose: bool = True,
                 A_transformation: Callable = None,
                 h0_function: Callable = None,
                 m_value: float = None,
                 quasi_random: bool = False,
                 display_performance: bool = False,
                 **kwargs):  # kwargs used to allow to call function with more arguments which will not be use
    """
    compute a price using Longstaff-schwartz mehod

    underlying (Underlying): object containing a function simulating the underlying path
    risk_free_rate (float): rate used for actualisation
    terminal_payoff_function (func): function for a given underlying computes the terminal payoff - without actualisations
    exercise_times (list): list of exectution times
    nb_simulations (int): number of simulations
    L (int): max degree of polynomial functions used for regressing conditional expectation
    polynomial_function (func): polynomial function defining used for regression
    verbose (bool): if set to true will print the steps from the function
    A_transformation (func): A function applied in Antithetic variance reduc. if given, antithetic method is applied
    h0_func (func): h0_func function applied in Control variance reduc. if given, control method is applied
    m_value (float): m = E[h0(X)] deterministic value in controle reduc. must be given with h0 if h0 is given
    quasi_random (bool): if set to true, quasi random variance reduc. will be applied
    display_performance (bool): if set to true, will display graph with IC performance
    
    exemple of use : 

        >>> BASIC USE : 
            mcls_pricing(underlying= my_basket_object , 
                risk_free_rate= rf, 
                terminal_payoff_function= lambda x : max(x-strike,0), 
                exercise_times= times_path, 
                nb_simulations = 9000,
                verbose = True,
                display_performance= True)

        >>> ANTITHETIC :
            mcls_pricing(underlying= my_basket_object , 
                risk_free_rate= rf, 
                terminal_payoff_function= lambda x : max(x-strike,0), 
                exercise_times= times_path, 
                nb_simulations = 9000,
                verbose = True,
                A_transformation = lambda x: -x) 

        >>> CONTROL :
            mcls_pricing(underlying= my_basket_object , 
                risk_free_rate= rf, 
                terminal_payoff_function= lambda x : max(x-strike,0), 
                exercise_times= times_path, 
                nb_simulations = 9000,
                h0_function= lambda brownian_path_vec : h0(brownian_path_vec,my_basket_object,times_path,lambda x : np.exp(-rf*T)*max(x-strike,0)),
                m_value=m_value)

    """

    ##################### USEFUL FUNCTION ######################

    get_sublist = lambda liste, indices: [liste[i] for i in indices]

    def ordinary_least_squares(x: Matrix, y):
        """
        Perform Ordinary Least Squares (OLS) regression.

        x (array): Independent variable matrix of shape (n, d).
        y (array): Dependent variable vector of shape (n,).
        Returns: Estimated coefficients (intercept, alpha_1,.. alpha_d).
        """
        # Add a constant term for the intercept
        X = np.column_stack((np.ones(x.get_shape()[0]), x.m))

        # Calculate the coefficients (intercept, slope)
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        return beta

    ##################################################################
    ##################### LS ALGORITHM FUNCTION ######################
    ##################################################################

    def LS_algorithm_applied_to_gaussians(d_gaussian_path_vec):
        """

        d_gaussian_vec (array): array of N size (d,n) N= number of simulations with d=number of asset and n=nuber of periods 

        Operation : 
            This function returns a vector of payoff backward iterated up to time 0
            | takes normal gaussian simulations
            | simulate paths for asset for each exercise time
            | iterate backward and for each time do the following:
            |    compute immediate payoff
            |    run OLS regression to compute conditional expectation values
            |    determine optimal continuation decision
            |    compute optimal payoff value
            | obtain at t0 the iterated payoff
            v
        """
        nb_periods = len(exercise_times)

        # compute simulations for given brownians
        simulations_vec = [underlying.simulate_path_with_brownian(exercise_times, brownians) for brownians in d_gaussian_path_vec]

        # fill spot matrix
        #row(i) = (S^i_t0,...,S^i_tnb_periods)
        #col(j) = (S^0_tj,...,S^n_tj)
        spot_matrix = Matrix(simulations_vec)

        ############# BACKWARD ITERATION #############

        print_log_info("MCLS initialisation", verbose)

        S_T_vec = spot_matrix.get_col(nb_periods - 1)
        payoff_vec = np.array(
            [np.exp(-actualisation_rate * (exercise_times[-1] - exercise_times[-2])) * terminal_payoff_function(S_T) for
             S_T in S_T_vec])  # actualise payoff needed in first iteration

        print_log_info("MCLS backward iterating", verbose)

        for i_period in range(nb_periods - 2, 0, -1):
            ## filter ITM spots ##
            asset_value_period = spot_matrix.get_col(i_period)
            itm_index = [i for i, value in enumerate(asset_value_period) if terminal_payoff_function(value) > 0]

            if len(itm_index) == 0:  # case where no path is ITM -> will lead to error
                payoff_vec *= np.exp(-actualisation_rate * (exercise_times[i_period] - exercise_times[i_period - 1]))
                continue

            payoff_itm_vec = get_sublist(payoff_vec, itm_index)
            asset_itm_vec = get_sublist(asset_value_period, itm_index)

            ## build ITM Matrix ##
            A = Matrix.empty_matrix(len(itm_index), L)
            for i in range(len(itm_index)):
                for j in range(L):
                    A[i, j] = polynomial_function(asset_itm_vec[i], j + 1)

            ## run regression ##
            print_log_info(f"Period {round(exercise_times[i_period], 2)}: perform LS", verbose)

            alpha_star = ordinary_least_squares(A, payoff_itm_vec)

            ## continuation values ##
            A_adjusted = np.column_stack((np.ones(A.get_shape()[0]), A.m))
            continuation_vec = np.dot(A_adjusted, alpha_star)

            ## optimal continuation paths ##
            print_log_info(f"Period {round(exercise_times[i_period], 2)}: compute optimal decision", verbose)

            cpt = 0
            for i in tqdm(range(nb_simulations), disable=not verbose):
                if i in itm_index:
                    if terminal_payoff_function(asset_itm_vec[cpt]) > continuation_vec[cpt]:
                        payoff_vec[i] = terminal_payoff_function(asset_itm_vec[cpt])
                    cpt += 1
                else:
                    # do nothing as we do not exercise and keep next val
                    continue

                    # actualise value of next period
            payoff_vec *= np.exp(-actualisation_rate * (exercise_times[i_period] - exercise_times[i_period - 1]))

        return payoff_vec

    ##################################################################
    ############################## MAIN ##############################
    ##################################################################
    """
    operations : 

    -> if a transformation function A is given, LS_algorithm is run and the transformation and antithetic variable will be computed

        >>> A: x -> -x 
            compute (LS_algorithm(x) + LS_algorithm(A(x)))*0.5

    -> if a function h0 and a value m is given , the estimator will be computed as 
        >>> compute  LS_algorithm(x) - b*(h0(X)-m)
    
    -> if a function A, h0 and m are given
        >>> compute (LS_algorithm(x) + LS_algorithm(A(x)))*0.5 - b*( 0.5*(h0(X)+h0(A(X))) -m)
    """

    ########## ASSERTIONS & PARAMETERS ###########
    def is_sorted_ascending(lst):
        return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

    assert is_sorted_ascending(exercise_times), "exercise periods must be chronologically sorted"
    assert exercise_times[0] == 0, "exercise periods must contain 0 as first element"

    ############## PATH SIMULATIONS ##############
    print_log_info("MCLS path simulation", verbose)

    # simulate n brownians
    random_simulations = underlying.generate_random_values_in_required_format(nb_periods=len(exercise_times))(nb_simulations) #[[generate_n_gaussian(underlying.dimension) for _ in exercise_times] for i in range(nb_simulations)]

    if quasi_random:
        random_simulations = underlying.generate_random_values_in_required_format(nb_periods=len(exercise_times,is_quasi = True))(nb_simulations) #generate_n_gaussian_quasi_paths(nb_asset, nb_simulations, len(exercise_times))

    ############## APPLY ALGORITHM ##############

    result_vec = LS_algorithm_applied_to_gaussians(random_simulations)

    ############# VARIANCE REDUCTION #############
    if A_transformation:
        print_log_info("Antithetic method", verbose)

        random_simulations_A_shift = apply_transformation_recursive(random_simulations,A_transformation)
        result_vec_A = LS_algorithm_applied_to_gaussians(random_simulations_A_shift)

        result_vec = [0.5 * (p + p_a) for p, p_a in zip(result_vec, result_vec_A)]

    if h0_function:
        assert m_value != None, "if h0 is given, m = E[h0(X)] must be given too"

        print_log_info("Control variate method", verbose)

        ##### DEFINE FUNCTION NEEDED FOR CONTROL VAR. ESTIMATOR #####
        def compute_almost_control_estimator(d_gaussian_path_vec, h_vector, m):
            """
            d_gaussian_path_vec (array): array of size N (d,n) with N = number of simulations, d=number of asset and n=nuber of periods 
            h_vector (array): array of size N corresponding to final american vector iterated to t0

            Returns : 
                compute -b_star*(h0(X)-m)
                this result in "almost" control variate estimator contrary to  h(X) - b_star*(h0(X)-m)
            """

            ### computing b which minimize variance ###
            # current version of b is biased
            def b_function(h0_vector, h_vector):
                """
                computes b* the value of coefficient b that minimize the variance BIASED VERSION
                """
                h_bar = np.mean(h_vector)
                up_vec = [(i_h0 - m) * (i_h - h_bar) for i_h0, i_h in zip(h0_vector, h_vector)]
                bottom_vec = [(i_h0 - m) ** 2 for i_h0 in h0_vector]
                b_star = np.sum(up_vec) / np.sum(bottom_vec)
                return b_star

            h0_vector = np.array([h0_function(d_gaussian_path) for d_gaussian_path in d_gaussian_path_vec])

            b_star = -b_function(h0_vector, h_vector)

            almost_controle_var_vec = -b_star * (h0_vector - m)

            return almost_controle_var_vec

        almost_controle_var_vec = compute_almost_control_estimator(random_simulations, result_vec, m_value)

        if A_transformation:  # applying antithetic to control
            almost_controle_var_vec_A = compute_almost_control_estimator(random_simulations_A_shift, result_vec_A,m_value)
            almost_controle_var_vec = 0.5 * (almost_controle_var_vec + almost_controle_var_vec_A)

        ###### build control variate estimator 
        # result_vec <=> h(X)
        # almost_controle_var <=> -b(h0(X)-m)
        # result_vec + almost_controle_var <=> h(X)-b(h0(X)-m)
        result_vec = result_vec + almost_controle_var_vec

    ############# ESTIMATOR PERFORMANCE ############

    if display_performance:
        extra_info = ""
        if A_transformation: extra_info += "Antithetic|"
        if h0_function: extra_info += "Controle var.|"
        if quasi_random: extra_info += "Quasi.|"
        display_single_boxplot(result_vec, f"LS {extra_info} estimator performance")

    #################### RESULT ##################
    result = np.mean(result_vec)

    print_log_info(f"LCLS giving result: {round(result, 2)}", verbose)

    return result
