"""
document containing all Monte-Carlo methods relations functions and plot functions
"""

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from tqdm import tqdm  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from typing import Callable
import time

from SDE.utils.random_generation import generate_n_gaussian

def print_log_info(text, verbose=True, line_length=80, c="*"):
    """
    prints text by adjusting to line overal length as fixed and filling with caracter
    """
    if not verbose: return  # not print if deactivate

    #if len(text)>line_length : print(f"{text} (text too long)")
    nb_equal_upper_one_side = int((line_length - len(text)) / 2)
    text_to_print = f"{c * nb_equal_upper_one_side} {text} {c * nb_equal_upper_one_side}"
    print(text_to_print)
    
################################################################
########### GENERAL WRITING OF ESTIMATORS FUNCTIONS ############
################################################################

def simple_estimator_general(simulations, h_function: Callable):
    """
    simulations (array): sets of simulations of N(0,1)
    h_function (func): returns value given one simulation of N(0,1)
    """
    h_vector = [h_function(simulation) for simulation in simulations]
    return np.mean(h_vector)


def antithetic_estimator_general(
    simulations, h_function: Callable, A_transformation: Callable
):
    """
    simulations (array): sets of simulations of N(0,1)
    h_function (func): returns value given one simulation of N(0,1)
    A_transformation (func): function such that A(x) is equal in law to x
    """

    h_vector = [
        0.5 * (h_function(simulation) + h_function(A_transformation(simulation)))
        for simulation in simulations
    ]
    return np.mean(h_vector)


def control_estimator_general(
    simulations, h_function: Callable, h0_function: Callable, m: float
):
    """
    simulations (array): sets of simulations of N(0,1)
    h_function (func): returns value given one simulation of N(0,1)
    h0_function (func): control function
    m (func): deterministic value for E[h0(X)]
    """

    ### computing b which minimize variance ###
    # current version of b is biased
    def b(normal_sim):
        """
        computes b* the value of coefficient b that minimize the variance BIASED VERSION
        """
        h_bar = np.mean([h_function(simulation) for simulation in normal_sim])
        up_vec = [
            (h0_function(simulation) - m) * (h_function(simulation) - h_bar)
            for simulation in normal_sim
        ]
        bottom_vec = [(h0_function(simulation) - m) ** 2 for simulation in normal_sim]
        b_star = np.sum(up_vec) / np.sum(bottom_vec)
        return b_star

    b_star = b(simulations)

    ######
    h_vector = [
        h_function(simulation) - b_star * (h0_function(simulation) - m)
        for simulation in simulations
    ]
    return np.mean(h_vector)


def control_antithetic_estimator_general(
    simulations,
    h_function: Callable,
    h0_function: Callable,
    m: float,
    A_transformation: Callable,
):
    """
    simulations (array): sets of simulations of N(0,1)
    h_function (func): returns value given one simulation of N(0,1)
    h0_function (func): control function
    m (func): deterministic value for E[h0(X)]
    A_transformation (func): function such that A(x) is equal in law to x

    Return : mean between control var. estimator and control var. applied to A transformation
    """

    # apply A shift to simulations
    simulations_A = [A_transformation(simulation) for simulation in simulations]

    control_variate_estimator_value = control_estimator_general(
        simulations, h_function, h0_function, m
    )
    control_variate_estimator_value_antithetic = control_estimator_general(
        simulations_A, h_function, h0_function, m
    )

    return 0.5 * (
        control_variate_estimator_value + control_variate_estimator_value_antithetic
    )


################################################################
################ MULTI ESTIMATORS PLOT FUNCTIONS ###############
################################################################


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
    plt.boxplot(data)  # , labels=[""])

    plt.xlabel("Price estimation value")
    # plt.ylabel('Estimations')
    plt.title(title)

    # Disable the grid
    plt.grid(False)

    # Add annotations for mean, standard deviation, and quantiles
    mean = np.mean(data)
    std_dev = np.std(data)
    quantiles = np.percentile(data, [25, 50, 75])
    text = f"Mean: {mean:.2f}\nStd Dev: {std_dev:.2f}\nQ1: {quantiles[0]:.2f}\nMedian: {quantiles[1]:.2f}\nQ3: {quantiles[2]:.2f}"
    plt.text(
        1.2,
        mean,
        text,
        verticalalignment="center",
        bbox=dict(facecolor="white", alpha=0.5),
    )


def plot_asset_paths(time_path, asset_paths):
    """
    Plot simultaneously each asset path

    asset_paths (list of lists): List containing paths for each simulation
    Returns: None
    """
    plt.figure(figsize=(20, 6))  # Set the figure size

    for i, path in enumerate(asset_paths):
        plt.plot(time_path, path, label=f"Asset {i + 1}")  # Plot each asset's path

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Underlying simulated Paths")
    # plt.legend()
    plt.grid(False)
    plt.show()


def compute_cumulated_std_vec(estimations):
    """
    Computes the levels of variance for each growing N subsets of input
    estimations (array)
    Returns : vector with aggregated variance
    """
    """
    e_array = np.array(estimations)
    n = len(e_array)

    len_array= np.array(range(1,n+1))
    len_array_shift= np.array(range(n))
    cummean_array = np.cumsum(e_array)/len_array

    cum_var_array = (e_array-cummean_array)**2/len_array_shift
    
    #adjust first elements
    cum_var_array[0]=np.nan
    """
    start_index = 2

    n_sim = len(estimations)
    cum_std_array = [np.nan] * n_sim
    for i in range(start_index, n_sim):
        mean_i = np.mean(estimations[: i + 1])
        unbiased_var = np.sum([(e - mean_i) ** 2 for e in estimations[: i + 1]]) / (
            i - 1
        )
        cum_std_array[i] = np.sqrt(unbiased_var)
    cum_std_array = np.array(cum_std_array)

    return cum_std_array


def display_multi_estimators(estimators_dict, title, ylim=None):
    """
    Plot each estimator converging to the target

    estimators_dict (dict(str,list)): sequence of estimators value for each method
    title (str)
    ylim (tuple(float,float)): interval on which focus the graphs
    Returns : void
    """

    plt.figure(figsize=(20, 7))

    colors = plt.cm.Set2

    # check length of estimators :
    assert min([len(seq) for seq in estimators_dict.values()]) == max(
        [len(seq) for seq in estimators_dict.values()]
    ), "all sequences must be of same size"

    nb_simu = range(len(list(estimators_dict.values())[0]))
    time_range = list(nb_simu)

    # Plot estimators sequence :
    for i, (name, estimator_seq) in enumerate(estimators_dict.items()):
        estimation = estimator_seq[-1]
        plt.plot(
            time_range,
            estimator_seq,
            label=f"{name}: {round(estimation, 2)}",
            color=colors(i + 1),
        )

    # Add labels and title
    plt.xlabel("Number of simulation")
    plt.ylabel("Estimation")
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.title(title)

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def display_multi_variance_estimators(estimators_dict, title):
    """
    Plot each estimator variance

    estimators_dict (dict(str,list)): sequence of estimators value for each method
    title (str)
    Returns : void
    """

    plt.figure(figsize=(20, 7))

    colors = plt.cm.Set2

    # check length of estimators :
    assert min([len(seq) for seq in estimators_dict.values()]) == max(
        [len(seq) for seq in estimators_dict.values()]
    ), "all sequences must be of same size"

    nb_simu = range(len(list(estimators_dict.values())[0]))
    time_range = list(nb_simu)

    # Plot estimators sequence :
    for i, (name, estimator_seq) in enumerate(estimators_dict.items()):
        variance_estimation_vec = compute_cumulated_std_vec(estimator_seq)
        variance_estimation = variance_estimation_vec[-1]
        plt.plot(
            time_range,
            variance_estimation_vec,
            label=f"{name}: {round(100 * variance_estimation, 3)}%",
            color=colors(i + 1),
        )

    # Add labels and title
    plt.xlabel("Number of simulation")
    plt.ylabel("Std")
    plt.title(title)

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


################################################################
############## CONFIDENCE INTERVAL PLOT FUNCTIONS ##############
################################################################


def display_single_estimators_IC(estimator_seq, title):
    """
    Plot estimator converging within confidence interval

    """
    nb_simu = range(len(estimator_seq))
    time_range = list(nb_simu)

    sequence = np.array(estimator_seq)
    var_sequence = compute_cumulated_std_vec(sequence)

    sequence_low = sequence - var_sequence
    sequence_high = sequence + var_sequence

    plt.figure(figsize=(20, 7))

    ### determine interval of confidence ###
    plt.plot(time_range, sequence, label=f"estimator δ", color="salmon")
    plt.plot(
        time_range,
        sequence_low,
        label=f"δ-σ: {round(sequence_low[-1], 2)}",
        color="grey",
    )
    plt.plot(
        time_range,
        sequence_high,
        label=f"δ+σ: {round(sequence_high[-1], 2)}",
        color="grey",
    )

    # Add convergence value
    target = sequence[-1]
    plt.axhline(
        y=target,
        label=f"Estimation: {round(target, 2)}",
        color="powderblue",
        linestyle="--",
    )

    # Add labels and title
    plt.xlabel("number of simulation")
    plt.ylabel("Value")
    plt.title(title)

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def display_multi_estimators_IC(estimators_dict, title, ylim=None):
    """
    Plot each estimator converging to the target

    estimators_dict (dict(str,list)): sequence of estimators value for each method
    title (str)
    ylim (tuple(float,float)): interval on which focus the graphs
    Returns : void
    """

    # Create a new figure and adjust the size
    num_subplots = len(estimators_dict.keys())
    fig, axs = plt.subplots(1, num_subplots, figsize=(20, 7), sharex=True, sharey=True)

    # Call the function for each liquidation strategy
    for i, (name, estimator_seq) in enumerate(estimators_dict.items()):
        ax = axs[i]
        plt.sca(ax)  # Set the current axes to the ith subplot

        nb_simu = range(len(estimator_seq))
        time_range = list(nb_simu)

        sequence = np.array(estimator_seq)
        var_sequence = compute_cumulated_std_vec(sequence)

        sequence_low = sequence - var_sequence
        sequence_high = sequence + var_sequence

        ### determine interval of confidence ###
        ax.plot(time_range, sequence, label=f"estimator δ:", color="salmon")
        ax.plot(
            time_range,
            sequence_low,
            label=f"δ-σ: {round(sequence_low[-1], 2)}",
            color="grey",
        )
        ax.plot(
            time_range,
            sequence_high,
            label=f"δ+σ: {round(sequence_high[-1], 2)}",
            color="grey",
        )

        # Add convergence value
        target = sequence[-1]
        ax.axhline(
            y=target,
            label=f"Estimation: {round(target, 2)}",
            color="powderblue",
            linestyle="--",
        )

        # Add labels and title
        ax.set_xlabel("number of simulation")
        ax.set_ylabel("Value")
        ax.set_title(name)

        # Add legend
        ax.legend()

    # Zoom on y_level around target if not predefined
    if not ylim:
        range_coeff = 10  # coeff to multiply the range
        var_level_period = 0.1  # decide to take variance of X% of sequence
        var = var_sequence[int(len(var_sequence) * var_level_period)]
        ylim = (target - range_coeff * var, target + range_coeff * var)

    for ax in axs:
        ax.set_ylim(ylim[0], ylim[1])

    fig.suptitle(f"{title}")  # , fontsize=15, style='italic')

    # Show the plot
    plt.show()


################################################################
################### GENERAL PRECISION FUNCTIONS ################
################################################################


def fit_times_using_linear(x, times_vec):
    """
    running times seem to be linearly following a trend. However, due to imprecision to machine and other process running on the same time, results could be falsely guided
    trying to find a linear pattern to get rid of outliers and have a smooth line.
    """
    # converting into array for next iter
    x = np.array(x)
    times_vec = np.array(times_vec)

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), times_vec)

    result_vec = model.predict(x.reshape(-1, 1))

    return result_vec


def compute_performance_estimators(
    nb_asset: int, estimator_function: Callable, tol: float, NMax: int = 10000
):
    """
    Determine number of simulations required to enter a particular CI corresponding to given tolerance and comute time

    ***** ATTENTION ****** : not adequate to compare performance between estimators as not computed using same brownians -> HIGH BIAS

    nb_asset (int): dimension of brownian expected by estimator
    estimator (func): expect sets of brownian and outputs estimation
    tol (float): width of CI
    NMax (int): maximum number of simulations
    Returns : void -> prints result
    """
    NMin = 50  # we assume width of interval not meaningful under NMin simulations

    ### determine number of simulations required to enter CI
    brownian_simu_vec = [generate_n_gaussian(nb_asset) for _ in range(NMin)]

    # make sur tolerance is meaningful
    estimations_vec = [
        estimator_function(brownian_simu_vec[: i + 1])
        for i, _ in enumerate(brownian_simu_vec)
    ]
    std = np.std(estimations_vec)
    if 2 * std < tol:
        print(f"Tolerance of {tol} reached for less than {NMin} - decrease tol")
        return

    cpt = NMin
    IC_widths = []
    running_times = []

    with tqdm() as pbar:
        while 2 * std > tol and cpt < NMax:
            cpt += 1

            new_brownian = generate_n_gaussian(nb_asset)
            brownian_simu_vec.append(new_brownian)

            ### measure time required for above number of simulations
            start_time = time.time()
            estimation = estimator_function(brownian_simu_vec)

            end_time = time.time()
            running_time = (end_time - start_time) * 1000  # expressed in ms
            running_times.append(running_time)

            ### compute precision
            estimations_vec.append(estimation)
            std = np.std(estimations_vec)
            IC_widths.append(2 * std)

            # update progression bar
            pbar.set_description(
                f"***** Number of simulations = {cpt} | tol = {round(2 * std, 4)} ******"
            )
            pbar.update(1)

    if cpt == NMax:
        print(
            f"Tolerance not reached for maximum nb of interations {NMax} - increase tol"
        )
        return

        ### print results
    print(
        f"""
    =================== RESULTS ==================
                tolerance : {tol}
    number of simulations : {cpt}
                 run time : {round(running_time)} ms
    ==============================================
    """
    )

    ### display time/result curve
    x_axis_vec = list(range(NMin, cpt))

    # Create figure and primary axes
    fig, ax1 = plt.subplots(figsize=(20, 7))

    # Plot the stds on the first y-axis
    color = "tab:red"
    ax1.set_xlabel("Number of simulations")
    ax1.set_ylabel("Interval width = 2σ", color=color)
    ax1.plot(x_axis_vec, IC_widths, color=color, label="Interval width")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.legend()

    # Create a secondary y-axis with a different scale
    ax2 = ax1.twinx()

    # Plot the running time on the second y-axis
    fitted_times = fit_times_using_linear(x_axis_vec, running_times)
    color = "tab:blue"
    ax2.set_ylabel("Running Time (ms)", color=color)
    # ax2.plot(x_axis_vec,running_times, color=color)    # real running times curve
    ax2.plot(
        x_axis_vec, fitted_times, color=color, label="linear fitted execution costs"
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.legend()

    # Add title and show plot
    plt.title("Interval width & Running Time across number of simulations")
    plt.show()


def compute_performance_multi_estimators(
    nb_asset: int, estimator_function_dict, tol: float, NMax: int = 10000
):
    """
    Determine number of simulations required to enter a particular CI corresponding to given tolerance and comute time
    Compare performance using pseudo random simulations

    nb_asset (int): dimension of brownian expected by estimator
    estimator (dict(str,callable)): dict of functions expecting sets of brownian and outputs estimation
    tol (float): width of CI
    NMax (int): maximum number of simulations
    Returns : void -> prints result
    """

    ############################################## FUNCTIONS #############################################
    def display_results(name, cpt, running_time):
        line_length = 50
        lower_band = "=" * line_length
        middle_upper = f" RESULTS : {name.upper()} "
        nb_equal_upper_one_side = int((line_length - len(middle_upper)) / 2)
        upper_band = (
            "=" * nb_equal_upper_one_side + middle_upper + "=" * nb_equal_upper_one_side
        )
        print(
            f"""
        {upper_band}
                    tolerance : {tol}
        number of simulations : {cpt}
                    run time : {round(running_time)} ms
        {lower_band}
        """
        )

    ############################################# PARAMETERS #############################################
    NMin = 50  # we assume width of interval not meaningful under NMin simulations

    ################################################# RUN ################################################

    results_dict = {}

    # generate starting vector of brownians
    brownian_simu_vec = [generate_n_gaussian(nb_asset) for _ in range(NMin)]

    for i, (name, estimator_function) in enumerate(estimator_function_dict.items()):

        ### determine number of simulations required to enter CI

        # make sur tolerance is meaningful
        estimations_vec = [
            estimator_function(brownian_simu_vec[: i + 1]) for i in range(NMin)
        ]
        std = np.std(estimations_vec)
        if 2 * std < tol:
            print(f"Tolerance of {tol} reached for less than {NMin} - decrease tol")
            return

        cpt = NMin
        IC_widths = []
        running_times = []

        with tqdm() as pbar:
            while 2 * std > tol and cpt < NMax:
                cpt += 1

                ### ATTENTION : MUST BE COMPUTED USING SAME BROWNIANS TO BE MEANINGFUL -> fix a seed
                if len(brownian_simu_vec) < cpt:
                    new_brownian = generate_n_gaussian(nb_asset)
                    brownian_simu_vec.append(new_brownian)

                ### measure time required for above number of simulations
                start_time = time.time()
                estimation = estimator_function(brownian_simu_vec[:cpt])

                end_time = time.time()
                running_time = (end_time - start_time) * 1000  # expressed in ms
                running_times.append(running_time)

                ### compute precision
                estimations_vec.append(estimation)
                std = np.std(estimations_vec)
                IC_widths.append(2 * std)

                # update progression bar
                pbar.set_description(
                    f"***** Number of simulations = {cpt} | tol = {round(2 * std, 4)} ******"
                )
                pbar.update(1)

        if cpt == NMax:
            print(
                f"Tolerance not reached for maximum nb of interations {NMax} - increase tol"
            )
            return

            ### print results
        display_results(name, cpt, running_time)

        ### store results
        results_dict[name] = (IC_widths, running_times)

    #### setting graph ####
    fig, ax1 = plt.subplots(figsize=(20, 7))  # create figure
    ax2 = ax1.twinx()  # Create a secondary y-axis with a different scale
    colors = plt.cm.Set2  # color palette

    #### plotting on graph ####

    for i, (name, (IC_widths, running_times)) in enumerate(results_dict.items()):
        x_axis_vec = list(range(NMin, NMin + len(IC_widths)))

        ax1.plot(x_axis_vec, IC_widths, color=colors(i), label=f"Interval width {name}")

        fitted_times = fit_times_using_linear(x_axis_vec, running_times)
        # ax2.plot(x_axis_vec,running_times, color=color)    # real running times curve
        ax2.plot(
            x_axis_vec,
            fitted_times,
            color=colors(i),
            label=f"costs {name}",
            linestyle="--",
        )

    #### finalising graph ####
    ax1.set_xlabel("Number of simulations")
    ax1.set_ylabel("Interval width = 2σ")
    ax1.tick_params(axis="y")
    ax2.set_ylabel("Running Time (ms)")
    ax2.tick_params(axis="y")
    ax1.legend()
    ax2.legend()

    # Add title and show plot
    plt.title(f"Performance comparison of different estimators for a {tol}-wide IC")
    plt.show()
