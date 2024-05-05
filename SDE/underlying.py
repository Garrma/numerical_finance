import numpy as np                                  # type: ignore
from typing import List, Callable, Union, Iterable
from copy import deepcopy
from scipy.stats import norm                        # type: ignore

from SDE.utils.matrix import Matrix
from SDE.utils.random_generation import generate_n_gaussian, generate_n_gaussian_quasi_paths, generate_n_gaussian_quasi   # type: ignore


################################################################
###################### DEFINE GENERIC CLASS ####################
################################################################

class Underlying:

    initial_spot : Union[List[float],float]                 # intial spot in 1 or d-dimension
    dimension: int                                          # 1 or d for if single or basket
    generator_value_function : Callable[[int],List[float]]  # function allowing to simulate N(0,1) values dynamically or for a fixed seed

    def __init__(self,initial_spot,dimension,live_seed = True):
        self.initial_spot = initial_spot
        self.dimension = dimension
        self.generator_function = lambda x : generate_n_gaussian(x,live_seed=live_seed)

    def price(self,actualisation_rate:float, maturity:float, strike:float, option_type=str):
        """
        this function will be define in subclasses when applicable in subclass 

        Returns: deterministic price of called payoff for asset
        """
        raise TypeError(f"No deterministic function possible for {self.__class__.__name__} underlying")

    def simulate_path(self,time_vec):
        """

        Returns: one random value for an underlying at each time of the time_vec
        """
        
        # easier to implement only the function using N(0,1) as an input as it will be more useful and 
        # avoid implementing twice same function 

        return self.simulate_path_with_brownian(time_vec,self.generate_random_values_in_required_format(len(time_vec))(1))
    
    def simulate(self, time):
        """
        Simulate one value for given time and brownian
        
        time (float): maturity
        brownian (float): simulation of an N(0,1)
        Returns : value of asset time t
        """

        # we can use the path on a single terminal value and extract the final value
        # ADVANTAGES : no need to redefine the function for each subclass of underlying path
        # _NOTE : brownian is only used in the second index, first is useless 

        return self.simulate_path_with_brownian([0,time])[-1]
    
    def generate_random_values_in_required_format(self,nb_periods,is_quasi_random:bool=False)->Callable[[int],float]:
        """

        Return : function which for a number of simulations returns the simulated N(0,1) in required format which is expected by function simulating using N(0,1) in input
        """

        assert hasattr(self, "nb_brownians_per_sim"), "object must have an attribute representing the number of simulations required per simulation of asset"
        
        dimension = self.dimension
        nb_brownians_per_sim = self.nb_brownians_per_sim

        def generate_values(nb_simulations):
            """
            
            nb_simulations (int): returns N(0,1) in format expected by underlying
            """
            nb_simulations_required = dimension*nb_periods*nb_brownians_per_sim*nb_simulations
            
            ######## CASE IF QUASI RANDOM #######
            if is_quasi_random:
                if nb_periods == 1: 
                    return generate_n_gaussian_quasi(dimension,nb_simulations)
                
                return generate_n_gaussian_quasi_paths(dimension, nb_simulations,nb_periods)
            
            ######## CASE IF PSEUDO RANDOM #######
            simulations_in_list = self.generator_function(nb_simulations_required)

            if dimension == 1 and nb_brownians_per_sim == 1 and nb_periods == 1:
                # do nothing as already in list
                simulations = simulations_in_list
            elif dimension == 1 and nb_brownians_per_sim == 1:
                simulations = np.array(simulations_in_list).reshape(nb_simulations, nb_periods).tolist()
            elif dimension == 1 and nb_brownians_per_sim > 1:
                simulations = np.array(simulations_in_list).reshape(nb_simulations,nb_brownians_per_sim, nb_periods).tolist()
            elif dimension >1 and nb_brownians_per_sim == 1 and nb_periods ==1:
                simulations = np.array(simulations_in_list).reshape(nb_simulations,dimension).tolist()
            else : simulations = np.array(simulations_in_list).reshape(nb_simulations, nb_periods, dimension).tolist()

            # return simple list if only one simulation is asked
            if nb_simulations == 1: simulations = simulations[0]

            return simulations
        
        return generate_values

    def simulate_path_with_brownian(self, time_vec, brownian_vec):
        """
        this function will be defined in subclasses

        Returns: one random value for an underlying at each time of the time_vec
        """
        raise NotImplementedError("Function must be overloaded in subclass")
    
    def simulate_with_brownian(self, time, brownian):
        """
        Simulate one value for given time and brownian
        
        time (float): maturity
        brownian (float): simulation of an N(0,1)
        Returns : value of asset time t
        """

        # we can use the path on a single terminal value and extract the final value
        # ADVANTAGES : no need to redefine the function for each subclass of underlying path
        # _NOTE : brownian is only used in the second index, first is useless 

        return self.simulate_path_with_brownian([0, time], [brownian, brownian])[-1]

    def h(self, maturity, payoff_function: Callable) -> Callable:
        """
        
        ** ATTENTION ** : payoff_function corresponds to actualised payoff 
        
        payoff_function (func): actualised payoff function computing the price for given underlying terminal values
        Returns : function such that h((w1,..wd)) = price corresponding to payoff_function for the given simulations to the basket
        """

        def terminal_payoff_applied_to_simulation(simulations):
            basket_value = self.simulate_with_brownian(maturity, simulations)
            terminal_payoff = payoff_function(basket_value)

            return terminal_payoff

        return terminal_payoff_applied_to_simulation
    
################################################################
####################### DEFINE ASSET CLASS #####################
################################################################

class Asset1D(Underlying):
    vol: float
    rf: float

    def __init__(self, initial_spot,vol,rf,live_seed = True):
        super().__init__(initial_spot,dimension=1, live_seed=live_seed)
        
        self.vol = vol
        self.rf = rf

class AssetND(Underlying):

    # list including initial spot of each asset 
    assets : List[Asset1D]
    correl_matrix : Matrix                          # correl_matrix matrix between assets 
    nb_brownians_per_sim : int = None               # represent number of N(0,1) expected for simulation of one asset (use is 1 but could be more e.g in Heston =2)

    def __init__(self, assets: List[Asset1D], correl_matrix : Union[Matrix,Iterable],live_seed = False):
        
        # convert into matrix format
        if not isinstance(correl_matrix,Matrix):
            correl_matrix = Matrix(correl_matrix)

        
        ### check size of inputs
        num_rows, num_cols = correl_matrix.get_shape()
        
        assert num_rows == num_cols , "correlation matrix is not in correct format"
        assert num_rows == len(assets), "correlation matrix not built properly for given assets"

        ### check type of assets
        assert all([isinstance(i_asset, Asset1D) for i_asset in assets]), "each asset must be of type Asset"   
        assert all(isinstance(i_asset, type(assets[0])) for i_asset in assets), "all input assets must be of syme type"

        dimension = len(assets)

        initial_spot = [a.initial_spot for a in assets]
        super().__init__(initial_spot,dimension,live_seed)

        self.assets = deepcopy(assets)
        self.correl_matrix = deepcopy(correl_matrix)
        self.nb_brownians_per_sim = assets[0].nb_brownians_per_sim

if __name__ == "__main__":
    print("here comes the main")

    a = Asset1D(100,0.2,0.05)