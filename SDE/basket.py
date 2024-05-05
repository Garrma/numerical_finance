import numpy as np                                  # type: ignore
from copy import deepcopy
from typing import Callable, List, Union

from SDE.underlying import Underlying, Asset1D, AssetND
from SDE.bs import BS_AssetND

################################################################
####################### DEFINE BASKET CLASS #####################
################################################################

class Basket(Underlying):

    weights : np.array                                  # weights for given baskets
    assetND : AssetND                                   # used AssetND object to simulate dynamique
    aggregate_function : Callable[[np.array], float]    # function to determine perf of basket ex np.min, np.max, np.mean
    nb_brownians_per_sim : int = None                   # represent number of N(0,1) expected for simulation of one asset (use is 1 but could be more e.g in Heston =2)

    def __init__(self,weights,assets:AssetND,aggregate_function:Callable[[np.array], float],live_seed = True):    
        
        assert len(weights) == assets.dimension, "weights list is not corresponding to basket"

        self.weights = weights
        self.assetND = deepcopy(assets)
        self.aggregate_function = aggregate_function
        self.nb_brownians_per_sim = assets.nb_brownians_per_sim

        initial_spot = self.aggregate_function([w*a.initial_spot for w,a in zip(weights,self.assetND.assets)])
        dimension = len(self.weights)

        super().__init__(initial_spot,dimension,live_seed)
        

    ############### DYNAMIC ##############

    def simulate_path_with_brownian(self,time_vec,brownians_vec)->List[float]:
        """
        Simulate one set of values simulating a path for given times and brownians
        
        **** ADVICE **** : use self.generate_random_values_in_required_format(nb_periods=nb_periods)(nb_sim) when using brownians_vec

        ****ATTENTION**** : for output presenting, time_list[0] must be 0 meaning the path starts at 0, technically we only need one brownian less in the list but for simplicity reasons, we will keep the same number and not use the first one

        time_list (list): list of periods to simulate
        brownian_list_vec (list): list of simulation of an d N(0,1) with d = len(self.asset)
        Returns : values of basket at each time form time_list

        >>> exemple 

        """

        asset_path_ND = self.assetND.simulate_path_with_brownian(time_vec,brownians_vec)

        # agregate using agregating function given and use log on each value
        basket_path = [self.aggregate_function([alpha_i*S_i for alpha_i,S_i in zip(self.weights,assets_value_vec)]) for assets_value_vec in asset_path_ND]

        return basket_path
    
    ############### LOG DYNAMIC ##############

    def simulate_log_path_with_brownian(self,time_vec,brownian_list_vec)->List[float]:
        """

        **** ADVICE **** : use self.generate_random_values_in_required_format(nb_periods=nb_periods)(nb_sim) when using brownians_vec

        time_list (list): list of periods to simulate
        brownian_list_vec (list): list of simulation of an d N(0,1) with d = len(self.asset)
        
        Returns : values of basket at each time form time_list with log dynamic
        """

        asset_path_ND = self.assetND.simulate_path_with_brownian(time_vec,brownian_list_vec)

        # agregate using agregating function given and use log on each value
        log_basket_path = [np.exp(np.sum([alpha_i*np.log(S_i) for alpha_i,S_i in zip(self.weights,assets_value_vec)])) for assets_value_vec in asset_path_ND]

        return log_basket_path
    
    def log_simulate_with_brownian(self, time:float,w: Union[List[float],float])->float:
        """
        Simulate one value of log dynamic for asset for given time and brownian

        **** ADVICE **** : use self.generate_random_values_in_required_format(nb_periods=nb_periods)(1) when using brownians_vec

        time_list (list): list of periods to simulate
        w (float)or(list(float)): value of d N(0,1) with d = len(self.asset)
        
        time (float): maturity
        Returns : value of asset time t
        """
        return self.simulate_log_path_with_brownian([0,time],[w,w])[-1]
    
    def log_simulate(self,time):
        """

        Returns: one random value for an underlying at each time of the time_vec
        """
        return self.log_simulate_with_brownian(time,self.generate_random_values_in_required_format(1)(1))
    
    ############### PROD DYNAMIC ##############

    def simulate_prod_path_with_brownian(self,time_vec,brownian_list_vec)->List[float]:
        """
        Simulate one simulate for basket dynamic using (a*b*c*..)^{1/d}

        **** ADVICE **** : use self.generate_random_values_in_required_format(nb_periods=nb_periods)(nb_sim) when using brownians_vec

        time_list (list): list of periods to simulate
        brownian_list_vec (list): list of simulation of an d N(0,1) with d = len(self.asset)

        Returns: one random value for the product dynamic an underlying at each time of the time_vec
        """

        asset_path_ND = self.assetND.simulate_path_with_brownian(time_vec,brownian_list_vec)

        # agregate using agregating function given and use log on each value
        log_basket_path = [np.power(np.product([alpha_i*S_i for alpha_i,S_i in zip(self.weights,assets_value_vec)]),1/self.dimension) for assets_value_vec in asset_path_ND]

        return log_basket_path
    
    def prod_simulate_with_brownian(self, time,w:Union[List[float],float])->float:
        """
        Simulate one simulate for basket dynamic using (a*b*c*..)^{1/d}
        
        **** ADVICE **** : use self.generate_random_values_in_required_format(nb_periods=nb_periods)(1) when using brownians_vec

        time_list (list): list of periods to simulate
        w (float)or(list(float)): list of simulation of an d N(0,1) with d = len(self.asset)
        
        time (float): maturity
        Returns : value of asset time t
        """
        return self.simulate_prod_path_with_brownian([0,time],[w,w])[-1]


class BS_Basket(Basket):

    def __init__(self,weights,assets:List[Asset1D],correl_matrix,aggregate_function:Callable[[np.array], float],live_seed = True):    

        assetND = BS_AssetND(assets,correl_matrix)
        super().__init__(weights,assetND,aggregate_function,live_seed)


class Weighted_Basket(BS_Basket):

    def __init__(self,weights,assets:List[Asset1D],correl_matrix,live_seed=True):    
        """
        define a basket object where the performance of the basket is computed as the weighted performance of the asset (regular basket)
        """

        aggregate_function = lambda x : np.sum(x)
        super().__init__(weights,assets,correl_matrix,aggregate_function,live_seed)

