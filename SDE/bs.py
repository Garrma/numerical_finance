import numpy as np                  # type: ignore
from scipy.stats import norm        # type: ignore

from SDE.underlying import Asset1D, AssetND 
import SDE.utils.random_generation as rd

################################################################
################ DEFINE BS ASSET CLASS RELATED #################
################################################################

class BS_Asset(Asset1D):
    """
    Modelize Asset under black & scholes framework
    """
    
    scheme : str = "euler"         # dynamic simulation type -> euler or milstein
    nb_brownians_per_sim = 1

    def __init__(self, initial_spot, volatility, risk_free_rate,scheme = "euler"):
        """
        Initialize the Black-Scholes asset with given parameters.

        initial_price (float): Initial price of the asset.  The current price of the asset at time t=0.
        volatility (float): Volatility of the asset.  The standard deviation of the asset's returns per unit of time.
        risk_free_rate (float): Risk-free interest rate. The annualized continuously compounded risk-free rate.
        scheme (str): indicate dynamic to simulate -> euler or milstein
        """
        
        assert scheme in ["euler","milstein"], f"no scheme for black & scholes dynamic simulation known for {scheme}, -> use euler or milstein"
        
        self.scheme = scheme

        super().__init__(initial_spot,volatility,risk_free_rate)
        

    def simulate_path_with_brownian(self, time_list,brownian_list):
        """
        Simulate BS one set of values simulating a path for given times and brownians

        **** ADVICE **** : use self.generate_random_values_in_required_format(nb_periods=nb_periods)(nb_sim) when using brownians_vec
        
        time_list (list): list of periods to simulate
        brownian_list (list(float))): list of N(0,1) required for simulating dynamic
        Returns : values of basket at each time form time_list
        """

        # checking  
        assert time_list[0] == 0, "first period of path must be inception (=0)"

        S0 = self.initial_spot
        sigma = self.vol
        rf = self.rf 
        scheme = self.scheme

        nb_periods = len(time_list)
        asset_price_path = [0]*nb_periods
        asset_price_path[0] = S0

        # deciding which scheme apply
        if scheme == "euler" : spot_next_step = lambda spot,mu,vol,dt,w : spot*np.exp((mu-vol**2/2)*dt + vol*np.sqrt(dt)*w ) #spot + spot*(mu*dt + vol*np.sqrt(dt)*w )
        if scheme == "milstein": spot_next_step = lambda spot,mu,vol,dt,w : spot + spot*(mu*dt + vol*np.sqrt(dt)*w ) + spot*0.5*(vol**2)*((np.sqrt(dt)*w)**2 - dt)

        for i in range(1, nb_periods):
            asset_price_path[i] = spot_next_step(asset_price_path[i-1],rf,sigma,(time_list[i] - time_list[i-1]),brownian_list[i])

        return asset_price_path
    
    def price(self,actualisation_rate:float, maturity:float, strike:float, option_type='call'):
        """
        Price European call or put options using the Black-Scholes formula.

        actualisation_rate (float): rf rate to actualise payoff
        maturity (float): maturity of option
        strike (float): strike price of  option.
        option_type (str): Type of option, 'call' or 'put'.
        Returns: Price of the option.
        """
        if not option_type :
            raise TypeError("Option type is not given")
        
        S = self.initial_spot
        r = self.rf
        sigma = self.vol

        d1 = (np.log(S / strike) + (r + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
        d2 = d1 - sigma * np.sqrt(maturity)
        
        if option_type == 'call':
            option_price = S * norm.cdf(d1) - strike * np.exp(-actualisation_rate * maturity) * norm.cdf(d2)
        elif option_type == 'put':
            option_price = strike * np.exp(-actualisation_rate * maturity) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise NotImplementedError(f"Call only price call or put : not {option_type}")

        return option_price
    
    
class BS_AssetND(AssetND):

    def simulate_path_with_brownian(self, time_list, brownian_list):
        """
        Simulate one set of values simulating a path for given times and brownians
        
        time_list (list): list of periods to simulate
        brownian_list (list): format expected is list of [ e for e in range(nb_period)] with len(e)=nb_asset
            -> ***in case of doubt*** use self.generate_random_values_in_required_format(nb_periods=nb_periods)(nb_sim) when using brownians_vec

        Returns : values of basket at each time form time_list
        """
                
        assert time_list[0] == 0, "First period of path must be inception (=0)"

        nb_assets = self.dimension
        nb_periods = len(time_list)
        asset_price_path = np.zeros((nb_periods, nb_assets))
        asset_price_path[0] = self.initial_spot

        # Simulating normal values
        #brownians_in_list = my_normal_generation_function(nb_assets*nb_periods)     # easier to simulate all brownians directly and reshape after
        #brownian_list = np.reshape(brownians_in_list, (nb_periods,nb_assets))

        correlated_brownians = [rd.build_gaussian_vector(self.correl_matrix, brownian_vec) for brownian_vec in brownian_list]

        for i in range(1, nb_periods):
            for j in range(nb_assets):
                # deciding which schme apply
                if self.assets[j].scheme == "euler" : spot_next_step = lambda spot,mu,vol,dt,w : spot + spot*(mu*dt + vol*np.sqrt(dt)*w )
                if self.assets[j].scheme == "milstein": spot_next_step = lambda spot,mu,vol,dt,w : spot + spot*(mu*dt + vol*np.sqrt(dt)*w ) + spot*0.5*(vol**2)*((np.sqrt(dt)*w)**2 - dt)

                asset_price_path[i][j] = spot_next_step(asset_price_path[i-1][j],self.assets[j].rf,self.assets[j].vol,(time_list[i] - time_list[i-1]),correlated_brownians[i][j])

        return asset_price_path