import numpy as np 

from SDE.underlying import my_normal_generation_function
from SDE.underlying import Asset1D 

################################################################
################ DEFINE BS ASSET CLASS RELATED #################
################################################################

class Heston_Asset(Asset1D):

    kappa : float           # mean reversion rate
    theta : float           # Long term variance
    vol_of_vol : float      # volatility of volatility
    correlation :           # correlation between vol and asset processes 
    nb_brownians_per_sim = 2 
    
    def __init__(self, initial_spot, volatility, risk_free_rate, mean_reversion, long_term_volatility, volatility_of_volatility, correlation):
        """
        Initialize the Heston asset with given parameters.

        initial_price (float): Initial price of the asset.  The current price of the asset at time t=0.
        volatility (float): Volatility of the asset.  The standard deviation of the asset's returns per unit of time.
        risk_free_rate (float): Risk-free interest rate. The annualized continuously compounded risk-free rate.

        A FAIRE 
        """

        super().__init__(initial_spot,volatility,risk_free_rate)

        self.kappa = mean_reversion
        self.theta = long_term_volatility
        self.vol_of_vol = volatility_of_volatility
        self.correlation = correlation


    def simulate_path_with_brownian(self, time_list):
        """
        Simulate one path of the Heston model for given times.

        Parameters:
            time_list (list): List of time points at which to simulate the path.

        Returns:
            asset_price_path (list): Simulated asset prices corresponding to the time points.
        """
        nb_periods = len(time_list)
        asset_price_path = [0] * nb_periods
        asset_price_path[0] = self.initial_spot

        # Simulating normal values
        brownians_in_list = my_normal_generation_function(2*nb_periods)     # easier to simulate all brownians directly and reshape after
        brownians = np.reshape(brownians_in_list, (nb_periods,2))

        # Simulating the path
        for i in range(1, nb_periods):
            t = time_list[i]-time_list[i-1]
            w1,w2 = brownians[i][0],brownians[i][1]
            w2 = self.correlation*w1 + np.sqrt(1-self.correlation**2)*w2

            drift = self.mean_reversion * (self.long_term_volatility - vt) * dt[i-1]
            diffusion = np.sqrt(max(0, vt) * self.volatility_of_volatility * dt[i-1]) * dw2[i-1]
            asset_price_path[i] = asset_price_path[i-1] + drift + diffusion

        return asset_price_path