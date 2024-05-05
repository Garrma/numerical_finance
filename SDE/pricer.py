import numpy as np
from typing import List, Callable
from scipy.stats import norm


from SDE.underlying import Underlying
from SDE.basket import Basket
import SDE.montecarlo as mc
import SDE.utils.montecarlo_utils as mcu

class PricingConfiguration:
    _payoff_name: str = (
        None  # payoff name used to set variance reductions method by default
    ) 
    _nb_simulations: int = 1000         # number of simulation when computing monte carlo techniques
    _pricing_model_name: str = "CF"     # CF (closed form), MC, MCLS
    _A_transformation: Callable = None  # A function for antithetic estimator
    _h0_function: Callable = None       # h0 function for control estimator
    _m_value: float = None              # m=E[h0(X)] value for control estimator
    _is_antithetic: bool = False        # turn to True to price with antithetic
    _is_control_variate: bool = False   # turn to True to price with control variate
    _is_quasi_random: bool = False      # turn to True to price with quasi random

    def __init__(
        self,
        payoff_name: str = None,            # payoff name used to set variance reductions method by default
        nb_simulations: int = 1000,         # number of simulation when computing monte carlo techniques
        pricing_model_name: str = "CF",     # CF (closed form), MC, MCLS
        A_transformation: Callable = None,  # A function for antithetic estimator
        h0_function: Callable = None,       # h0 function for control estimator
        m_value: float = None,              # m=E[h0(X)] value for control estimator
        is_antithetic: bool = False,        # turn to True to price with antithetic
        is_control_variate: bool = False,   # turn to True to price with control variate
        is_quasi_random: bool = False,      # turn to True to price with quasi random
    ):

        self._payoff_name = payoff_name
        self._nb_simulations = nb_simulations
        self._pricing_model_name = pricing_model_name
        self._A_transformation = A_transformation
        self._h0_function = h0_function
        self._m_value = m_value
        self._is_antithetic = is_antithetic
        self._is_control_variate = is_control_variate
        self._is_quasi_random = is_quasi_random

    ##### REPRESENTATION ####
    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    #### GETTERS METHODS ####
    def get_payoff_name(self) -> str:  return self._payoff_name
    def get_nb_simulations(self) -> int:  return self._nb_simulations
    def get_pricing_model_name(self) -> str: return self._pricing_model_name
    def get_pricing_method_name(self) -> str:  return self._pricing_method_name
    def get_A_transformation(self) -> Callable: return self._A_transformation
    def get_h0_function(self) -> Callable:  return self._h0_function
    def get_m_value(self) -> float: return self._m_value
    def is_antithetic(self) -> bool: return self._is_antithetic
    def is_control_variate(self) -> bool: return self._is_control_variate
    def is_quasi_random(self) -> bool: return self._is_quasi_random

    #### SETTERS ####
    def set_antithetic(self, attr: bool): self._is_antithetic = attr
    def set_control_variate(self, attr: bool): self._is_control_variate = attr
    def set_quasi_random(self, attr: bool):  self._is_quasi_random = attr
    def set_nb_simulations(self, nb_simulations):  self._nb_simulations = nb_simulations
    def set_pricing_reduction_techniques(
        self,
        A_transformation: Callable = None,
        h0_function: Callable = None,
        m_value: float = None,
    ):
        """
        allows the user to define reduction variance techniques as he wish

        underlying (Underlying): needs the underlying to take its values to compute the technique
        A_transformation (func): transformation used in antithetic technique. signature is A : float -> float
        h0_function (func): h0_function used in control variate technique. must take as an input the required shape of gaussians and return value (see ex.)
        m_value (float): value used in control technique s.t m = E[h0(X)]

        >>> a
        """
        # only fills value with what is given -> keep predefined techniques if not provided again

        if A_transformation:
            self._A_transformation = A_transformation
        if h0_function:
            self._h0_function = h0_function
        if m_value:
            self._m_value = m_value


class Option:
    # attributes are set private
    _strike: float
    _terminal_payoff_function: Callable
    _actualised_payoff_function: Callable
    _actualisation_rate: float
    _maturity: float
    _exercise_times: List[float]
    _payoff_name: str = None
    _exercise_type: str = "european"  # european, bermudean
    pricing_configuration: (
        PricingConfiguration  # pricing configuration used when pricing
    )

    def __init__(
        self,
        strike: float,
        actualisation_rate: float,
        terminal_payoff_function: Callable = None,
        maturity: float = None,
        exercise_times: List[float] = None,
        exercise_type: str = "european",
        payoff_name: str = "",
        pricing_configuration: PricingConfiguration = PricingConfiguration(),
    ) -> None:
        """
        strike (float):
        terminal_payoff_function (Callable):
        actualisation_rate (float) : risk
        maturity (float) :
        exercise_periods (List[float]) **optional** : if given, must contain 0 as first element
        exercise_type (str) **optional** : must be "european" or "bermudean"
        """
        self._strike = strike
        self._terminal_payoff_function = terminal_payoff_function
        self._actualisation_rate = actualisation_rate
        self._maturity = maturity
        self._exercise_times = exercise_times
        self._exercise_type = exercise_type
        self._payoff_name = payoff_name
        self.pricing_configuration = pricing_configuration

        if exercise_times:
            self._exercise_times = exercise_times
            if not maturity:
                maturity = exercise_times[-1]
        else:
            assert maturity, "Maturity must be given if no exercise_times are provided"
            self._exercise_times = [0, maturity]

        # initiate quick payoff if known
        if self._payoff_name == "call":
            self._terminal_payoff_function = lambda x: max(x - strike, 0)
        if self._payoff_name == "put":
            self._terminal_payoff_function = lambda x: max(strike - x, 0)

        if self._terminal_payoff_function:
            self._actualised_payoff_function = lambda x: np.exp(
                -actualisation_rate * maturity
            ) * terminal_payoff_function(x)

    ######### SETTERS ##########
    def set_pricing_configuration(self, pricing_config: PricingConfiguration):
        assert (
            type(pricing_config) == PricingConfiguration
        ), "pricing configuration must be of type PricingConfiguration"
        self.pricing_configuration = pricing_config

    # temporary function -> avoid to define all setters but allow to keep regularity
    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        attributes = vars(self)
        attribute_strings = [f"{attr}: {value}" for attr, value in attributes.items()]
        return "\n".join(attribute_strings)

    ######### SET EXISTING PRICING CONFIG ##########
    """
    To facilitate the use of the pricer, we give inbuilt methods for variance reduction techniques to avoid for the user to have to do it mannually
    """

    def set_predefined_pricing_reduction_techniques(self, underlying: Underlying):
        """
        Fill the existing method A, h0 and m for a given payoff

        payoff_name (str): can only be of "call","put"
        """

        ######################## DEFINING GENERIC FUNCTIONS ######################
        def m_function(basket_object: Basket, r, K, T):
            """
            computing E[h0(X)] for given basket and values
            """
            assert (
                type(basket_object) == Basket
            ), f"Type of underlying for such variance reduction method must be Basket, now is {type(basket_object)}"

            covar_matrix = basket_object.correl_matrix.m
            mu_star = np.array(
                [
                    alpha_i * a.vol * np.sqrt(T)
                    for alpha_i, a in zip(basket_object.weights, basket_object.assets)
                ]
            )
            sigma_star = np.sqrt(np.dot(np.dot(mu_star.T, covar_matrix), mu_star))
            cst = np.exp(
                np.sum(
                    [
                        alpha_i * (np.log(a.initial_spot) + (a.rf - a.vol**2 / 2) * T)
                        for alpha_i, a in zip(
                            basket_object.weights, basket_object.assets
                        )
                    ]
                )
            )

            d1 = np.log(cst / K) / sigma_star + sigma_star
            d2 = d1 - sigma_star

            m = (
                cst * np.exp(0.5 * sigma_star**2) * norm.cdf(d1) - K * norm.cdf(d2)
            ) * np.exp(-r * T)
            return m

        """ ** ATTENTION ** : for controle var. , payoff_function must be ACTUALISED if applying """

        def h0_bermudean(
            brownian_path_vec,
            basket_object: Basket,
            times_path,
            actualised_payoff_function: Callable,
        ) -> Callable:
            
            log_basket_terminal_value = basket_object.simulate_log_path_with_brownian(times_path, brownian_path_vec)[-1]

            return actualised_payoff_function(log_basket_terminal_value)

        def h0_european(
            brownian_vec, basket_object: Basket, T, actualised_payoff_function: Callable
        ) -> Callable:

            return actualised_payoff_function(basket_object.log_simulate(T, brownian_vec))

        ######################## SETTING EXISTING METHOD #########################
        payoff_name = self._payoff_name
        maturity = self._maturity
        exercise_type = self._exercise_type
        underlying_type = type(underlying)

        assert (
            payoff_name
        ), "Payoff name must be defined when setting pre defined variance reduction techniques"
        assert (
            exercise_type
        ), "Exercise type name must be defined when setting pre defined variance reduction techniques"

        A = None
        h0_function = None
        m_value = None

        ############################ ANTITHETIQUE ###########################
        A = lambda x: -x

        mcu.print_log_info(
            f" Antithetic var. set for pricing  {exercise_type} {payoff_name} on {underlying_type}",
            verbose=True,
        )

        if payoff_name == "call":
            ########################## CONTROL VARIATE ##########################
            if underlying_type == Basket:
                m_value = m_function(
                    underlying, self._actualisation_rate, self._strike, self._maturity
                )

                if exercise_type == "european":
                    h0_function = lambda d_brownian: h0_european(
                        d_brownian,
                        underlying,
                        maturity,
                        self._actualised_payoff_function,
                    )
                    mcu.print_log_info(
                        f" Control var. set for pricing  {exercise_type} {payoff_name} on {underlying_type}",
                        verbose=True,
                    )
                elif exercise_type == "bermudean":
                    h0_function = lambda d_brownian_path: h0_bermudean(
                        d_brownian_path,
                        underlying,
                        self._exercise_times,
                        self._actualised_payoff_function,
                    )
                    mcu.print_log_info(
                        f" Control var. set for pricing  {exercise_type} {payoff_name} on {underlying_type}",
                        verbose=True,
                    )
                else:
                    mcu.print_log_info(
                        f" No control var. found for pricing  {exercise_type} {payoff_name} on {underlying_type} object",
                        verbose=True,
                    )

            else:
                mcu.print_log_info(
                    f" No control var. found for pricing  {exercise_type} {payoff_name} on {underlying_type} object",
                    verbose=True,
                )

        elif payoff_name == "put":
            mcu.print_log_info(
                f" No control var. found for pricing  {exercise_type} {payoff_name} on {underlying_type} object",
                verbose=True,
            )
        else:
            raise TypeError(f"No pricing reduction techniques found for {payoff_name}")

        self.pricing_configuration.set_pricing_reduction_techniques(
            A_transformation=A, h0_function=h0_function, m_value=m_value
        )

    ##### PRICING FUNCTION ######
    def set_pricing_antithetic(self):
        """Turn on the antithetic reduction variance method when pricing"""
        self.pricing_configuration.set_antithetic(True)

    def set_pricing_control_variate(self):
        """Turn on the control variate reduction variance method when pricing"""
        self.pricing_configuration.set_control_variate(True)

    def set_pricing_quasi_random(self):
        """Turn on the quasi random reduction variance method when pricing"""
        self.pricing_configuration.set_quasi_random(True)

    def reset_pricing_techniques(self):
        """
        make sure to turn to False and price without antithetic, control & quasi
        """
        self.pricing_configuration.set_antithetic(False)
        self.pricing_configuration.set_control_variate(False)
        self.pricing_configuration.set_antithetic(False)

    def Price(self, underlying_object: Underlying, display_info=False):
        """

        underlying_object (Underlying):
        display_info (bool): if true display summary of pricing methodology
        """

        ####### GET PARAMETERS #######
        actualisation_rate = self._actualisation_rate
        maturity = self._maturity
        exercise_times = self._exercise_times
        terminal_payoff_function = self._terminal_payoff_function
        exercise_type = self._exercise_type
        payoff_name = self._payoff_name
        pricing_config = self.pricing_configuration

        # info from pricing configuration
        nb_simulations = pricing_config.get_nb_simulations()
        pricing_model_name = pricing_config.get_pricing_model_name()
        A_transformation = pricing_config.get_A_transformation()
        h0_function = pricing_config.get_h0_function()
        m_value = pricing_config.get_m_value()
        is_antithetic = pricing_config.is_antithetic()
        is_control_variate = pricing_config.is_control_variate()
        is_quasi_random = pricing_config.is_quasi_random()

        ############ ASSERT ###########
        if pricing_model_name != "CF":
            assert (
                terminal_payoff_function
            ), "Terminal payoff function must be given when not pricing in closed form"
        if pricing_model_name == "CF":
            assert (
                payoff_name
            ), "Name of payoff function must be given when pricing in closed form"
        if pricing_model_name != "CF":
            if is_antithetic:
                assert (
                    A_transformation
                ), "A function must be provided when pricing in Antithetic variance reduction"
            if is_control_variate:
                assert (
                    h0_function and m_value
                ), "h0_function and m_value must be provided when pricing in control variate variance reduction"

        ########### PRICING ###########
        if not is_antithetic:
            A_transformation = None
        if not is_control_variate:
            h0_function, m_value = None, None

        if pricing_model_name == "CF":
            if exercise_type and exercise_type != "european":
                raise ValueError(
                    "Closed form formula is not applicable for no other than european options"
                )

            # call deterministic function defined in corresponding asset class
            price = underlying_object.price(
                actualisation_rate, maturity, self._strike, payoff_name
            )

        elif pricing_model_name == "MC":
            if exercise_type != "european":
                raise ValueError(
                    "MC price is not applicable for no other than european options"
                )

            price = mc.mc_pricing(
                underlying=underlying_object,
                actualisation_rate=actualisation_rate,
                terminal_payoff_function=terminal_payoff_function,
                maturity=maturity,
                exercise_times=exercise_times,
                nb_simulations=nb_simulations,
                A_transformation=A_transformation,
                h0_function=h0_function,
                m_value=m_value,
                quasi_random=is_quasi_random,
                verbose=False,
                display_performance=False,
            )

        elif pricing_model_name == "MCLS":
            # can price european and bermudean

            price = mc.mcls_pricing(
                underlying=underlying_object,
                actualisation_rate=actualisation_rate,
                terminal_payoff_function=terminal_payoff_function,
                maturity=maturity,
                exercise_times=exercise_times,
                nb_simulations=nb_simulations,
                A_transformation=A_transformation,
                h0_function=h0_function,
                m_value=m_value,
                quasi_random=is_quasi_random,
                verbose=False,
                display_performance=False,
            )

        else:
            raise ValueError(
                f"No pricing method found for {pricing_model_name}, allowed are {['CF', 'MC', 'MCLS']}"
            )

        if display_info:
            # MAKE SUMMARY
            has_price_with_antithetic = A_transformation and is_antithetic
            has_price_with_control = h0_function and m_value and is_control_variate
            has_been_applied_str = "Not " if pricing_model_name == "CF" else ""
            print(
                f"""
            =========================== RESULTS ==========================
                            result : {price}
                       option name : {payoff_name}
                     exercise type : {exercise_type}
                     pricing model : {pricing_model_name}
             number of simulations : {nb_simulations} ({has_been_applied_str}Applicable)
                        antithetic : {has_price_with_antithetic} ({has_been_applied_str}Applicable)
                   control variate : {has_price_with_control} ({has_been_applied_str}Applicable)
                      quasi random : {is_quasi_random} ({has_been_applied_str}Applicable)
            ==============================================================
            """
            )

        return price

if __name__ == '__main__': 

    from underlying import BS_Asset, BS_Basket

    print("Here comes your main")

    risk_free_rate = 0.05

    # taking real life parameters found above
    asset_AAPL = BS_Asset(169.300003,0.20,risk_free_rate)
    asset_AMZN = BS_Asset(179,0.32,risk_free_rate)
    asset_TSLA = BS_Asset(179.99,0.53,risk_free_rate)
    correl_matrix = [[1.000000, 0.884748, 0.827503], [0.884748, 1.000000, 0.793269], [0.827503, 0.793269, 1.000000]]

    adjusted_weights = [2*0.196889,2*0.18622,2*0.185195]
    my_basket_object = BS_Basket(adjusted_weights,[asset_AAPL,asset_AMZN,asset_TSLA],correl_matrix)

    call_option_object = Option(
                    strike = 200,
                    terminal_payoff_function=lambda x: max(x-200,0),
                    actualisation_rate= 0.05,
                    maturity= 1,
                    payoff_name= "call"
                    )

    print(f"Inital spot of AAPL: {round(asset_AAPL.initial_spot,2)} \n")

    print(f"CF European price is: {call_option_object.Price(asset_AAPL)} \n")

    pricing_config = PricingConfiguration(nb_simulations=1000000,pricing_model_name="MC")
    call_option_object.set_pricing_configuration(pricing_config)
    print(f"MC European price is: {call_option_object.Price(asset_AAPL)} \n")

    pricing_config = PricingConfiguration(nb_simulations=1000000,pricing_model_name="MCLS")
    call_option_object.set_pricing_configuration(pricing_config)
    print(f"MCLS European price is: {call_option_object.Price(asset_AAPL)} \n")

    pricing_config = PricingConfiguration(nb_simulations=1000000,pricing_model_name="MC")
    call_option_object.set_pricing_configuration(pricing_config)
    call_option_object.set_predefined_pricing_reduction_techniques(asset_AAPL)
    call_option_object.set_pricing_antithetic()
    print(f"MC European price is: {call_option_object.Price(asset_AAPL,display_info=True)} \n")