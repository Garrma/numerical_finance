# <center> Pricing of European and Bermudan </center> 
## <center> *applications to variance reduction methods*  </center>

This repository is project made within the context of Msc 203 Numerical Finance course by CASTELLARNAU Romain, GARRIGA Maxime and SAULIS Lukas.

This project constitutes a first approach to multiple stocks options pricing with a focus on european & bermuda basket calls. The work also includes a few Monte Carlo reduction variance methods with antithetic and control variate.

The choice between C++ and python for the project was free. To stay within the framework class and challenge ourselves, we developped from sratch random number generators without using inbuilt python packages. Have a look at the folder 
[here](https://github.com/Garrma/numerical_finance/blob/main/docs/Generators.pdf). This is not mandatory for option pricing but added much more challenge and effors to our work.

## Instructions

The instructions for the project are available here ➡️
[See instructions](https://github.com/Garrma/numerical_finance/blob/main/docs/Project%20Instruction.pdf)


## Results 

The main element of such project is to get an introduction to variance reduction implementation.<br>
Some results for the pricing of a european call are presented below. Full results are available here ➡️
[See results](https://github.com/Garrma/numerical_finance/blob/main/docs/Numerical_Finance_Project.pdf)

<!-- image result -->
![Hello World](https://res.cloudinary.com/dq4xpsevx/image/upload/v1715023923/Github/Numerical%20Finance/convergence_quasi_techniques.png)


## Extension

This project offers a quick python pricing tool which can be used for other purposes. Here is a small example of what you can do but feel free to dive into the classes to see everything what you can do.

**instantiate your underlying and options as below**
```python
# Iniate single asset or basket
risk_free_rate = 0.05
asset_AAPL = BS_Asset(169.300003,0.20,risk_free_rate)
asset_AMZN = BS_Asset(179,0.32,risk_free_rate)
asset_TSLA = BS_Asset(179.99,0.53,risk_free_rate)
correl_matrix = [[1.000000, 0.884748, 0.827503], [0.884748, 1.000000, 0.793269], [0.827503, 0.793269, 1.000000]]
adjusted_weights = [2*0.196889,2*0.18622,2*0.185195]                    # target intial value of 200

# play with aggregate_function to build any type of payoff -> here best of basket
my_basket_object = Basket(aggregate_function = np.sum, weights = adjusted_weights,assets=BS_AssetND([asset_AAPL,asset_AMZN,asset_TSLA],correl_matrix = correl_matrix))

# Creating ATM option
call_option_object = Option(
            strike = 200,
            terminal_payoff_function= lambda x : max(x-200,0),
            actualisation_rate= 0.05,
            maturity= 1,
            #payoff_name= "call"     # allows not to give payoff for some known payoffs
            )
```

**price any european options as below**
```python
######### EXEMPLE OF PRICING USING MC ##############
# 1/ build pricing config
pricing_config = PricingConfiguration(nb_simulations=100000,pricing_model_name="MC")       
# 2/ set pricing config to option object
call_option_object.set_pricing_configuration(pricing_config)                                
# 3/ price with according underlying
print(f"MC European price is: {call_option_object.Price(my_basket_object,display_info=True)} \n")   
```

**price non european options as below**
```python
######## EXEMPLE OF PRICING USING MCLS #############
# suitable for american options -> 
bermudan_call_option_object = Option(
            strike = 200,
            terminal_payoff_function = lambda x : max(x-200,0),
            actualisation_rate= 0.05,
            exercise_times= [0,0.25,0.5,0.75,1],
            )

pricing_config = PricingConfiguration(nb_simulations=10000,pricing_model_name="MCLS")     
bermudan_call_option_object.set_pricing_configuration(pricing_config)
print(f"MCLS Bermudan price is: {bermudan_call_option_object.Price(my_basket_object,display_info=True)} \n")
```

## Contributors

<a href="https://github.com/Garrma/numerical_finance/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Garrma/numerical_finance" />
</a>
