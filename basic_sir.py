import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import odeint 
plt.style.use('ggplot')


def heaviside(center, val): 
    y = 1. if val < center else 0. 
    return y 

# POPULATION MODEL 
def dpop(y, t, config):

    # CONSCIOUS, PASSIVE, UNCONSCIOUS
    c,p,u,e = y  



    # UPDATE CONSCIOUS 
    # SUBSTRACT FAITH LOSS 
    # ADD CONVERSION DYNAMICS WITH UNCONSCIOUS 
    # ADD CONVERSION DYNAMICS WITH PASSIVE
    # ADD PASSIVE CONVERSION FROM HIGH PRICES 
    # SUBSTRACT CONVERSION TO PASSIVE FROM LOW PRICES 
    dcdt =  -config['c2u_faith'] * c \
            + (config['u2c_talk'] - config['c2u_talk']) * c* u \
            + c * p *(config['p2c_talk'] - config['c2p_talk']) \
            + heaviside(e, config['p_price_conversion']) * config['price_conversion_ratio'] * p \
            - heaviside(config['c_price_conversion_low'], e) * config['price_conversion_ratio'] * c


    # UPDATE PASSIVE  
    # ADD CONVERSION DYNAMICS WITH CONSCIOUS 
    # ADD CONVERSION DYNAMICS WITH PASSIVE
    # SUBSTRACT PASSIVE CONVERSION FROM HIGH PRICES 
    # ADD UNCONSCIOUS CONVERSION FROM HIGH PRICES 
    # ADD CONSCIOUS CONVERSION FROM LOW PRICES 
    # SUBSTRACT CONVERSION TO UNCONSCIOUS FROM LOW PRICES
    dpdt =  - c * p *(config['p2c_talk'] - config['c2p_talk']) \
            + u * p *(config['u2p_talk'] - config['p2u_talk']) \
            - heaviside(e, config['p_price_conversion']) * config['price_conversion_ratio'] * p \
            + heaviside(e, config['u_price_conversion']) * config['price_conversion_ratio'] * u \
            + heaviside(config['c_price_conversion_low'], e) * config['price_conversion_ratio'] * c \
            - heaviside(config['p_price_conversion_low'], e) * config['price_conversion_ratio'] * p
    
    # UPDATE UNCONSCIOUS 
    # ADD FAITH LOSS FROM CONSCIOUS 
    # ADD CONVERSION DYNAMICS WITH CONSCIOUS
    # ADD CONVERSION DYNAMICS WITH PASSIVE 
    # SUBSTRACT PASSIVE CONVERSION FROM HIGH PRICES 
    # SUBSTRACT CONVERSION FROM PASSIVE FROM LOW PRICES 
    dudt =  config['c2u_faith'] * c  \
            + (config['c2u_talk'] - config['u2c_talk']) * c* u \
            + (config['p2u_talk'] - config['u2p_talk']) * p* u \
            - heaviside(e, config['u_price_conversion']) * config['price_conversion_ratio'] * u \
            + heaviside(config['p_price_conversion_low'], e) * config['price_conversion_ratio'] * p


    # ENERGY PRICE UPDATE 
    dedt = 0.05 +  config['energy_oscillation_magn'] * np.cos(t * config['energy_price_period'])


    return [dcdt, dpdt, dudt, dedt]


def simulate_pop_dynamics(config, timesteps): 

    pop_counscious = 100 
    pop_passive = 100 
    pop_unconscious = 100 
    energy_price = 20.

    t = np.arange(timesteps)
    result = odeint(dpop, [pop_counscious, pop_passive, pop_unconscious, energy_price], t, 
        args = (config, ))

    return result
    


def plot_results_pop(result): 

    f, axes = plt.subplots(result.shape[1], 1)
    titles = ['CONSCIOUS', 'PASSIVE', 'UNCONSCIOUS', 'PRICE']
    for i, (title, ax) in enumerate(zip(titles, axes)): 
        ax.plot(result[:,i])
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":


    config = {'c2u_faith': 0.00003, 
              'c2u_talk' : 0.00001, 
              'u2c_talk' : 0.00005, 
              'p2c_talk' : 0.00006, 
              'c2p_talk' : 0.00004, 
              'u2p_talk' : 0.00006, 
              'p2u_talk' : 0.00004, 
              'energy_oscillation_magn': 0.5, 
              'energy_price_period': 0.06, 
              'u_price_conversion': 45.,  
              'p_price_conversion': 50, 
              'price_conversion_ratio': 0.05, 
              'p_price_conversion_low' : 30, 
              'c_price_conversion_low': 20, 
             }
    
    plot_results_pop(simulate_pop_dynamics(config, 363))
