import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd  
from basic_sir import simulate_pop_dynamics 
plt.style.use('ggplot')


# TRAINED RL AGENT OUTPUT 
ref_load = pd.read_csv('ref_load.csv')['load'].values.reshape(-1,1)[:363*24,:]


# SURCONSO FACTOR FOR POP
base_people = 1.
sober_light = 0.8
sober_heavy = 0.65




def make_conso(daily_conso, pop_base,pop_light, pop_heavy): 

    consos = []
    for pop, ratio in zip([pop_base, pop_light, pop_heavy], [base_people, sober_light, sober_heavy]):
        consos.append((daily_conso.reshape(-1,1) - min_conso) * pop * ratio)
        
    total_conso = np.sum(np.hstack(consos), axis = 1)
    

    return total_conso



if __name__ == "__main__": 


    # BASE CONFIG FOR THE CURRENT MODEL. MIGHT HAVE TO BE MODIFIED 

    base_config = {'c2u_faith': 0.00003, 
              'c2u_talk' : 0.00001, 
              'u2c_talk' : 0.00005, 
              'p2c_talk' : 0.00006, 
              'c2p_talk' : 0.00004, 
              'energy_oscillation_magn': 0.5, 
              'energy_price_period': 0.04, 
              'u_price_conversion': 45.,  
              'p_price_conversion': 50, 
              'price_conversion_ratio': 0.1, 
              'p_price_conversion_low' : 30, 
              'c_price_conversion_low': 20, 
             }

    results = []


    # PARAMETER RANGE EXPLORATION. 
    # THIS IS AN EXAMPLE AND OTHER VARIABLES COULD BE CONSIDERED
    for price_conversion_ratio in np.linspace(0.1,0.3,5): 
        for c2u_faith in np.linspace(0.00002,0.00004,5): 
                
            # COPY BASE CONFIG 
            config = base_config.copy()

            # UPDATE PARAMETERS
            config['price_conversion_ratio'] = price_conversion_ratio
            config['c2u_faith'] = c2u_faith

            # GENERATE POPULATIONS BASED ON THE CONFIG HYPOTHESIS
            pop_energy = simulate_pop_dynamics(config, 363)
            pops = pop_energy[:,:-1] # REMOVE ENERGY PRICES 

            # NORMALIZE
            pops_normalized = np.exp(pops) / np.sum(np.exp(pops), 1, keepdims = True)
            
            # SIMULATE FOR A YEAR 
            full_conso = None
            for i in range(pops_normalized.shape[0]): 
                daily_conso = base_conso[i*24: (i+1)*24,: ]

                # SCALE CONSUMPTION BASED ON CURRENT POPULATION DISTRIBUTION 
                conso = make_conso(daily_conso, *pops_normalized[i])
                if full_conso is not None: 
                    full_conso = np.vstack([full_conso, conso.reshape(-1,1)])
                else: 
                    full_conso = conso.reshape(-1,1)



            # TEST CONFIG 

            
            # FAILURE HAPPENS WHEN ENERGY REQUIRED > 110% LOAD AVAILABLE 
            # LOAD IS GIVEN BY TRAINED RL AGENT
            failures = np.where(full_conso / ref_load > 1.1, 1., 0.)

            # THE SCENARIO CAN BE CONSIDERED INADEQUATE WHEN THERE ARE OVER 200 HOURS OF FAILURE OVER A YEAR 
            results.append([price_conversion_ratio, c2u_faith, np.sum(failures) > 200])


    # DISPLAY EXAMPLE 
    
    results = np.array(results)
    colors = np.array([[1.,0,0, 1.] if r == 0 else [0,1.,0, 0.] for r in results[:,2]]) 
    plt.scatter(results[:,0], results[:,1], c = colors)
    plt.show()

    