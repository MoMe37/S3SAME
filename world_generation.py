import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd  
from basic_sir import simulate_pop_dynamics 
plt.style.use('ggplot')


# DEFAULT BASE CONSO 

x = np.linspace(0., np.pi, 24)
min_conso = 20
base_conso = pd.read_csv('rl_s3same_env.csv')['conso'].values.reshape(-1,1)[:363*24,:]
ref_load = pd.read_csv('ref_load.csv')['load'].values.reshape(-1,1)[:363*24,:]


# SURCONSO FACTOR FOR POP
base_people = 1.
sober_light = 0.8
sober_heavy = 0.65




def make_conso(daily_conso, pop_base,pop_light, pop_heavy): 

    # conso_militia = (base_conso.reshape(1,-1) - min_conso) * militia_conso * pop_m.reshape(-1,1)
    # conso_worker = (base_conso.reshape(1,-1) - min_conso) * pop_w.reshape(-1,1)

    consos = []
    for pop, ratio in zip([pop_base, pop_light, pop_heavy], [base_people, sober_light, sober_heavy]):
        consos.append((daily_conso.reshape(-1,1) - min_conso) * pop * ratio)
        
    total_conso = np.sum(np.hstack(consos), axis = 1)
    

    return total_conso

    # # 2D plots 
    # f, axes = plt.subplots(2,1)
    # axes = axes.flatten()

    # for i in range(total_conso.shape[0]): 
    #     axes[0].plot(total_conso[i], label = 'Day {}'.format(i))
    # axes[0].hlines(min_conso, 0, total_conso.shape[1], linestyle = 'dashed', label = 'Min conso')
    # axes[0].legend()
    # axes[0].set_ylim(0, np.max(total_conso) * 1.1)
    # axes[0].set_title('Conso', weight = 'bold')

    # axes[1].plot(pop_w, label = 'worker')
    # axes[1].plot(pop_m, label = 'militia')
    # axes[1].plot(pop_m + pop_w, label = 'Total')
    # axes[1].legend()
    # axes[1].set_title('Populations', weight = 'bold')

    # plt.show()





if __name__ == "__main__": 


    results = []
    for alpha in np.linspace(0.2,0.5, 5): 
        for beta in np.linspace(0.5,0.7,5): 
                
            config = {'alpha': alpha, 'beta': beta}
            pop_fractions = simulate_pop_dynamics(config, 363)
            full_conso = None
            for i in range(pop_fractions.shape[0]): 
                daily_conso = base_conso[i*24: (i+1)*24,: ]

                # ===================
                # pop_fractions = np.random.uniform(0,100, size =(3,))
                # pop_fractions = np.exp(pop_fractions) / np.sum(np.exp(pop_fractions))
                # ===================

                conso = make_conso(daily_conso, *pop_fractions[i])
                if full_conso is not None: 
                    full_conso = np.vstack([full_conso, conso.reshape(-1,1)])
                else: 
                    full_conso = conso.reshape(-1,1)

                # TEST CONFIG 

            diff = ref_load - full_conso
            
            results.append([alpha, beta, diff.mean() > 9.5])


    results = np.array(results)
    colors = np.array([[1.,0,0] if r > 0 else [0,1.,0] for r in results[:,2]]) 
    plt.scatter(results[:,0], results[:,1], c = colors)
    plt.show()

    