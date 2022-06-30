import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
plt.style.use('ggplot')


# DEFAULT BASE CONSO 

x = np.linspace(0., np.pi, 24)
min_conso = 15
base_conso = np.sin(x * 3) + 1 + min_conso

# SURCONSO FACTOR FOR POP
militia_conso = 1.8 




def make_conso(pop_w,pop_m): 

    conso_militia = (base_conso.reshape(1,-1) - min_conso) * militia_conso * pop_m.reshape(-1,1)
    conso_worker = (base_conso.reshape(1,-1) - min_conso) * pop_w.reshape(-1,1)

    total_conso =conso_militia + conso_worker 

    # 2D plots 
    f, axes = plt.subplots(2,1)
    axes = axes.flatten()

    for i in range(total_conso.shape[0]): 
        axes[0].plot(total_conso[i], label = 'Day {}'.format(i))
    axes[0].legend()

    axes[1].plot(pop_w, label = 'worker')
    axes[1].plot(pop_m, label = 'militia')
    axes[1].plot(pop_m + pop_w, label = 'Total')
    axes[1].legend()

    # 3D plots 

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    # for i in range(conso_worker.shape[0]):
    #     z= np.ones_like(conso_worker[0]) * i
    #     ax.plot(np.arange(conso_worker.shape[1]), total_conso[i,:], z)

    # ax.set_xlabel('Hours')
    # ax.set_ylabel('Conso')
    # ax.set_zlabel('Day')
    plt.show()





if __name__ == "__main__": 


    daily_pop_workers = np.arange(3,10)[::-1] 
    daily_pop_militia = np.max(daily_pop_workers) - daily_pop_workers

    make_conso(daily_pop_workers, daily_pop_militia)