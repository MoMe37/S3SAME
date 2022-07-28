import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import odeint 
plt.style.use('ggplot')

def test(y, t): 

    dydt = 0.05 + 0.2* np.cos(t * 0.1)
    print(t)
    # dydt = 0.05 * 2. * y**2 - 0.2 * y * np.cos(y * 0.1)
    return dydt


if __name__ == '__main__': 


    result = odeint(test, 5., np.linspace(0.,200.,1000)).T
    print(result)
    plt.plot(result.flatten())
    plt.show()