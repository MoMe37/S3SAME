import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import odeint 
plt.style.use('ggplot')

def d_sir0(y, t, n, beta, gamma): 

    S,I,R = y
    dsdt = - beta * S * I/n
    didt = beta * S * I/n - gamma * I
    drdt = gamma * I

    return dsdt, didt, drdt

def run_sir(**kwargs): 

    pop_size = 1000
    infected = 1
    recovered = 0 
    susceptible = pop_size - (infected + recovered)
    beta = kwargs['beta']
    gamma = kwargs['gamma']
    # beta = 0.2 
    # gamma = 1./10.
    t = np.linspace(0,160,160)

    y0 = susceptible, infected, recovered
    result = odeint(d_sir0, y0, t, args = (pop_size, beta, gamma))

    S,I,R = result.T 

    return S,I,R, t
    # plt.plot(t, S/pop_size, label = 'Susceptible')
    # plt.plot(t, I/pop_size, label = 'Infected')
    # plt.plot(t, R/pop_size, label = 'Recovered')

    # plt.xlabel('Time (days)')
    # plt.ylabel('Population size')
    # plt.ylim(0.,1.)
    # plt.legend()

    # plt.title('Basic SIR', weight = 'bold')
    # plt.show()

def display_alpha_influence(): 

    Ss, Is, Rs = [], [], []
    betas = np.linspace(0.1,0.8, 8)
    for beta in betas: 
        s,i,r, t = run_sir(beta = beta, gamma = 0.1)
        Ss.append(s)
        Is.append(i)
        Rs.append(r)

    for i, beta in zip(Is, betas): 
        plt.plot(t, i/1000, label = 'Alpha = {:.2f}'.format(beta))
    plt.legend()
    plt.title('Infection percentage wrt alpha',weight = 'bold')
    plt.xlabel('Time (days)')
    plt.ylabel('Pop percentage')
    plt.ylim(0.,1.)
    plt.show()

def d_basic(y, t, alpha): 

    dydt = -alpha * y
    return dydt


def basic_test(): 

    n_pop = 1000
    t = np.arange(200)
    alphas = np.linspace(0.01,0.1,8)
    for alpha in alphas: 
        print(alpha)
        result = odeint(d_basic, n_pop, t, args = (alpha,))

        pop_evol = result.T
        plt.plot(t,pop_evol.flatten(), label ='{:.2f}'.format(alpha))
    plt.legend()
    plt.show()

def heaviside(center, val): 
    y = 1. if val < center else 0. 
    return y 

def dzombies(y,t,n,z1,z2,z3,m1,f4,f5,m3,h3,r1,r2,r3,alpha,beta): 

    sw,sm,sh,sz,r = y 

    dfl = r  / (sw + r2 * sm + r3 * sh)

    dzdt = z1 * (sw * sz / n) + z2 * (sh * sz / n) + (z3 - m1) * (sm * sz / n)
    dwdt = -(z1 + alpha * (1-z1) + beta * (1. - z1 - alpha * (1-z1))) * sw * sz / n + \
        0.5 * heaviside(4, dfl) * sm + 0.5 * heaviside(2, dfl) * sh + f4 * sw - f5 * sw * sm /n

    dmdt = -z3 * sz * sm / n + beta * (1 - z1 - alpha * (1. - z1)) * sz * sw / n - m3 * sm * sm / n - \
    0.5 * heaviside(4, dfl) - alpha * (1 - z3) * sz * sm / n + beta * (1 - z2) * sh * sz / n

    dhdt = - z2 * sz * sh / n + alpha * (1 - z1) * sw * sz / n - h3 * sh * sm / n - 0.5 * heaviside(2, dfl) * sh + \
    alpha * (1- z3) * sz * sm / n - beta* (1 - z2) * sh * sz/ n

    drdt = (r1 - 1) * sw - r2* sm - r3 * sh 

    return dwdt, dmdt, dhdt, dzdt, drdt 




def zombies_0(beta, m1): 

    pop_worker = 3000
    pop_mole = 0
    pop_militia = 0
    pop_zombies = 1
    n = 30
    R = 1000
    z1 = 0.1
    z2 = 0.05 
    z3 = 0.15
    # m1 = 0.2
    f4 = 6e-6
    f5 = 0.01 
    m3 = 0.01 
    h3 = 0.01 
    r1 = 3 
    r2 = 2 
    r3 = 0.4
    alpha = 0.#0.8 
    # beta = 0.1 


    t = np.arange(200)
    result = odeint(dzombies, [pop_worker, pop_militia, pop_mole, pop_zombies, R], t,
                   args = (n,z1,z2,z3,m1,f4,f5,m3,h3,r1,r2,r3,alpha,beta))


    pop_evol = result.T 
    pop_names = ['workers', 'militia', 'moles', 'zombies', 'supply']
    return pop_evol
    # for i in range(len(pop_names)-1):
    #     plt.plot(t, pop_evol[i,:], label = pop_names[i])
    # plt.legend()
    # plt.show()


if __name__ == "__main__":

    survived = []
    for i,m1 in enumerate(np.linspace(0.01,0.99,20)):
        for j,beta in enumerate(np.linspace(0.01,0.99,20)):
            pops = zombies_0(beta, m1)
            if np.sum(pops[:3, -1]) > 1:
                survived.append([beta, m1]) 
    
    plt.scatter(np.array(survived)[:,0], np.array(survived)[:,1])
    plt.xlim(0.,1.)
    plt.ylim(0.,1.)

    plt.show()