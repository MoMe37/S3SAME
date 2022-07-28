import numpy as np 
import gym 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from stable_baselines3 import PPO  
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from tqdm import tqdm 
mpl.use('TkAgg')

plt.style.use('ggplot')

class S3Env(gym.Env): 

    def __init__(self): 
        super().__init__()
        self.data = pd.read_csv('./rl_s3same_env.csv')
        target_cols = [c for c in self.data.columns if c.endswith('n')]
        self.data_n = self.data[target_cols]
        self.observation_space = gym.spaces.Box(low= -100., high = 100., shape = (len(target_cols) + 1,))
        self.action_space = gym.spaces.Box(low = -1., high = 1., shape = (1,))


        self.max_ts = 240

        self.max_storage = 50. 
        self.max_hourly_storage_ratio = 0.25 
        self.max_hourly_storage = self.max_storage * self.max_hourly_storage_ratio

        # CAREFUL !! ARBITRARY VALUE !!
        # ============================
        self.conso_factor = 0.1 
        # ============================


    def reset(self, idx = None):

        if idx is not None: 
            self.current_idx = idx 
        else: 
            self.current_idx = np.random.randint(0, self.data_n.shape[0] - (self.max_ts + 2))
        self.ts = 0
        self.current_load = 0.5 * self.max_storage

        return self.get_obs() 

    def step(self, action): 


        action = action[0]

        next_prod = self.data['prod'][self.current_idx + self.ts + 1]
        next_conso = self.data['conso'][self.current_idx + self.ts + 1] * self.conso_factor

        to_grid = 0. 

        if action > 0: #LOAD
            withdrawn_energy = 0. 
            
            to_storage = action * next_prod
            to_grid = (1. - action) * next_prod

            self.current_load = np.min([self.current_load + to_storage, self.max_storage])

        else: 
            withdrawn_energy = np.min([np.abs(action) * self.max_hourly_storage * self.max_storage, self.current_load])
            to_grid = withdrawn_energy + next_prod 
            self.current_load = np.clip(self.current_load - withdrawn_energy, 0., self.max_storage)


        reward = self.compute_reward(next_conso, to_grid)
        

        self.ts += 1
        done = True if self.ts >= self.max_ts else False
        

        return self.get_obs(), reward, done, {"conso" : next_conso, "prod" : next_prod, "diff" :next_conso - to_grid,  "grid": to_grid, "load": self.current_load, "load_normalized": self.current_load / self.max_storage, "withdrawn_energy": to_grid - next_prod}

    def compute_reward(self, next_conso, to_grid): 

        # GOAL IS TO SEND AN ENERGY VOLUME AS CLOSE AS POSSIBLE TO THE NEXT DEMAND  
        d = np.abs(next_conso - to_grid)
        return np.exp(-d)


    def get_obs(self): 

        data = self.data_n.values[self.current_idx + self.ts]
        obs = np.hstack([data.flatten(), self.current_load / self.max_storage])

        return obs.flatten()

    def random_action(self): 
        return np.random.uniform(-1.,1., size = (self.action_space.shape[0]))


class S3R2(S3Env):

    def compute_reward(self, next_conso, to_grid): 
        d = to_grid - next_conso
        r = 1. if d > 0 else d
        return r 

class DummyModel: 
    def __init__(self, action_space): 
        self.size = action_space.shape[0]

    def predict(self, x):
        return np.random.uniform(-1.,1., size = (self.size, ))

def show_ep(env, model = None, idx = None): 

    f, axes = plt.subplots(3,3)
    axes = axes.flatten()

    


    for name, m in zip(['ppo', 'dummy'], [model, DummyModel(env.action_space)]):
        
        grid = []
        conso = []
        prod = []
        diff = []
        load = []
        rewards = []
        withdrawn_energy = []
        diff = []

        s = env.reset(idx = idx)
        idx = env.current_idx
        done = False 


        while not done: 
            a = m.predict(s)
            s, r, done, info = env.step(a)
            grid.append(info['grid'])
            conso.append(info['conso'])
            prod.append(info['prod'])
            diff.append(info['diff'])
            load.append(info['load'])
            withdrawn_energy.append(info['withdrawn_energy'])
            rewards.append(r)


        axes[0].plot(conso, label = name, alpha = 0.7)
        axes[0].legend()
        axes[0].set_title("conso", weight = 'bold')

        axes[1].plot(prod, label = name, alpha = 0.7)
        axes[1].legend()
        axes[1].set_title("prod", weight = 'bold')

        axes[2].plot(grid, label = name, alpha = 0.7)
        axes[2].legend()
        axes[2].set_title("to_grid", weight = 'bold')

        axes[3].plot(load, label = name, alpha = 0.7)
        axes[3].legend()
        axes[3].set_title("load", weight = 'bold')
        axes[3].set_ylim(0, env.max_storage)

        axes[4].plot(rewards, label = name, alpha = 0.7)
        axes[4].legend()
        axes[4].set_title("rewards", weight = 'bold')

        axes[5].plot(np.cumsum(rewards), label = name, alpha = 0.7)
        axes[5].legend()
        axes[5].set_title("Cum sum rewards", weight = 'bold')

        axes[6].plot(diff, label = name, alpha = 0.7)
        axes[6].legend()
        axes[6].set_title("Difference", weight = 'bold')

        axes[7].plot(withdrawn_energy, label = name, alpha = 0.7)
        axes[7].legend()
        axes[7].set_title("withdrawn_energy", weight = 'bold')


    plt.show()

    return 

def make_ref_sequence(model, env): 

    s = env.reset(0)
    
    ref_load = []
    pbar = tqdm(total = env.data.shape[0] -2)
    for i in range(env.data.shape[0] - 2): 
        a = model.predict(s)
        s, r, _ ,info = env.step(a)
        ref_load.append(float(info['load']))

        pbar.update(1)
    pbar.close()


    df = pd.DataFrame(np.array(ref_load).reshape(-1,1), columns = ['load'])
    df['date'] = env.data['date'][:-2]

    df.to_csv('./ref_load.csv', index =False)


def train_agents(): 

    for env in [S3Env, S3R2]: 
        venv = SubprocVecEnv([env for i in range(2)])
        model = PPO('MlpPolicy', VecMonitor(venv), verbose = 1, device = 'cpu')
        model.learn(total_timesteps = 1000000)
        model.save('ppo_s3_{}'.format(env.__name__))

def test_agents(): 


    for env in [S3R2, S3Env]: 
        agent = PPO.load('ppo_s3_{}'.format(env.__name__))
        show_eps(env(), agent)
        # show_ep(env(), agent)

def show_eps(env, agent, nb_eps = 100): 

    deltas = None
    for ep in range(nb_eps): 
        s = env.reset()
        done = False
        ep_deltas = []
        while not done: 
            action = agent.predict(s)
            s, r, done, info = env.step(action)
            ep_deltas.append(float(info['grid'] / info['conso'] - 1.))
        if deltas is not None: 
            deltas = np.hstack([deltas, np.array(ep_deltas).reshape(-1,1)])
        else: 
            deltas = np.array(ep_deltas).reshape(-1,1)
    
    f, axes = plt.subplots(2,1)
    axes = axes.flatten()
    results = np.sum(np.where(deltas < -0.2, 1.,0.), 0)
    axes[0].bar(np.arange(nb_eps), results/deltas.shape[0])
    axes[1].fill_between(np.arange(nb_eps), deltas.mean(0) - deltas.std(0), deltas.mean(0) + deltas.std(0), alpha = 0.3)
    for i in range(deltas.shape[1]):
        axes[1].scatter(np.arange(deltas.shape[0])*0 + i, deltas[:,i], color= ['k' if deltas[j,i] > 0 else 'r' for j in range(deltas.shape[0])], alpha = 0.5)
    axes[1].hlines(-0.2, 0, deltas.shape[1]-1, label ='Acceptance Threshold: 80%')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Energy Ratio: Delivered / Needed')
    axes[1].set_title('Hourly ratio of delivered energy vs needed')
    axes[1].legend()

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Normalized number of system failure occurences (ep = 240 steps)')
    axes[0].set_ylim(0.,1.)
    axes[0].set_title('Failed hours (relative to threshold)')
    axes[0].legend()
    plt.show()




if __name__ == "__main__": 

    # train_agents()
    test_agents()
    # model= PPO.load('./ppo_s3')
    # make_ref_sequence(model, S3Env())
    # show_ep(S3Env(), model)