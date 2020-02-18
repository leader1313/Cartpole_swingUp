# coding: utf-8
import gym, keyboard, pickle, sys, time
import numpy as np

from gym_cartpole_swingup.envs import CartPoleSwingUpEnv

env = CartPoleSwingUpEnv()

max_episode = 30
episode_num = 1
np.random.seed(int(time.time()))
while 1 :
    results =  {'x_pos': [],
            'x_dot': [],
            'cos': [],
            'sin': [],
            'theta_dot' : [],
            'action': [],
            'reward': [],
            'state' : [],
            }
    done = 0
    total_reward = 0
    total_timesteps = 0
    obs = env.reset()

    while not done:
        total_timesteps += 1
        a = 0
        if keyboard.is_pressed('a'):
            a = -1 
        elif keyboard.is_pressed('d'):
            a = 1
        
        action = np.random.randn(1)*1+a
        # print("[%i]timesteps  Action %0.2f" % (total_timesteps, action))
        results['x_pos'].append(obs[0])
        results['x_dot'].append(obs[1])
        results['cos'].append(obs[2])
        results['sin'].append(obs[3])
        results['theta_dot'].append(obs[4])
        results['state'].append(obs)
        results['action'].append(a)
        
        obs, rew, done, info = env.step(a)
        
        if rew > 0.99 :
            total_reward += rew
        if total_reward > 100:
            with open('data/sup_demo'+str(episode_num)+'.pickle', 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            episode_num += 1
            break
        elif total_timesteps > 500 :
            break
        env.render()
        time.sleep(0.1)
        print("%i episode now"%(episode_num))
        print("[%i]timesteps  reward %0.2f" % (total_timesteps, total_reward))
    if episode_num > max_episode :
        break