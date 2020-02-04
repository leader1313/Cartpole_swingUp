# coding: utf-8
from tools.GMLRM import GMLRM
import gym, keyboard, pickle, sys, time, torch
import matplotlib.pyplot as plt
import numpy as np
from gym_cartpole_swingup.envs import CartPoleSwingUpEnv

env = CartPoleSwingUpEnv()
initialize = {}
max_episode = 20
test_time = 100
graph =  {
            'episode_num': [],
            'success': [],
            }
for i in range(max_episode) :
    print("++"*30)
    print("Episode " + str(i+1) + " learner test start")
    print("++"*30)
    graph['episode_num'].append(i+1)
#load GMM_model    
    # with open('model/learner'+ str(i+1) + '.pickle', 'rb') as f:
    #     learner = pickle.load(f)
    # model = learner['model']
    # Weight = learner['Weight']
    # var = learner['var']
#load GP_model
    PATH = 'model/learner_'+str(i+1)
    model = torch.load(PATH)
    success = 0
    for t in range(test_time):
        done = 0
        total_reward = 0
        total_timesteps = 0
        obs = env.reset()
        while not done:
            total_timesteps += 1
        #GMM_model
            # action = 1*model.predict(obs,Weight)
        #GP_model    
            obser = obs[None,...]
            te_obser = torch.from_numpy(obser).float()
            learner_action, _ = model.predict(te_obser)
            
            action = learner_action

            # print('\t [Action] : ',action)
            obs, rew, done, info = env.step(action)
            if rew > 0.99 :
                total_reward += rew
            if total_reward > 100 :
                success += 1
                break
            elif total_timesteps > 500 :
                break
            env.render()
            # time.sleep(0.1)
            # print("[%i]timesteps  reward %0.2f" % (total_timesteps, total_reward))
        print('No. %i test Finished success : %i' %(t+1, success))
    graph['success'].append(success)
plt.figure()
X = graph['episode_num']
Y = graph['success']
plt.xlabel("episode_num")
plt.ylabel("success_rate")
plt.plot(X, Y, "*")
plt.plot(X,Y)
plt.show()