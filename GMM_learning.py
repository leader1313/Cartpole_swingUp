#-*- coding:utf-8 -*-
import pickle
from tools.GMLRM import GMLRM
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, det, pinv
from mpl_toolkits.mplot3d import Axes3D

#=========================sample data 정의================================
learning = {'state': [0,0,0,0],
                'action': [],
                'dataset': []}
for n in range(10):
    with open('data/sup_demo'+str(n+1)+'.pickle', 'rb') as handle:
        results = pickle.load(handle)

    results['state'] = np.array(results['state'])
    results['action'] = np.array(results['action'])[...,None]
        #learning episode each
    # learning['state'] = np.array(results['state'])
    # learning['action'] = np.array(results['action'])
        #learning episode merge
    if n == 0 :
        learning['state'] = np.array(results['state'])
        learning['action'] = np.array(results['action'])
    else :
        learning['state'] = np.append(learning['state'],results['state'], axis= 0)
        learning['action'] = np.append(learning['action'],results['action'])[...,None]

    N = learning['state'].shape[0]
    X = learning['state']
    Y = learning['action']


#==================== random initialize parameter =======================

    K = 1                                       # solution 수
    M = 10                                    # Number of model
    GM = GMLRM(X,Y,K,M)
    Weight ,var = GM.EM()
    learner = {'model': GM, 'Weight': Weight, 'var' : var}
    print("="*40)
    with open('model/learner'+str(n+1)+'.pickle', 'wb') as handle:
        pickle.dump(learner, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(" \t model %i saved" %(n+1))
    print(" \t Number of step %i " %(N))
    print("="*40)
    
# predict1 = np.zeros(test_num)[:,None]
# predict2 = np.zeros(test_num)[:,None]
# test_x = np.linspace(0, np.pi *2, test_num)[:, None]
# Pi = np.zeros((test_num,M))
# for m in range(M):
#     for n in range(test_num):
#         Pi[n,m] = GM.Gaussian_bias(test_x[n],m*GM.phi_mean,GM.phi_sigma)
    #  

# for n in range(test_num):
#     predict = np.zeros(K)
#     predict = GM.predict(test_x[n],Weight)
#     predict1[n] = predict[0]
#     predict2[n] = predict[1]
# ss = np.sqrt(var)


    

    



