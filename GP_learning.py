from tools.gpr import GPRegression
from tools.kernel import GaussianKernel
import sys, pickle, torch
import matplotlib.pyplot as plt
import numpy as np

def main():
    plt.style.use("ggplot")
    episode_num = 10
    graph = {'loss': [],
             'episode_num' : []}

    learning = {'state': [0,0,0,0],
                'action': [],
                'dataset': []}
    for i in range(episode_num) :
        
        #load pickle file
        with open('data/sup_demo'+str(i+1)+'.pickle', 'rb') as handle:
            results = pickle.load(handle)
        
        # ==============================================
        # Data slice and preprocess
        # ==============================================
        results['state'] = np.array(results['state'])
        results['action'] = np.array(results['action'])[...,None]
        
        #learning episode each
        # learning['state'] = np.array(results['state'])
        # learning['action'] = np.array(results['action'])
        
        #learning episode merge
        if i == 0 :
            learning['state'] = np.array(results['state'])
        else :
            learning['state'] = np.append(learning['state'],results['state'], axis= 0)

        learning['action'] = np.append(learning['action'],results['action'])[...,None]

        # None : (N, 1) 형태를 유지할 수 있게해줌
        
        learning['dataset'] = np.append(learning['state'],learning['action'],axis = 1)
        
        dataset = torch.from_numpy(learning['dataset']).float()
        
        # validate test and tain data
        train_size = int(1 * len(dataset))
        test_size = len(dataset) - train_size
        train, test = torch.utils.data.random_split(dataset, [train_size, test_size])
        # type change 
        train_X = train[:]
        train_X = train_X[:,:5]
        train_Y = train[:]
        train_Y = train_Y[:,5:6]


        kern = GaussianKernel()
        model = GPRegression(train_X, train_Y, kern)

        print("params", torch.exp(model.kern.param()[0]), torch.exp(model.sigma), model.negative_log_likelihood())
        model.learning()
        print("params", torch.exp(model.kern.param()[0]), torch.exp(model.sigma), model.negative_log_likelihood())
        PATH = 'model/learner_'+str(i+1)
        torch.save(model, PATH)
    

if __name__=="__main__":
    main()