#-*- coding:utf-8 -*-
import numpy as np
from numpy.linalg import inv, det, pinv


class GMLRM:
    def __init__(self, X, Y, K, M):
        self.X , self.Range, self.Max, self.Min = self.normalization(X)
        self.Y = Y
        self.D = X.shape[1]
        self.N = len(X)
        self.K = K
        self.M = M
        self.prior = np.random.rand(self.K)                   # 어느 클래스에 속할 확률 vector
        self.Weight = np.random.rand(self.M,self.K)           # mean vector
        self.var = np.zeros(1)+2                              # variance

        self.Phi = np.zeros((self.N,self.M))
        self.phi_mean = np.zeros(self.D)+ 2
        self.phi_sigma = np.diag((np.zeros(self.D)+1))

    def normalization(self,x):
        D = x.shape[1]
        N = x.shape[0]
        X = np.zeros((N,D))
        Max = np.zeros(D)
        Min = np.zeros(D)
        Range = np.zeros(D)
        for d in range(D):
            Max[d] = max(x[:,d])
            Min[d] = min(x[:,d])
            Range[d] = Max[d] - Min[d]
            for n in range(N):
                X[n,d] = (x[n,d]- Min[d])/Range[d]
        return X, Range, Max, Min


    def dot(self,x,y,z):
        a = np.dot(x,y)
        b = np.dot(a,z)
        return b

    def uni_Gaussian_Distribution(self,x, mean, sigma):
        E = np.exp((-1/(2*((sigma)**2)))*((x-mean)**2))
        y = (1/(((2*np.pi)*((sigma)**2))**(1/2))) * E
        return y

    def Gaussian_Distribution(self,x, D, mean, sigma):
        E = np.exp((-1/2)*self.dot((x-mean).T,(inv(sigma)),(x-mean)))
        y = (1/((2*np.pi)**(D/2)))*(1/((det(sigma))**(1/2)))*E        
        return y

    def uni_Gaussian_bias(self,x, mean, sigma):
        B = np.exp((-1/(2*((sigma)**2)))*((x-mean)**2))
        return B

    def Gaussian_bias(self,x, mean, sigma):
        B = np.exp((-1/2)*self.dot((x-mean).T,(inv(sigma)),(x-mean)))
        return B
    
    def cal_phi(self, X, m):
        phi = self.Gaussian_bias(X,m*self.phi_mean,self.phi_sigma)
        return phi

    def Init_phi(self):
        for m in range(self.M):
            for n in range(self.N):
                self.Phi[n,m] = self.cal_phi(self.X[n], m/self.M)
        return self.Phi

    def responsibility(self, n, k, Weight, Phi, var, prior):
        K = self.K
        t = self.Y[n]
        wk_bn = np.dot(self.Weight[:,k].T,self.Phi[n])
        p = prior
        sum_r = np.zeros(1)[...,None]
        r_k = p[k]*self.uni_Gaussian_Distribution(t, wk_bn, var)
        for j in range(K):
            r1 = p[j]*self.uni_Gaussian_Distribution(t, np.dot(Weight[:,j].T, Phi[n]), var)
            sum_r += r1
        return r_k/sum_r

    def expectation(self, prior, Weight, Phi, var):
        N = self.N
        K = self.K
        r = np.zeros((N,K))             # responsibility
        R = np.zeros((K,N,N))
        for n in range(N):
            for k in range(K):
                r[n,k] = self.responsibility( n, k, Weight, Phi, var, prior)
                R[k,n,n] = r[n,k]
        return r, R

    def maximization(self, r, R, Phi):
        N = self.N
        K = self.K
        Y = self.Y
        prior = self.prior
        Weight = self.Weight
        var = self.var
        sum_r = np.zeros(K)
        sum_rd = np.zeros(1)
        for k in range(K):
            for n in range(N):
                re = r[n,k]
                sum_r[k] += re

            prior[k] = sum_r[k]/N

            invPRP = pinv(self.dot(Phi.T,R[k],Phi))
            PtRy = self.dot(Phi.T,R[k],Y)
            Weight[:,k][:,None] = np.dot(invPRP, PtRy)
        
            for n in range(N):
                re = r[n,k]
                d = (Y[n]-np.dot(Weight[:,k].T, Phi[n]))**2

                sum_rd += re * d

        var = sum_rd/N

        return prior, Weight, var

    def EM(self):
        prior = self.prior
        Weight = self.Weight
        var = self.var
        Phi = self.Init_phi()
        T = 1000
        for t in range(T):
            r, R = self.expectation(prior,Weight,Phi,var)
            prior, Weight, var = self.maximization(r, R, Phi)
            print("=",end='')
        print("\n")
        return Weight, var

    def predict(self, new_X, Weight):
        new_phi = np.zeros(self.M)
        predict = np.zeros(self.K)
        X       = np.zeros(self.D)
        for d in range(self.D):
            X[d] = (new_X[d]- self.Min[d]) /self.Range[d]
        
        for m in range(self.M):
            new_phi[m] = self.cal_phi(X,m/self.M)
        for k in range(self.K):
            predict[k] = np.dot(Weight[:,k].T,new_phi)
        return predict
    
