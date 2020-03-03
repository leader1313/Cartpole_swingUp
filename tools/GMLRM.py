#!/usr/bin/env python
#-*- coding:utf-8 -*-
# import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, det, pinv
import sys


class GMLRM:
    def __init__(self, X, Y, K, M, T):
        #Constant
        self.X = X
        self.Y = Y
        self.D = X.shape[1]
        self.N = X.shape[0]
        self.K = K
        self.M = M
        self.Num_Model = (self.M)**(self.D)
        self.T = T
        self.Phi = np.zeros((self.N,self.Num_Model))
        self.sigma = np.array([1,1,1,1,1])
        self.phi_sigma = np.diag(self.sigma)
        self.X_Max = np.array([2,4.2,1,1,12])
        self.X_Min = np.array([-2,-4,-1,-1,-12.2])

        self.range = np.zeros(self.D)
        self.biasRange()
        self.Init_phi()
        
        #Variable
        self.prior = np.random.rand(self.K)                   # 어느 클래스에 속할 확률 vector
        self.Weight = np.random.rand(self.Num_Model,self.K)   # mean vector
        self.var = np.zeros(1)+1                              # variance
        self.r = np.zeros((self.N,K))                              # responsibility
        self.R = np.zeros((K,self.N,self.N))

    #=============== Support method =================
    def dot(self,x,y,z):
        a = np.dot(x,y)
        b = np.dot(a,z)
        return b

    def uni_Gaussian_Distribution(self,x, mean, var):
        E = np.exp(-1/(2*(var))*((x-mean)**2))
        y = (1/(((2*np.pi)*(var)**(1/2)))) * E
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
    #phi : bias
    def cal_phi(self, X, m):
        x = (np.zeros(self.D)+1)
        x = self.Base_10_to_n(x, m, 0)
        mean = self.X_Min + np.dot(self.range,np.diag(x))
        sigma = self.phi_sigma
        phi = self.Gaussian_bias(X, mean, sigma)
        return phi

    def Base_10_to_n(self, X ,n, i):
        if (int((n)/(self.M))):
            X[i] = (n)%(self.M)+1
            self.Base_10_to_n(X,int((n)/(self.M)), i+1)
        else : X[i] = (n)%(self.M)+1
        return X

    #=============== Update method =================
    def biasRange(self):
        self.range = (self.X_Max-self.X_Min)/(self.M + 1)
        
    def Init_phi(self):
        for m in range(self.Num_Model):
            for n in range(self.N):
                self.Phi[n,m]=self.cal_phi(self.X[n],m)
            
    def responsibility(self, n, k):
        t = self.Y[n]
        p = self.prior
        wk_bn = np.dot(self.Weight[:,k].T,self.Phi[n])
        
        sum_r = np.zeros(1)[...,None]
        r_k = p[k]*self.uni_Gaussian_Distribution(t, wk_bn, self.var)[...,None]
        for j in range(self.K):
            wj_bn = np.dot(self.Weight[:,j].T, self.Phi[n])
            r1 = p[j]*self.uni_Gaussian_Distribution(t, wj_bn, self.var)
            sum_r += r1

        return r_k/sum_r

    def expectation(self): 
        for n in range(self.N):
            for k in range(self.K):
                self.r[n,k] = self.responsibility(n, k)
                self.R[k,n,n] = self.r[n,k]

    def maximization(self):
        sum_r = np.zeros(self.K)
        sum_rd = np.zeros(1)
        for k in range(self.K):
            for n in range(self.N):
                re = self.r[n,k]
                sum_r[k] += re
            self.prior[k] = sum_r[k]/self.N            
            invPRP = pinv(self.dot(self.Phi.T,self.R[k],self.Phi))
            PtRy = self.dot(self.Phi.T,self.R[k],self.Y)
            
            self.Weight[:,k][:,None] = np.dot(invPRP, PtRy)
            for n in range(self.N):
                re = self.r[n,k]
                d = (self.Y[n]-np.dot(self.Weight[:,k].T, self.Phi[n]))**2

                sum_rd += re * d
        self.var = sum_rd/self.N

    def EM(self):
        for t in range(self.T):
            self.expectation()
            self.maximization()

    def predict(self, new_X):
        new_phi = np.zeros(self.Num_Model)
        predict = np.zeros(self.K)
        X = new_X
        for m in range(self.Num_Model):
            new_phi[m] = self.cal_phi(X,m)
        for k in range(self.K):
            predict[k] = np.dot(self.Weight[:,k].T,new_phi)
        return predict