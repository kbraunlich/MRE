#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 10:19:31 2018

@author: kurtb
"""


import pandas as pd
from scipy.spatial.distance import euclidean, pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import time
import glob
from scipy import stats
import clarte as cl
from scipy.optimize import minimize

#%%

def similarity_func(u, v):
    return 1/(1+euclidean(u,v))


def similarity_matrix(m):
    ''' m is n*d'''
    dists = pdist(m, similarity_func)
    return squareform(dists)

def calc_Qc(Rc,X):
    '''Rc is diagonal, X is shared reduced space proposal'''
    RcX = np.dot(X,Rc)
    Qc = similarity_matrix(RcX)#,sigmaSquared=None)    
    return Qc

def calc_Ec(Pc,Qc):
    '''mean row-wise[?] KL-divergence '''
    lkc = []
    
    # normalize
    for i in range(Qc.shape[0]):
        Qc[i,:] = [Qc[i,j]/np.sum(Qc[i,:]) for j in range(Qc.shape[1])]
        Pc[i,:] = [Pc[i,j]/np.sum(Pc[i,:]) for j in range(Pc.shape[1])]

    for pc,qc in zip(Pc,Qc): # 
#       lkc.append(stats.entropy(pc,qc)) 
       lkc.append(KL(pc,qc))  
    return np.mean(lkc)

    
def KL(P,Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    P = P.copy()+epsilon
    Q = Q.copy()+epsilon
    
    divergence = np.sum(P*np.log(P/Q))
    return divergence

def loss(vRcX,P,ndim):
    '''P: dictionary of similarity matrices'''
    mRcX = np.reshape(vRcX,(P['euc'].shape[0]+len(P),ndim))
    X = mRcX[len(P):,:]
    lEc = []
    for i,(n,Pc) in enumerate(P.items()):
        Rc = np.diag(mRcX[i,:])
        Qc = calc_Qc(Rc,X)
        lEc.append(calc_Ec(Pc,Qc))
    sumLec = np.sum(lEc)
    print(sumLec)
    return sumLec


#%% create "P"
    
timeMarker = time.strftime("%Y%m%d-%H%M%S")
plt.close('all')
plt.figure(figsize=(17,9))
nCars = 3
nViews = 35
carNumbs = [6,8,15,19,23,27,69,76,91,100,][:nCars] # 11 is a cup with different 'zoom' factor.
f='/home/kurtb/Dropbox/code/multiple_relation_embed/coil-100_grey'
resf = '/home/kurtb/Dropbox/code/multiple_relation_embed/res'
imTemplate = scipy.misc.imread('/home/kurtb/Dropbox/code/multiple_relation_embed/coil-100_grey/obj76__0.png')[:,:,0]
P_euc = np.zeros((nViews*len(carNumbs),nViews*len(carNumbs)))

i=-1
Y = np.zeros((nViews*len(carNumbs),len(imTemplate.flatten())))
for icar,car in enumerate(carNumbs):
    ps = np.sort(glob.glob(f+'/obj%d_*.png'%car))
    ps = ps[2:72:2]
    for ip,p in enumerate(ps):
        i+=1
        im = scipy.misc.imread(p).mean(axis=2)#[:,:,0]
        plt.subplot(nCars,len(ps),i+1);plt.imshow(im)
        Y[i,:] = im.flatten()

plt.pause(.01)
#plt.savefig(resf+'/stim_%s.png'%timeMarker) 
P_euc = similarity_matrix(Y)
for i in range(P_euc.shape[0]):
    vsum = np.sum(P_euc[i,:])
    P_euc[i,:] = [np.divide(P_euc[i,j],np.sum(vsum)) for j in range(P_euc.shape[1])]
        
P_block = scipy.linalg.block_diag(*[np.ones((nViews,nViews))/nViews]*nCars)
                    
P = {'euc':P_euc,'block':P_block}

plt.figure()
for i,(n,m) in enumerate(P.items()):
    plt.subplot(1,2,i+1);plt.imshow(m,cmap='viridis');cl.colorbar();plt.title(n)
plt.pause(.1)
plt.tight_layout()
#plt.savefig(resf+'/P_%s.png'%timeMarker) 

#%%
ndim = 3 # of latent space
#mBound = np.zeros((P['euc'].shape[0]+len(P),ndim))
#mBound[:2,:] = 1
#vBound = mBound.flatten()
#bounds = []
#for i in range(len(vBound)):
#    if vBound[i]==0:
#        bounds.append( [-1,1] )
#    else:
#        bounds.append( [None,None] )

#%
        
vRcX = (1+np.random.randn(P['euc'].shape[0]+len(P),ndim).flatten())*(1/ndim) # random init

res = minimize(loss,vRcX,args=(P,ndim),options={'maxiter':200,'disp':True},
               method='Powell',tol=.01)

mRcX = np.reshape(res.x,(P['euc'].shape[0]+len(P),ndim))
X = mRcX[len(P):,:]
dRc = {}
for i,n in enumerate(P.keys()):
    dRc[n] = {'diag':np.diag(mRcX[i,:]),'w':mRcX[i,:]}
    print(n,np.round(mRcX[i,:],2))
    
#%%
cs = np.array(list(range(nViews))*nCars)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2])#,c=cs,cmap=sns.color_palette("GnBu_d"))
# car