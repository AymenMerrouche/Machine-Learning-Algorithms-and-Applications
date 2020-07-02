#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:46:19 2020

@author: Touzari
"""

import numpy as np
import pandas as pd
import math
import statistics
from scipy.stats import norm

def methode_hist(coord,grid):
    cpt=0
    res=[]
    f=False
    N=len(coord)
    x=sorted(list(set([i[0] for i in grid]))) #2
    y=sorted(list(set([i[1] for i in grid])))
    
    for i1 in range(len(x)-1):
            for i2 in range(len(y)-1):
                cpt=0
                for j in coord:
                    if((x[i1]<=j[1]) and (j[1]<x[i1+1]) and (y[i2]<=j[0]) and (j[0]<y[i2+1])):
                        cpt+=1
                    
                res.append(cpt)
    i1=len(x)-1
    for i2 in range(len(y)-1):
                cpt=0
                for j in coord:
                    if((x[i1]<=j[1]) and (j[1]<x[i1+1]) and (y[i2]<=j[0]) and (j[0]<y[i2+1])):
                        cpt+=1
                    
                res.append(cpt)  
    i2=len(y)-1
    for i1 in range(len(x)-1):
                cpt=0
                for j in coord:
                    if((x[i1]<=j[1]) and (j[1]<x[i1+1]) and (y[i2]<=j[0]) and (j[0]<y[i2+1])):
                        cpt+=1
                    
                res.append(cpt)
    res.append(N-sum(res))
    
    
    return np.array([i/N for i in res])

def phi(c,g,h):
    if((np.abs(c[0]-g[1])<h) and (np.abs(c[1]-g[0])<h)):
        return 1
    return 0

def methode_noyaux(coord,grid,h):
    xx=[]
    yy=[]
    for c in coord:
        xx.append(c[0])
        yy.append(c[1])
    #ar = np.array([xx,yy])
    #df = pd.DataFrame(ar)
    proba=[]
    N=len(coord)
    rd=h**2
    for i in range(len(grid)):
        cpt=0
        for j in range(N):
            cpt+=phi(coord[j],grid[i],h)
            
        
        proba.append((cpt/(N*rd)))  
    som=sum(proba)
    return np.array([i/float(som) for i in proba])     
        

def phi_gauss(c,g,h,var):
    
    return norm.cdf(c[0],g[1],var[0])*norm.cdf(c[1],g[0],var[1])


def methode_noyaux_gauss(coord,grid,h):
    xx=[]
    yy=[]
    for c in coord:
        xx.append(c[0])
        yy.append(c[1])
    #ar = np.array([xx,yy])
    #df = pd.DataFrame(ar)
    proba=[]
    N=len(coord)
    rd=h**2
    #var=[statistics.variance(xx),statistics.variance(yy)]
    
    for i in range(len(grid)):
        cpt=0
        for j in range(N):   
            cpt+=phi_gauss(coord[j],grid[i],h,[1,1])
            
        #print(cpt/(N*rd))
        proba.append((cpt/(N*rd)))  
    som=sum(proba)
    return np.array([i/float(som) for i in proba])     
    #df['P']=proba  
    #return df
def knn(geo_mat,note_mat,k):
    res=[]
    for i in range(len(note_mat)):
        dist={}
        for j in range(len(note_mat)):
            d=math.sqrt((geo_mat[j][0]-geo_mat[i][0])**2 + (geo_mat[j][1]-geo_mat[i][1])**2)
            dist[j]=d
        dist_ord={k: v for k, v in sorted(dist.items(), key=lambda item: item[1])}
        it=0
        moy=0
        for ind in dist_ord.keys():
            moy+=note_mat[ind]
            it+=1
            if(it==k):
                break;
        res.append(round(moy/k,2))
    return res
            
def NadarayaWatson(geo_mat,note_mat,h):
    res=[]
    d_mat=[[(math.sqrt((pos2[0]-pos1[0])**2 +(pos2[1]-pos1[1])**2)) for pos2 in geo_mat] for pos1 in geo_mat]
    for i in range(len(note_mat)):
        s=0
        for j in range(len(note_mat)):
            s+=((d_mat[i][j]/h)/(sum(d_mat[i])/h))*note_mat[j]
        res.append(round(s,2))
    return res   











