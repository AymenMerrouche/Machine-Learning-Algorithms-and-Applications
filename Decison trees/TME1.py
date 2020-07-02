#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:57:13 2020

@author: 3803192
"""

import collections as c
import math
import  pickle
import  numpy as np
from  decisiontree  import  DecisionTree
import pydot
import matplotlib.pyplot as plt

def entropie(vect):
    l=len(vect)
    cnt=c.Counter()
    for y in vect:
        cnt[y]+=1
    H=0
    for y in cnt:
        p=cnt[y]/l
        H+=-(p*math.log(p))
    return H

def entropie_cond(list_vect):
    long=len([y for x in list_vect for y in x])
    H=0
    for P in list_vect :
        pi= len(P)/long
        hi=entropie(P)
        H += -pi*hi
    return H

[data , id2titles , fields ]= pickle.load(open("imdb_extrait.pkl","rb"))
# la  derniere  colonne  est le vote
datax=data [: ,:32]
datay=np.array ([1 if x[33] >6.5  else  -1 for x in data])

"""  
######## Entropie  
vect=[1,2,3,4,5,6,7,89,1,2,3,1,1,1]
vvect=[[1,2,2],[4,2,1],[8,5,7]]
print(entropie(vect))
print(entropie_cond(vvect))
"""

"""
######## Visualiser l'arbre de décision
dt = DecisionTree ()
dt.max_depth = 3#on fixe la  taille  de l’arbre a 5
dt.min_samples_split = 2#nombre  minimum d’exemples  pour  spliter  un noeud
dt.fit(datax ,datay)
dt.predict(datax [:5 ,:])
print(dt.score(datax ,datay))# dessine l’arbre  dans un  fichier  pdf   si pydot  est  installe.
dt.to_pdf("/tmp/test_tree.pdf",fields)# sinon  utiliser  http :// www.webgraphviz.com/
dt.to_dot(fields)#ou dans la  console
print(dt.print_tree(fields ))
"""
"""
######## Le score par profondeur
res=[]
for i in range (1,15):
    dt = DecisionTree ()
    dt.max_depth = i
    dt.min_samples_split = 2
    dt.fit(datax ,datay)
    dt.predict(datax [:5 ,:])
    res.append(dt.score(datax ,datay))
p=np.arange(1,15)
plt.plot(p,res)
plt.show()
"""
######## Le score par profondeur Pour Test/Train
"""
taux_test=0.8
taux_train=0.2
trainx=[]
testx=[]
trainy=[]
testy=[]

trainx1=[]
testx1=[]
trainy1=[]
testy1=[]

trainx2=[]
testx2=[]
trainy2=[]
testy2=[]


for i in range(int(taux_train*len(datax))):
    trainx.append(list(datax[i]))
    trainy.append(datay[i])
    testx2.append(list(datax[i]))
    testy2.append(datay[i])
for j in range(i,len(datax)):
    testx.append(list(datax[j]))
    testy.append(datay[j]) 
    trainx2.append(list(datax[j]))
    trainy2.append(datay[j])
for i in range(int(0.5*len(datax))):
    trainx1.append(list(datax[i]))
    trainy1.append(datay[i])
for j in range(i,len(datax)):
    testx1.append(list(datax[j]))
    testy1.append(datay[j])  
trainx=np.array(trainx)
testx=np.array(testx)
trainy=np.array(trainy)
testy=np.array(testy)

trainx1=np.array(trainx1)
testx1=np.array(testx1)
trainy1=np.array(trainy1)
testy1=np.array(testy1)

trainx2=np.array(trainx2)
testx2=np.array(testx2)
trainy2=np.array(trainy2)
testy2=np.array(testy2)

res1_test=[]
res1_train=[]
for i in range (1,15):
    dt = DecisionTree ()
    dt.max_depth = i
    dt.min_samples_split = 2
    dt.fit(trainx ,trainy)
    res1_train.append(dt.score(trainx ,trainy))
    res1_test.append(dt.score(testx ,testy))
    
p=np.arange(1,15)
plt.plot(p,res1_train,label="Score en apprentissage")
plt.plot(p,res1_test,label="Score en test")
plt.legend()
plt.show()
    
res2_test=[]
res2_train=[]
for i in range (1,15):
    dt = DecisionTree ()
    dt.max_depth = i
    dt.min_samples_split = 2
    dt.fit(trainx1 ,trainy1)
    res2_train.append(dt.score(trainx1 ,trainy1))
    res2_test.append(dt.score(testx1 ,testy1))
plt.plot(p,res2_train,label="Score en apprentissage")
plt.plot(p,res2_test,label="Score en test")
plt.legend()
plt.show()    
res3_test=[]
res3_train=[]
for i in range (1,15):
    dt = DecisionTree ()
    dt.max_depth = i
    dt.min_samples_split = 2
    dt.fit(trainx2 ,trainy2)
    res3_train.append(dt.score(trainx2 ,trainy2))
    res3_test.append(dt.score(testx2 ,testy2))
plt.plot(p,res3_train,label="Score en apprentissage")
plt.plot(p,res3_test,label="Score en test")
plt.legend()
plt.show()
"""
######## Validation croisée
nb_folds=3
size=int(len(datax)/nb_folds)
foldsx=[]
foldsy=[]
for i in range(nb_folds):
    x=[]
    y=[]
    for j in range(i*size,(i+1)*size):
        x.append(datax[j])
        y.append(datay[j])
    foldsx.append(np.array(x))
    foldsy.append(np.array(y))

error_it=[]
for i in range(nb_folds):
    testx=foldsx[i]
    testy=foldsy[i]
    tx=[]
    ty=[]
    for j in range(nb_folds):
        if(j != i):
            tx.append(foldsx[j])
            ty.append(foldsy[j])
    trainx=np.concatenate(tx,axis=0)
    trainy=np.concatenate(ty,axis=0)
    dt = DecisionTree ()
    dt.max_depth = 4
    dt.min_samples_split = 2
    dt.fit(trainx ,trainy)
    error_it.append(dt.score(testx ,testy))   
error_moy=sum(error_it)/nb_folds   
print(error_it)
print(error_moy)