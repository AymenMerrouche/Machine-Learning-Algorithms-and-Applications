import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import tme2 as t
from sklearn import metrics
plt.ion()
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')

## coordonnees GPS de la carte
xmin,xmax = 2.23,2.48   ## coord_x min et max
ymin,ymax = 48.806,48.916 ## coord_y min et max

def show_map():
    plt.imshow(parismap,extent=[xmin,xmax,ymin,ymax],aspect=1.5)
    ## extent pour controler l'echelle du plan

poidata = pickle.load(open("data/poi-paris.pkl","rb"))
## liste des types de point of interest (poi)
print("Liste des types de POI" , ", ".join(poidata.keys()))

## Choix d'un poi
typepoi = "night_club"

## Creation de la matrice des coordonnees des POI
geo_mat = np.zeros((len(poidata[typepoi]),2))
notes_mat=np.zeros((len(poidata[typepoi])))
for i,(k,v) in enumerate(poidata[typepoi].items()):
    geo_mat[i,:]=v[0]
    notes_mat[i]=v[1]

## Affichage brut des poi
show_map()
## alpha permet de regler la transparence, s la taille
plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.8,s=3)


###################################################

# discretisation pour l'affichage des modeles d'estimation de densite
steps = 100
xx,yy = np.meshgrid(np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps))
grid = np.c_[xx.ravel(),yy.ravel()]

# A remplacer par res = monModele.predict(grid).reshape(steps,steps)

"""1------------------- Méthode histogramme"""
#res = np.random.random((steps,steps))
"""
print("Méthode histogramme")
res=t.methode_hist(geo_mat,grid).reshape(steps,steps)
plt.figure()
show_map()
plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()
plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.3)"""

"""2------------------- Méthode à noyaux"""
#--------------Fenêtre de Parzen
print("Méthode à noyaux")

h=0.005
"""
res=t.methode_noyaux(geo_mat,grid,h).reshape(steps,steps)
plt.figure()
show_map()
plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()
plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.3)
"""
#--------------Gaussienne
"""
res=t.methode_noyaux_gauss(geo_mat,grid,h).reshape(steps,steps)
plt.figure()
show_map()
plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()
plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.3)
"""

"""3------------------- Densité en fonction des POI"""

## Choix d'un poi
"""furniture_store, laundry, bakery, cafe, home_goods_store, clothing_store, atm, lodging, night_club, 
convenience_store, restaurant, bar"""

typepoi = "cafe"

## Creation de la matrice des coordonnees des POI
geo_mat = np.zeros((len(poidata[typepoi]),2))
for i,(k,v) in enumerate(poidata[typepoi].items()):
    geo_mat[i,:]=v[0]

## Affichage brut des poi
show_map()
## alpha permet de regler la transparence, s la taille
plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.8,s=3)


###################################################

# discretisation pour l'affichage des modeles d'estimation de densite
steps = 100
xx,yy = np.meshgrid(np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps))
grid = np.c_[xx.ravel(),yy.ravel()]

#--------------Fenêtre de Parzen
print("Méthode à noyaux")

h=0.01
"""
res=t.methode_noyaux(geo_mat,grid,h).reshape(steps,steps)
plt.figure()
show_map()
plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()
plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.3)
"""

"""4------------------- Prédiction de Notes"""
#--------------KNN
print("KNN")
emoy=[]
emax=[]
list_k=np.arange(1,21)
for k in range(1,21):
    res=t.knn(geo_mat,notes_mat,k)
    emoy.append(metrics.mean_squared_error(notes_mat, res))
    emax.append(metrics.max_error(notes_mat, res))
plt.plot(list_k,emoy,label="mean squared error")
plt.ylabel("K")
plt.title("Evaluation de KNN en fonction de k")
plt.plot(list_k,emax,label="max error")
plt.legend()
plt.show()
#-------------- Nadaraya-Watson
print("Nadaraya-Watson")
emoy2=[]
emax2=[]
list_h=np.arange(1,21)
for h in list_h:
    res=t.NadarayaWatson(geo_mat,notes_mat,h)
    emoy2.append(metrics.mean_squared_error(notes_mat, res))
    emax2.append(metrics.max_error(notes_mat, res))


plt.plot(list_h,emoy2,label="mean squared error")
plt.ylabel("h")
plt.title("Evaluation de Nadaraya-Watson en fonction de h")
plt.plot(list_h,emax2,label="max error")
plt.legend()
plt.show()












