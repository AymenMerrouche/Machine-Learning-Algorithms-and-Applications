import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_data(data,labels=None):
    """
    Affiche des donnees 2D
    :param data: matrice des donnees 2d
    :param labels: vecteur des labels (discrets)
    :return:
    """
    cols,marks = ["red", "green", "blue", "orange", "black", "cyan"],[".","+","*","o","x","^"]
    if labels is None:
        plt.scatter(data[:,0],data[:,1],marker="x")
        return
    for i,l in enumerate(sorted(list(set(labels.flatten())))):
        plt.scatter(data[labels==l,0],data[labels==l,1],c=cols[i],marker=marks[i])

def plot_frontiere(data,f,step=20):
    """ Trace un graphe de la frontiere de decision de f
    :param data: donnees
    :param f: fonction de decision
    :param step: pas de la grille
    :return:
    """
    grid,x,y=make_grid(data=data,step=step)
    plt.contourf(x,y,f(grid).reshape(x.shape),colors=('gray','blue'),levels=[-1,0,1])

def make_grid(data=None,xmin=-5,xmax=5,ymin=-5,ymax=5,step=20):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :param data: pour calcluler les bornes du graphe
    :param xmin: si pas data, alors bornes du graphe
    :param xmax:
    :param ymin:
    :param ymax:
    :param step: pas de la grille
    :return: une matrice 2d contenant les points de la grille
    """
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:,0]),  np.min(data[:,0]), np.max(data[:,1]), np.min(data[:,1])
    x, y =np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step), np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

def gen_arti(centerx=1,centery=1,sigma=0.1,nbex=1000,data_type=0,epsilon=0.02):
    """ Generateur de donnees,
        :param centerx: centre des gaussiennes
        :param centery:
        :param sigma: des gaussiennes
        :param nbex: nombre d'exemples
        :param data_type: 0: melange 2 gaussiennes, 1: melange 4 gaussiennes, 2:echequier
        :param epsilon: bruit dans les donnees
        :return: data matrice 2d des donnnes,y etiquette des donnnees
    """
    if data_type==0:
         #melange de 2 gaussiennes
         xpos=np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex//2)
         xneg=np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex//2)
         data=np.vstack((xpos,xneg))
         y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))
    if data_type==1:
        #melange de 4 gaussiennes
        xpos=np.vstack((np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex//4),np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex//4)))
        xneg=np.vstack((np.random.multivariate_normal([-centerx,centerx],np.diag([sigma,sigma]),nbex//4),np.random.multivariate_normal([centerx,-centerx],np.diag([sigma,sigma]),nbex//4)))
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))

    if data_type==2:
        #echiquier
        data=np.reshape(np.random.uniform(-4,4,2*nbex),(nbex,2))
        y=np.ceil(data[:,0])+np.ceil(data[:,1])
        y=2*(y % 2)-1
    # un peu de bruit
    data[:,0]+=np.random.normal(0,epsilon,nbex)
    data[:,1]+=np.random.normal(0,epsilon,nbex)
    # on mélange les données
    idx = np.random.permutation((range(y.size)))
    data=data[idx,:]
    y=y[idx]
    return data,y


def mse(datax,datay,w):
    
    """ retourne la moyenne de l'erreur aux moindres carres """   
    loss = np.dot((np.dot(datax,w.T)-datay),(np.dot(datax,w.T)-datay).T)  
    return loss/len(datay)

def mse_g(datax,datay,w):
    
    """ retourne le gradient moyen de l'erreur au moindres carres """        
    grad = 2*np.dot(datax.T,(np.dot(datax,w.T)-datay))
    return grad/len(datay)

def hinge(datax,datay,w):
    
    """ retourne la moyenne de l'erreur hinge """
    datax,datay=datax.reshape(len(datay),-1),datay.reshape(-1,1)
    if (len(datax.shape)==1): 
        datax = datax.reshape(1,-1)
    
    loss=-np.dot(datay.T,np.dot(datax,w.T))
    #loss=-datay*np.dot(datax,w.T).T
    loss_hinge=np.maximum(np.zeros(len(loss)),loss)
    return np.mean(loss_hinge)

def hinge_g(datax,datay,w):
    """yx=np.dot(datay.squeeze(),datax)
    #loss=-np.dot(datay,np.dot(datax,w.T))
    loss=-datay.squeeze()*np.dot(datax,w.T).squeeze()   # 1-n
    loss_hinge=np.maximum(np.zeros(len(loss)),loss)
    print(loss_hinge)
    
    grad=(loss_hinge*yx)/loss
    #print(loss_hinge)
    #print(grad)
    return np.mean(grad)"""
    """ retourne le gradient moyen de l'erreur hinge """
    datax,datay=datax.reshape(len(datay),-1),datay.reshape(-1,1)
    loss = np.maximum(np.zeros(len(datay)), -np.squeeze(np.dot(datax,w.T))*np.squeeze(datay))
    loss = loss.reshape(-1,1)
    l=(np.squeeze(np.dot(datax,w.T))*np.squeeze(datay)).reshape(-1,1)
    xy=np.dot(np.squeeze(datay).T,datax)
    
    g = (loss*xy)/l
    return np.mean(g, axis = 0)

def hinge_g_bias(datax,datay,w):
    """ retourne le gradient moyen de l'erreur hinge avec biais """
    datax,datay=datax.reshape(len(datay),-1),datay.reshape(-1,1)
    #On rajoute une 3éme dim en guise de biais contenant que des 1
    dim=np.ones((len(datay),1))
    Xbias = np.hstack((datax,dim))
    
    loss = np.maximum(np.zeros(len(datay)), -np.squeeze(np.dot(Xbias,w.T))*np.squeeze(datay))
    loss = loss.reshape(-1,1)
    l=(np.squeeze(np.dot(Xbias,w.T))*np.squeeze(datay)).reshape(-1,1)
    xy=np.dot(np.squeeze(datay).T,Xbias)
    g = (loss*xy)/l
    return np.mean(g, axis = 0)

class Lineaire(object):
    def __init__(self,loss=hinge,loss_g=hinge_g,max_iter=1000,eps=0.01,b=False):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g
        self.b=b
        self.saveW=[]
        self.saveP=[]
    def fit(self,datax,datay,testx=None,testy=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        # on transforme datay en vecteur colonne
        datay = datay.reshape(-1,1)
        N = len(datay)
        datax = datax.reshape(N,-1)
        D = datax.shape[1]
        if(self.b==True):
            self.w = np.random.random((1,D+1))
        else:
            self.w = np.random.random((1,D))
            
        #Perceptron
        if(self.loss_g==hinge_g):
            for itt in range(self.max_iter):
                avg_grad = hinge_g(datax, datay, self.w)
                self.saveW.append(self.w)
                self.w-=self.eps*avg_grad 
                
        if(self.loss_g==mse_g):
            for itt in range(self.max_iter):
                avg_grad= mse_g(datax, datay, self.w)
                self.w-=self.eps*np.squeeze(avg_grad)
        if(self.loss_g==hinge_g_bias):
            for itt in range(self.max_iter):
                avg_grad = hinge_g_bias(datax, datay, self.w)
                self.w-=self.eps*avg_grad
                
    def predict(self,datax):
        if len(datax.shape)==1:
            datax = datax.reshape(1,-1)
        predict=[]
        #Percepton
        if(self.b==False):#Sans biais 
            for i in range(len(datax)):
                val=np.dot(datax[i],self.w.T)
                if(val>0):
                    predict.append(1)
                else:
                    predict.append(-1)
        else:#Avec biais 
            
            for i in range(len(datax)):
                val=np.dot(np.concatenate((datax[i],[1])),self.w.T) 
                if(val>0):
                    predict.append(1)
                else:
                    predict.append(-1)
        self.saveP.append(np.array(predict))
        return np.array(predict)
    """def score(self,datax,datay):
        pred=self.predict(datax)
        return sum(pred*datay)/len(datay)"""
    def score(self,datax,datay):
        pred=self.predict(datax)
        res=np.maximum(np.array(pred*datay),np.zeros(len(datay)))
        return 1-(sum(res)/len(datay))
    


def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")



def plot_error(datax,datay,f,step=10):
    grid,x1list,x2list=a.make_grid(xmin=-4,xmax=4,ymin=-4)
    
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.show()

def get_two_classes(datax,datay,c1,c2,p=0.25):
    dx=[]
    dy=[]
    for i in range(len(datax)):
        if(datay[i]==c1):
            dy.append(1)
            dx.append(datax[i])
        if(datay[i]==c2):
            dy.append(-1)
            dx.append(datax[i])
    return np.array(dx),np.array(dy)


def plot_frontiere_proba( data , f , step=20):
    grid , x , y=make_grid( data=data , step=step )
    plt.contourf(x , y , f(grid).reshape(x.shape),255)
