import arftools as a
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from sklearn.model_selection import train_test_split 

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
    
class Stochastique(object):
    def __init__(self,loss=hinge,loss_g=hinge_g,max_iter=1000,eps=0.01,p=0.25,b=False):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
            :p: pourcentage du nombre de point visité pour chaque itération 
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g
        self.b=b
        self.p=p
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
        nbPoint=int(self.p*len(datay))
        if(self.loss_g==hinge_g):
            for itt in range(self.max_iter):
                for j in range(nbPoint):
                    ind =random.randint(0,len(datay)-1)
                    avg_grad = hinge_g(np.array(datax[ind]), np.array(datay[ind]), self.w)
                    self.w-=self.eps*avg_grad 
                
        if(self.loss_g==mse_g):
            for itt in range(self.max_iter):
                for j in range(nbPoint):
                    ind =random.randint(0,len(datay)-1)
                    avg_grad= mse_g(np.array(datax[ind]), np.array(datay[ind]), self.w)
                    self.w-=self.eps*np.squeeze(avg_grad)
        if(self.loss_g==hinge_g_bias):
            for itt in range(self.max_iter):
                for j in range(nbPoint):
                    ind =random.randint(0,len(datay)-1)
                    avg_grad = hinge_g_bias(np.array(datax[ind]), np.array(datay[ind]), self.w)
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
   
    def score(self,datax,datay):
        pred=self.predict(datax)
        res=np.maximum(np.array(pred*datay),np.zeros(len(datay)))
        return 1-(sum(res)/len(datay))
class Mini_Batch(object):
    def __init__(self,loss=hinge,loss_g=hinge_g,max_iter=1000,eps=0.01,p=0.25,b=False):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
            :p: pourcentage du nombre de point visité pour chaque itération 
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g
        self.b=b
        self.p=p
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
        nbPoint=int(self.p*len(datay))
        if(self.loss_g==hinge_g):
            for itt in range(self.max_iter):
                mini_datax=[]
                mini_datay=[]
                for j in range(nbPoint):
                    ind =random.randint(0,len(datay)-1)
                    mini_datax.append(datax[ind])
                    mini_datay.append(datay[ind])
                avg_grad = hinge_g(np.array(mini_datax), np.array(mini_datay), self.w)
                self.w-=self.eps*avg_grad 
                
        if(self.loss_g==mse_g):
            for itt in range(self.max_iter):
                for j in range(nbPoint):
                    ind =random.randint(0,len(datay)-1)
                    avg_grad= mse_g(np.array(datax[ind]), np.array(datay[ind]), self.w)
                    self.w-=self.eps*np.squeeze(avg_grad)
        if(self.loss_g==hinge_g_bias):
            for itt in range(self.max_iter):
                for j in range(nbPoint):
                    ind =random.randint(0,len(datay)-1)
                    avg_grad = hinge_g_bias(np.array(datax[ind]), np.array(datay[ind]), self.w)
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
    X_train, X_test, y_train, y_test = train_test_split(dx, dy, test_size=p, random_state=0)
    return np.array(X_train),np.array( X_test), np.array(y_train),np.array( y_test)
def get_two_classes_OneVsAll(datax,datay,c1,p=0.25):
    dx=[]
    dy=[]
    for i in range(len(datax)):
        if(datay[i]==c1):
            dy.append(1)
            dx.append(datax[i])
        else:
            dy.append(-1)
            dx.append(datax[i])
    X_train, X_test, y_train, y_test = train_test_split(dx, dy, test_size=p, random_state=0)
    return np.array(X_train),np.array( X_test), np.array(y_train),np.array( y_test)

def train_test(datax,datay,c1,c2,p):
    datay[datay == c1] = 1
    datay[datay == c2] = -1
    X_train, X_test, y_train, y_test = train_test_split(datax, datay, test_size=p, random_state=0)

if __name__=="__main__":
    """ Tracer des isocourbes de l'erreur """
    
    plt.ion()
    trainx,trainy =  a.gen_arti(nbex=1000,data_type=0,epsilon=1)
    testx,testy =  a.gen_arti(nbex=1000,data_type=0,epsilon=1)
    plt.figure()
    plot_error(trainx,trainy,mse)
    plt.figure()
    plot_error(trainx,trainy,hinge)
    
    
    #hinge (Perceptron)
    
    perceptron = Lineaire(hinge,hinge_g,max_iter=10,eps=0.1)
    perceptron.fit(trainx,trainy)
    print("Erreur hinge: train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    a.plot_frontiere(testx,perceptron.predict,200)
    a.plot_data(testx,testy)
    
    
    #mse (Descente de gradient)
    """
    perceptron = Lineaire(mse,mse_g,max_iter=3,eps=0.1)
    perceptron.fit(trainx,trainy)
    print("Erreur mse: train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    a.plot_frontiere(trainx,perceptron.predict,200)
    a.plot_data(trainx,trainy)
    """
    
    #Hinge avec Biais
    """
    perceptron = Lineaire(hinge,hinge_g_bias,max_iter=1000,eps=0.1,b=True)
    perceptron.fit(trainx,trainy)
    print("Erreur hinge: train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    a.plot_frontiere(testx,perceptron.predict,200)
    a.plot_data(testx,testy)
    """
    
    # Descente de gradient stochastique 
    """
    gradient = Stochastique(hinge,hinge_g,max_iter=1000,eps=0.1,p=0.1)
    gradient.fit(trainx,trainy)
    print("Erreur hinge: train %f, test %f"% (gradient.score(trainx,trainy),gradient.score(testx,testy)))
    plt.figure()
    a.plot_frontiere(testx,gradient.predict,200)
    a.plot_data(testx,testy)
    """
    #Mini-Batch
    """
    gradient = Mini_Batch(hinge,hinge_g,max_iter=1000,eps=0.1,p=0.25)
    gradient.fit(trainx,trainy)
    print("Erreur hinge: train %f, test %f"% (gradient.score(trainx,trainy),gradient.score(testx,testy)))
    plt.figure()
    a.plot_frontiere(testx,gradient.predict,200)
    a.plot_data(testx,testy)
    """
    
    # Données USPS
    """
    datax , datay = load_usps ("USPS_train.txt") 
    trainx,testx,trainy,testy=get_two_classes(datax,datay,1,0)
    trainx6,testx6,trainy6,testy6=get_two_classes_OneVsAll(datax,datay,6)
    
    sc=[]
    sct=[]
    y=[]
    for itt in range(200):
        
        perceptron = Lineaire(hinge,hinge_g,max_iter=itt,eps=0.1)
        perceptron.fit(trainx,trainy)
        sc.append(perceptron.score(trainx,trainy))
        sct.append(perceptron.score(testx,testy))
        y.append(itt)
        
    plt.plot(y,sc,label="train")
    plt.plot(y,sct,label="test")
    plt.xlabel("Nombre d'itération")
    plt.ylabel("Erreur hinge")
    plt.legend()
    plt.show()
    
    """

    #hinge
    """
    perceptron = Lineaire(hinge,hinge_g,max_iter=100,eps=0.1)
    perceptron.fit(trainx,trainy)
    print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    w = perceptron.w
    weights = w[0][0:256].reshape((16,16))
    plt.imshow(weights)
    plt.colorbar()
    plt.show()
    """
    #mse
    """
    perceptron = Lineaire(mse,mse_g,max_iter=100,eps=0.1)
    perceptron.fit(trainx,trainy)
    print("Erreur mse: train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    w = perceptron.w
    weights = w[0][0:256].reshape((16,16))
    plt.imshow(weights)
    plt.colorbar()
    plt.show()
    """
    #Tests 
    # 6 VS ALL
    """
    perceptron = Lineaire(mse,mse_g,max_iter=1000,eps=0.01)
    perceptron.fit(trainx6,trainy6)
    print(" classe 6 vs all classes")
    print("Erreur : train %f, test %f"% (perceptron.score(trainx6,trainy6),perceptron.score(testx6,testy6)))
    w = perceptron.w
    weights = w[0][0:256].reshape((16,16))
    
    plt.imshow(weights)
    plt.colorbar()
    plt.show()
    
    """
    # 6 VS ALL one by one
    """
    for i in range(10):
        if(i!=6):
            trainx,testx,trainy,testy=get_two_classes(datax,datay,i,6)
            #hinge
            print(" classe 6 vs classe ",i)
            perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1)
            perceptron.fit(trainx,trainy)
            print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
            w = perceptron.w
            weights = w[0][0:256].reshape((16,16))
            plt.imshow(weights)
            plt.colorbar()
            plt.show()
    """
    

    # Erreur hinge
    """
    sc=[]
    sct=[]
    y=[]
    for itt in range(1000):
        
        perceptron = Lineaire(hinge,hinge_g,max_iter=itt,eps=0.1)
        perceptron.fit(trainx,trainy)
        sc.append(perceptron.score(trainx,trainy))
        sct.append(perceptron.score(testx,testy))
        y.append(itt)
        
    plt.plot(y,sc,label="train")
    plt.plot(y,sct,label="test")
    plt.xlabel("Nombre d'itération")
    plt.ylabel("Erreur hinge")
    plt.legend()
    plt.show()
    """
   
    # Erreur mse
    """ 
    sc=[]
    sct=[]
    y=[]
    for itt in range(10,1000,10):
        
        perceptron = Lineaire(hinge,hinge_g,max_iter=itt,eps=0.5)
        perceptron.fit(trainx,trainy)
        sc.append(perceptron.score(trainx,trainy))
        sct.append(perceptron.score(testx,testy))
        y.append(itt)
        
    plt.plot(y,sc,label="train")
    plt.plot(y,sct,label="test")
    plt.xlabel("Nombre d'itération")
    plt.ylabel("Erreur mse")
    plt.legend()
    plt.show()
    
"""

    