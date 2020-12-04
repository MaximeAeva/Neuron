import requests
from bs4 import BeautifulSoup
import csv
import time
import codecs
import numpy as np
import math

'''Rennes
Bordeaux
Reims
Strasbourg
Perpignan
Embrun'''
'''Matching each climat cities'''
listeVilles = {
            "OceaniqueNordOuest": "07130", 
            "OceaniqueAquitain": "07510", 
            "OceaniqueDegrade": "07070", 
            "SemiContinental": "07190", 
            "Mediteraneen": "07747", 
            "Montagnard": "07591"} 

'''Checked param'''
param = ["temperature", "pression", "vent_moyen", "vent_direction", "pluie_3h"]

def ImportHTML (listeVilles, param):
    
    '''Request for post'''
    listeRequetes = {
                "stations[]":" ",
                "type": "3h",
                "jour1": "01",
                "mois1": "01",
                "an1": "2014",
                "jour2": "31",
                "mois2": "12",
                "an2": "2019",
                "param_3h[]": "",
                "param_mj[]": "tmoy"}
    
    '''Loop over cities and parameters'''
    for cle in listeVilles.keys():
        content = {}
        for parm in param:
            listeRequetes["param_3h[]"] =  parm
            listeRequetes["stations[]"] =  listeVilles[cle]
            
            '''requesting'''
            page = requests.post('https://www.infoclimat.fr/climatologie/stations_principales.php?', data= listeRequetes)
        
            '''Creating the soup'''
            soup = BeautifulSoup(page.content, 'html.parser')
            
            '''Save .html file'''
            with open(cle+parm+".html", "w", encoding='utf-8') as file:
                file.write(str(soup))
            
            '''Prevent from DDOS'''
            time.sleep(30)
            
def LoadDataSet (cle, param, size, train = 0.7):
    X = []
    Y = []
    content={} 
    for parm in param:
        '''Open .html from computer'''
        page = codecs.open("../data/"+cle+parm+".html", 'r', 'utf-8')
        
        ''' Creating the soup'''
        soup = BeautifulSoup(page, 'html.parser')    
           
        '''Find instance'''
        context = soup.find_all('td', class_="separation-param")
        
        '''Creating a dict for write'''
        content[parm]=[]
        for alone in context:
            '''Sort Nan, figures and others'''
            if not alone.text:
                content[parm].append('NaN')
            elif alone.text[len(alone.text)-1].isdigit() :
                content[parm].append(float(alone.text))
                
    #Througt the data
    for i in range(0, len(content[param[0]])):
        vect = []
        #We count the number of NaN
        for c in param:
            vect.append(content[c][i:i+size[0]+1].count('NaN'))
        #We go after the farest NaN
        indexOfNan = [i for i,val in enumerate(vect) if val!=0]
        step = [-1]
        for c in indexOfNan:
           step.append( max([i for i,val in enumerate(content[param[c]][i:i+size[0]+1]) if val=='NaN']))
        
        if max(step)==-1:
            x=[]
            y=[]
            for c in param:
                x=x+content[c][i:i+size[0]-1]
                y.append(content[c][i+size[0]])
            x = np.asarray(x)[:, np.newaxis]
            y = np.asarray(y)[:, np.newaxis]
            if X == []:
                X = x
                Y = y
            else:
                X = np.concatenate((X, x), axis = 1)
                Y = np.concatenate((Y, y), axis = 1)
            i = i+1
        else:
            i = i+max(step)+1
            
        if X.shape[1] == size[1]:
            permutation = list(np.random.permutation(X.shape[1]))
            X_train = X[:, permutation[0:math.floor(train*X.shape[1])]]
            Y_train = Y[:, permutation[0:math.floor(train*X.shape[1])]]
            X_test = X[:, permutation[math.floor(train*X.shape[1]):len(permutation)]]
            Y_test = Y[:, permutation[math.floor(train*X.shape[1]):len(permutation)]]
            return X_train, Y_train, X_test, Y_test
        if i>len(content[param[0]])-size[0]:
            permutation = list(np.random.permutation(X.shape[1]))
            X_train = X[:, permutation[0:math.floor(train*X.shape[1])]]
            Y_train = Y[:, permutation[0:math.floor(train*X.shape[1])]]
            X_test = X[:, permutation[math.floor(train*X.shape[1]):len(permutation)]]
            Y_test = Y[:, permutation[math.floor(train*X.shape[1]):len(permutation)]]
            return X_train, Y_train, X_test, Y_test
    
                
#X_train, Y_train, X_test, Y_test = LoadDataSet ("OceaniqueAquitain", param, [10, 1000], 0.8)
