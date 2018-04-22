import numpy as np
import pandas as pd


from scipy.sparse import hstack, csr_matrix



import pickle

import gc

file = open("models.pickle", "rb")
unpickler = pickle.Unpickler(file)
models = unpickler.load()


file = open("vetorizers.pickle", "rb")
unpickler = pickle.Unpickler(file)
vecs = unpickler.load()
char_vectorizer, word_vectorizer = vecs[0], vecs[1]

STOP_WORDS= ["EL JADI"," AIN SEB","PAIEMENT CARTE:","NADOR","EL","SIDI","ESSAOUIR","SALE","TETOUAN","AIN","AGADIR","AG","CASA","CASABLANC","RBT","RAK","MKEC","MARRAKE","ECOM","AGAD","TANG","TEMA","STE"," AIT"," MELLO","MOHAMMEDI"," LA","TANGER","DERB SU","EL KALA","HAY"]
num_categorie = 15
classes_names = ['Shopping','Cafes et Restaurants ','Courses ','Transport et Voiture',
           'Divers','Utilitaires','Multimedia et Electronique'
           ,'Sante et Pharmacie','Voyage et Vacance ','Divertissement',
           'Education ','AmeublementEtElectromenager','BanqueEtAssurance','Taxes']

classes_keys = range(1,num_categorie)
dictionnaire = dict(zip(classes_keys,classes_names))
def numcategorie_tovector(number):
    vec = np.zeros(num_categorie)
    if(number < 1 or number > num_categorie):
        return vec
    else:
        vec[number-1] = 1
    return vec


# you should upload the file from google frive to run the test
f = open('DATA_PFM.csv','w')
index = 1

for chunk in pd.read_csv("data-table.csv", chunksize=5000):
    print('chunk n : ',index)
    index+=1
    libelles_test =chunk["LIBELLE"]
    libelles_test = np.array(libelles_test)
    X = libelles_test
    test_word_features = word_vectorizer.transform(libelles_test)
    gc.collect()
    test_char_features = char_vectorizer.transform(libelles_test)
    test_features = hstack([test_char_features, test_word_features]).tocsr()
    del test_word_features
    del test_char_features
    preds = []
    for model in models:
                    yy_test = model.predict(test_features)
                    preds.append(yy_test)
    preds = np.array(preds)
    preds = np.mean(preds, axis=0)
    predictions = []
    for element in preds:
        #predictions.append(np.argmax(element, axis=0)+1)
        predictions.append(dictionnaire.get(np.argmax(element, axis=0)+1))
    output = zip(libelles_test, predictions)    
    for e in output :
        f.write(str(e[0])+','+str(e[1])+'\n')
f.close()

