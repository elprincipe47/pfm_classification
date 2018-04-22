import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

from scipy.sparse import hstack, csr_matrix

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import linear_model

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import time
import pickle
from sklearn.model_selection import KFold

import gc

train_csv = pd.read_csv("DATA_PFM_Train.csv",).fillna(' ')

models = [
     OneVsRestClassifier(estimator=linear_model.SGDClassifier(max_iter= 100)),
     OneVsRestClassifier(estimator=linear_model.PassiveAggressiveClassifier(max_iter= 100)),
     OneVsRestClassifier(estimator=linear_model.RidgeClassifier()),
     ExtraTreesClassifier(),
     OneVsRestClassifier(estimator=KNeighborsClassifier())
    ]
STOP_WORDS= ["EL JADI"," AIN SEB","PAIEMENT CARTE:","NADOR","EL","SIDI","ESSAOUIR","SALE","TETOUAN","AIN","AGADIR","AG","CASA","CASABLANC","RBT","RAK","MKEC","MARRAKE","ECOM","AGAD","TANG","TEMA","STE"," AIT"," MELLO","MOHAMMEDI"," LA","TANGER","DERB SU","EL KALA","HAY"]

num_categorie = 15

def numcategorie_tovector(number):
    vec = np.zeros(num_categorie)
    if(number < 1 or number > num_categorie):
        return vec
    else:
        vec[number-1] = 1
    return vec

libelles_train = train_csv["LIBELLEOPERATION"]
yy_train= train_csv["CATEGORIE"]

del train_csv
gc.collect()

num_unic_words = len(np.unique(np.hstack(libelles_train)))
y_train= yy_train.apply(lambda x: numcategorie_tovector(x))
liste_des_vecteurs_de_categories = []

for element in y_train:
    liste_des_vecteurs_de_categories.append(element)

liste_des_vecteurs_de_categories = np.array(liste_des_vecteurs_de_categories)
y_train = liste_des_vecteurs_de_categories

del liste_des_vecteurs_de_categories
del yy_train
gc.collect()
X = libelles_train

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words=STOP_WORDS,
    ngram_range=(1, 1),
    max_features=5000)

word_vectorizer.fit(X)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words=STOP_WORDS,
    ngram_range=(2, 3),
    max_features=10000)
char_vectorizer.fit(X)

pickle.dump([char_vectorizer, word_vectorizer], open('vetorizers.pickle', 'wb'), protocol=0)

print("split word_tfidf")

train_word_features = word_vectorizer.transform(libelles_train)

gc.collect()

print("split char_tfidf")

train_char_features = char_vectorizer.transform(libelles_train)

print("aggregate_features")

train_features = hstack([train_char_features, train_word_features]).tocsr()



for model in models:
                model.fit(train_features,y_train)

pickle.dump(models, open('models.pickle', 'wb'), protocol=0)
