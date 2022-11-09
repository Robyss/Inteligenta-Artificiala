print("Importing libraries...")
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Loading data...")

dataPath = './Data/'


X_stop_digits_lemma = np.load(dataPath + 'X_stop_digits_lemma.npy', allow_pickle=True)
X_stop_lemma        = np.load(dataPath + 'X_stop_lemma.npy', allow_pickle=True)
X_stop_stem         = np.load(dataPath + 'X_stop_stem.npy', allow_pickle=True)
X_stop              = np.load(dataPath + 'X_stop.npy', allow_pickle=True)
X                   = np.load(dataPath + 'X.npy', allow_pickle=True)

y = np.load(dataPath + 'y.npy', allow_pickle=True)

print("Data loaded.")
train_data = {'X': X, 'X_stop': X_stop, 'X_stop_stem': X_stop_stem, 'X_stop_lemma': X_stop_lemma, 'X_stop_digits_lemma': X_stop_digits_lemma}


# iterate with for through train_data
for key, value in train_data.items():
    print("Training with " + key + "...")
    X_train, X_test, y_train, y_test = train_test_split(value, y, test_size=0.2, random_state=42)
    
    for c in range(20000, 200000, 10000):
        vectorizer = TfidfVectorizer(strip_accents='unicode', sublinear_tf=True, max_features=c)
        x_train = vectorizer.fit_transform(X_train)
        x_test = vectorizer.transform(X_test)

        clf = LogisticRegression(random_state=0).fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(f'Accuracy for {key} with {c} features:  {accuracy_score(y_test, y_pred)} ')  
        print("")

        # Write to file results
        with open(key + '.txt', 'a') as f:
            f.write(f'Accuracy for {key} with {c} features:  {accuracy_score(y_test, y_pred)}\n')

print("Done.")