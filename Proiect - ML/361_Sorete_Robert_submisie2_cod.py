import numpy as np
import pandas as pd

# Citirea datelor
train_data_df= pd.read_csv('train_data.csv')
test_data_df = pd.read_csv('test_data.csv')

# Se elimina coloanele care nu sunt relevante
X = train_data_df['text']
X_TEST = test_data_df['text']
y = train_data_df['label']

# Atribuirea fiecarei clase un id
label2id = {'Ireland': 0, 'England': 1,  'Scotland': 2}
id2label = {0: 'Ireland', 1: 'England',  2: 'Scotland'}

y = [label2id[label] for label in y]


# Preprocesarea Datelor
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from simplemma import lang_detector, lemmatize  # pip install simplemma

nltk.download('stopwords')  # Descarcam stopwords

german_stopwords = stopwords.words('german')
italian_stopwords = stopwords.words('italian')
spanish_stopwords = stopwords.words('spanish')
dutch_stopwords = stopwords.words('dutch')
danish_stopwords = stopwords.words('danish')

# Constuirea dictionarului de stopwords
stop_words = {'de': german_stopwords, 'it': italian_stopwords, 'es': spanish_stopwords, 'nl': dutch_stopwords, 'da': danish_stopwords}

# Acest dictionar este folosit pentru stemmatizarea textului, deoarece folosesc o librarie diferita de lemmatizare
lang2id = { 'de': 'german', 'it': 'italian', 'es': 'spanish', 'nl': 'dutch', 'da': 'danish', 'unk': 'german'}     

# Functia de preprocesare
def preprocesare_text(text, stop_words=True, digits=False, stem=False, lemma=True):

    # Stergem cifrele
    if (digits):    
        text = re.sub(r'\d+', '', text) 

    # Extragem cuvintele din text sub forma de tokeni, eliminand caracterele speciale
    text = re.findall(r'\w+', text) 

    # Eliminam cuvintele de dimensiune 1
    text = [word for word in text if len(word) > 1]

    if(stop_words or stem or lemma):
        # Determinam limba textului
        language = lang_detector(text, lang=('de', 'it', 'es', 'nl', 'da'))
        language = language[0][0]

        if(stop_words):
            text = [word for word in text if word not in stop_words[language]]

        if(lemma):
            text = [lemmatize(word, language) for word in text]
        elif(stem):
            stemmer = nltk.stem.SnowballStemmer(lang2id[language])
            text = [stemmer.stem(word) for word in text]

    text = ' '.join(text)
    text = text.lower() 
    # Lematizarea nu reuseste in germana pentru substantive lowercase, 
    # de aceea lower la final

    return text


X_stop_lemma = [preprocesare_text(text) for text in X]              # Preprocesarea datelor de antrenare 
X_TEST_stop_lemma = [preprocesare_text(text) for text in X_TEST]    # Preprocesarea datelor de test


# Impartirea datelor in train si validation
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_stop_lemma, y, test_size=0.2, random_state=42)


# Vectorizarea datelor
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(strip_accents='unicode', sublinear_tf=True, min_df=1 ,max_features=100000)
X_train = vectorizer.fit_transform(X_train)
X_valid = vectorizer.transform(X_valid)
X_TEST = vectorizer.transform(X_TEST_stop_lemma)


# Antrenarea modelului folosind Logistic Regression
from sklearn.svm import LinearSVC

clf = LinearSVC(C=1.1, class_weight= {0: 0.63, 1: 0.33, 2: 0.63}).fit(X_train, y)

# predict
y_pred = clf.predict(X_TEST)

from sklearn.metrics import accuracy_score

print(f'Accuracy of LinearSVC with C=2: {accuracy_score(y_valid, y_pred)}') 
# 0.7235987490979071


# 5 Fold Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
accuracies = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = cv)

print(f'Accuracy using 5 Fold Cross-Validation: {accuracies.mean()*100:.2f} %')
print(f'Standard Deviation: {accuracies .std()*100:.2f} %')
print(f'Accuracies: {accuracies}')



# Matricea de confuzie
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_valid, y_pred)
df_cm = pd.DataFrame(cm, index = [i for i in label2id.keys()],
columns = [i for i in label2id.keys()])


plt.figure(figsize=(5,5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# Raportul de clasificare
from sklearn.metrics import classification_report

print(classification_report(y_valid, y_pred, target_names=label2id.keys()))



# Predictia datelor de test
y_pred = clf.predict(X_TEST)

# Salvarea datelor in fisierul de output
prediction = [id2label[label] for label in y_pred]
submission = pd.DataFrame({'id':range(1, len(prediction) + 1),'label': prediction})
submission.to_csv('submission.csv', index=False)

