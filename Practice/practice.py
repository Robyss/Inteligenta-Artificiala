# Preprocesarea datelor

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')


# Vectorizarea datelor

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_data_df['text'])

print(X.shape)

# Antrenarea modelului

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))

# Testarea modelului

test_data_df = pd.read_csv(os.path.join(data_path, 'test_data.csv'))

test_data_df['text'] = test_data_df['text'].apply(preprocesare_text)

X_test = vectorizer.transform(test_data_df['text'])

y_pred = model.predict(X_test)

print(y_pred)



stop_words = set(stopwords.words('italian'))

def preprocesare_text(text):

    # Tokenizare si eliminarea caracterelor speciale
    cuvinte = re.findall(r'\w+', text)

    # Eliminarea cuvintelor de stop
    cuvinte = [cuvant for cuvant in cuvinte if cuvant not in stop_words]

    # Lemmatizarea cuvintelor
    lemmatizer = WordNetLemmatizer()
    cuvinte = [lemmatizer.lemmatize(cuvant) for cuvant in cuvinte]

    return cuvinte

# train_data_df['text'] = train_data_df['text'].apply(preprocesare_text)

exemplu_italian = train_data_df[train_data_df['language'] == 'italiano']
text_italian = exemplu_italian['text'].iloc[0]
print(preprocesare_text(text_italian)[:20])






