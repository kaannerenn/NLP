"""
spam veri seti -> spam ve ham -> binary classification with Decision Tree
"""

# import libraries
import pandas as pd

# veri seti yükle
data = pd.read_csv("text_classification_spam_dataset.csv", encoding = "latin-1")
data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1, inplace = True)
data.columns = ["label","text"]

# EDA: Keşifsel veri analizi: missing value
print(data.isna().sum()) # Kayıp veri tespiti için kullanırız. Kayıp veri olmaması için 0 değerini görmeliyiz. Kayıp verimiz yok.


## Text cleaning and preprocessing: ozel karakterler, lowercase, tokenization ,stopwords, lemmatize
import nltk

nltk.download("stopwords") # cok kullanılan ve anlam tasımayan stop wordsler için
nltk.download("wordnet") # lemma bulmak için gerekli olan veri seti
nltk.download("omw-1.4") # wordnete ait farklı dillerin kelime anlamlarını içeren veri seti

import re
from nltk.corpus import stopwords # # stopwords'lerden kurtulmak için
from nltk.stem import WordNetLemmatizer # Lemmatization

text = list(data.text)
lemmatizer = WordNetLemmatizer()

corpus = []

for i in range(len(text)):
    
    r = re.sub("[^a-zA-Z]", " ", text[i]) # metin icerisinde harf olmayan tüm karakterlerden kurtuluyoruz.
    
    r = r.lower() # kücük harfe çevirdik
    
    r = r.split() # kelimeleri ayır
    
    r = [word for word in r if word not in stopwords.words("english")] # stopwords'lerden kurtul
    
    r = [lemmatizer.lemmatize(word) for word in r]
    
    r = " ".join(r)
    
    corpus.append(r)

data["text2"] = corpus

## Model training and evaluation

X = data["text2"]
y = data["label"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# feature extraction with bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)

# classifier training: model training and evaluation
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train_cv, y_train) # egitim

X_test_cv = cv.transform(X_test)

# prediction
prediction = dt.predict(X_test_cv)

# karşılaştırma
from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y_test, prediction)

# accuracy

accuracy = 100*(sum(sum(c_matrix)) - c_matrix[1,0] - c_matrix[0,1]) / sum(sum(c_matrix))

print(f"Accuracy: {accuracy}")

# Aslında overfitting var, yani modeli ezberlemiş fakat konuyu anlamak için güzel bir örnek.
train_pred = dt.predict(X_train_cv)
from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train, train_pred)
print(f"Train Accuracy: {train_acc}")

print(f"Tree Depth: {dt.get_depth()}")
