# import libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# örnek belge oluştur
documents = [
    "Köpek çok tatlı bir hayvandır.",
    "Köpek ve kuşlar çok tatlı hayvanlardır",
    "Inekler süt üretirler."]

# vectorizer tanımla
tfidf_vectorizer = TfidfVectorizer()

# metinleri sayisal hale cevir (vectorizer kullanarak)
X = tfidf_vectorizer.fit_transform(documents)

# kelime kümesini inceleyelim
feature_names = tfidf_vectorizer.get_feature_names_out()

# vektör temsilini incele
vector_representation = X.toarray()
print(f"tf-idf: {vector_representation}")

df_tfidf = pd.DataFrame(vector_representation, columns = feature_names)

# ortalama tf-idf değerlerine bakalım

tf_idf = df_tfidf.mean(axis = 0)