## 1. kısım
# import libraries
import os
os.environ["OMP_NUM_THREADS"] = "2" # memory leak riskini azaltmak için

import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from nltk.corpus import stopwords

# veri setini yükleme
df = pd.read_csv("IMDB Dataset.csv")

documents = df["review"]

# metin temizleme
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower() # kücük harf
    text = re.sub(r"\d+", "", text) # sayiları temizle
    text = re.sub(r"[^\w\s]", "", text) # özel karakterleri temizle
    text = " ".join([
        word for word in text.split()
        if len(word) > 2 and word not in stop_words
    ])
    
    return text
    # TODO stop words'lerin cıkarılmasını ekle
    
cleaned_documents = [clean_text(document) for document in documents]


# metin tokenization
tokenized_documents = [simple_preprocess(doc) for doc in cleaned_documents] 
# simple_preprocess işlemini istersek clean_text fonkisyonu içerisinde uygulayabilirdik.
# Fakat metin önişleme ve tokenizasyon kısmı ayrı olsun diye tercih etmiyoruz.

## 2. kısım
# Word2Vec modeli tanımla
word2_vec_model = Word2Vec(sentences = tokenized_documents, vector_size=50, window = 5, min_count=1, sg=0)
word_vectors = word2_vec_model.wv

words = list(word_vectors.index_to_key)[:500] # 500 ü kaldırıp tüm veriseti için dene

vectors = [word_vectors[word] for word in words]

# clustering KMeans K = 2
kmeans = KMeans(n_clusters=2) # Öklid uzaklığını kullanan algoritma
kmeans.fit(vectors)
clusters = kmeans.labels_ # 0 , 1 şeklinde 2 etiket alır

# PCA 50 -> 2
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# 2 boyutlu bir görselleştirme
plt.figure()
plt.scatter(reduced_vectors[:,0], reduced_vectors[:,1], c = clusters, cmap = "viridis")

centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0], centers[:,1], c = "red", marker = "x", s = 150, label = "Center")
plt.legend()

# figure üzerine kelimelerin eklenmesi
for i, word in enumerate(words):
    plt.text(reduced_vectors[i,0], reduced_vectors[i,1], word, fontsize = 7)
    
plt.title("Word2Vec")