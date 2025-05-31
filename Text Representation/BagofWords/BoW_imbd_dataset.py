# import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# veri setinin iceriye aktarılması
df = pd.read_csv("IMDB Dataset.csv")

# metin verilerini alalım
documents = df["review"]
labels = df["sentiment"] # positive veya negative

# Stop words temizliği
stop_words = set(stopwords.words('english'))

# metin temizleme
def clean_text(text):
    # buyuk kucuk harf cevirme
    text = text.lower()
    
    # rakamları temizleme
    text = re.sub(r"\d+", "", text)
    
    # özel karakterlerin kaldırılması
    text = re.sub(r"[^\w\s]", "", text)
    
    # kısa kelimelerin temizlenmesi
    words = [word for word in text.split() if len(word) > 2 and word not in stop_words]
    
    return " ".join(words) # temizlenmiş texti return ediceğiz

# metinleri temizle
cleaned_doc = [clean_text(row) for row in documents]

### bag of words kısmı

# vectorizer tanımla
vectorizer = CountVectorizer()

# metin -> sayisal hale getir
X = vectorizer.fit_transform(cleaned_doc[:75]) #uzun sürmemesi adına 75 tanesini aldık.

# kelime kümesi göster
feature_names = vectorizer.get_feature_names_out()

# vektor temsili göster
vector_representation = X.toarray()
print(f"Vektör temsili: {vector_representation}")

df_bow = pd.DataFrame(vector_representation, columns = feature_names)

# kelime frekanslarını göster
word_counts = X.sum(axis = 0).A1
word_frequency = dict(zip(feature_names, word_counts))

# ilk 5 kelime

most_common_5_words = Counter(word_frequency).most_common(6)

print(f"Most common 5 words = {most_common_5_words}")

# @TODO Stop wordleri çıkarıp programı yeniden düzenle....