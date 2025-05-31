# import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.corpus import stopwords

# veri seti yükle
df = pd.read_csv("spam.csv",encoding='latin1')

# @TODO veri önişleme ile verileri temizle
## boş sütunları silelim
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

## sütunlara anlamlı isimler verelim
df.columns = ['label', 'text'] 

## Boşlukları Temizleme & NaN Kontrolü
df['text'] = df['text'].astype(str)
df['text'] = df['text'].str.strip()
df.dropna(subset=['text'], inplace=True)

## Küçük harfe çevirme
df['text'] = df['text'].str.lower() 

## Noktalama işaretlerini kaldıralım
df['text'] = df['text'].str.translate(str.maketrans('', '', string.punctuation)) 

## Rakamları Silme
df['text'] = df['text'].str.replace(r'\d+', '', regex=True) 

## Stop words'leri kaldıralım
stop_words = set(stopwords.words('english')) 
df['text'] = df['text'].apply(
    lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words])
)

# tf-idf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df.text)

# kelime kümesini incele
feature_names = vectorizer.get_feature_names_out()
tfidf_score = X.mean(axis = 0).A1 # her kelimenin ortalama tf-idf değerleri.

# tf-idf skorlarını iceren bir df olustur
df_tfidf = pd.DataFrame({"word":feature_names, "tf-idf_score": tfidf_score})

# skorları sırala ve sonucları incele
df_tfidf_sorted =   df_tfidf.sort_values(by = "tf-idf_score", ascending = False)
print(df_tfidf_sorted.head(10))