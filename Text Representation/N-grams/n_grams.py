# import libraries
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# ornek metin olustur
documents = [
    "Bu çalışma NGram çalışmasıdır.",
    "Bu çalışma doğal dil işleme çalışmasıdır."]


# unigram, bigram , trigram seklinde 3 farklı N değerine sahip gram modeli
vectorizer_unigram = CountVectorizer(ngram_range = (1,1)) 
# N-gram range'i seçmezsek default olarak 1-1 alır. Bunu CountVectorizer'ın kaynak kodlarında görebiliriz.
vectorizer_bigram = CountVectorizer(ngram_range = (2,2))
vectorizer_trigram = CountVectorizer(ngram_range = (3,3))

# unigram
X_unigram = vectorizer_unigram.fit_transform(documents)
unigram_features = vectorizer_unigram.get_feature_names_out()

# bigram
X_bigram = vectorizer_bigram.fit_transform(documents)
bigram_features = vectorizer_bigram.get_feature_names_out()

# trigram
X_trigram = vectorizer_trigram.fit_transform(documents)
trigram_features = vectorizer_trigram.get_feature_names_out()

# Sayısallaştırma unigram
unigram_array = X_unigram.toarray()
unigram_score = unigram_array.mean(axis = 0)
df_unigram_score = pd.DataFrame({"word": unigram_features, "unigram_score": unigram_score})

# Sayısallaştırma bigram
bigram_array = X_bigram.toarray()
bigram_score = bigram_array.mean(axis=0)
df_bigram_score = pd.DataFrame({"word": bigram_features, "bigram_score": bigram_score})

# Sayısallaştırma trigram
trigram_array = X_trigram.toarray()
trigram_score = trigram_array.mean(axis=0)
df_trigram_score = pd.DataFrame({"word": trigram_features, "trigram_score": trigram_score})


# sonucların incelenmesi
print(f"unigram features = {unigram_features}")
print(f"bigram features = {bigram_features}")
print(f"trigram features = {trigram_features}")