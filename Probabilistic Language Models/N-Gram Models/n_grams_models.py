# import libraries
import nltk
from nltk.util import ngrams # n-gram oluşturmak için
from nltk.tokenize import word_tokenize # tokenizaton

from collections import Counter

# örnek veri seti oluştur
corpus = [
    "I love apple",
    "I love him",
    "I love NLP",
    "You love me",
    "He loves apple",
    "They love apple",
    "I love you and you love me"]

"""
Problem tanımı yapalım:
    dil modeli yapmak istiyoruz
    amac 1 kelimeden sonra gelecek kelimeyi tahmin etmek, metin türetmek, metin oluşturmak
    bunun icin n-gram dil modeli kullanacağız
    
    ex: I .... (love) .... (apple) parantez icindekiler boşluklara gelme olasılığı en yüksek olan kelimeler
"""

# verileri token haline getir
tokens = [word_tokenize(sentence.lower()) for sentence in corpus]

# bigram 2'li kelime grupları oluşturalım

bigrams = []
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))
    
bigrams_freq = Counter(bigrams)

# trigram
trigrams = []
for token_list in tokens:
    trigrams.extend(list(ngrams(token_list, 3)))
    
trigrams_freq = Counter(trigrams)

# model testing

# i love bigram'ından sonra "you" veya "apple" kelimelerinin gelme olasılıklarını hesaplayalım.

bigram = ("i","love") # hedef bigram

# i love 'you' olma olasılığı
prob_you = trigrams_freq[("i","love","you")] / bigrams_freq[bigram]

print(f"you kelimesinin olma olasılığı {prob_you}")

# i love 'apple' olma olasılığı
prob_apple = trigrams_freq[("i","love","apple")] / bigrams_freq[bigram]

print(f"apple kelimesinin olma olasılığı {prob_apple}")