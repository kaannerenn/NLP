comments = [
    "Bu ürün HARİKA! Tavsiye ederim 😍",
    "berbat, iade etmek istiyorum.",
    "Ne güzel paketlenmişti ama ürün kötü çıktı.",
    "KESİNLİKLE bir daha alışveriş yapmam.",
    "hızlı kargo teşekkürler 🙏"
]
# Kelimeleri küçük harfe çevirme
comments_lower = [sentence.lower() for sentence in comments]


# Noktalama işaretlerini silme

import string

comments_without_punctuation = [sentence.translate(str.maketrans("","",string.punctuation)) for sentence in comments_lower]


# Özel karakterleri silme

import re

comments_without_special_ch = [re.sub(r"[^A-Za-z0-9şŞıİçÇöÖüÜğĞ\s]","",sentence) for sentence in comments_without_punctuation]


# Kelimeleri tokenlarına ayırma
import nltk

nltk.download("punkt")

word_tokens = [nltk.word_tokenize(sentence) for sentence in comments_without_special_ch]


# Stop words'leri temizleme

import nltk
from nltk.corpus import stopwords

stop_words_tr = set(stopwords.words("turkish"))

comments_without_stop_words = [
    [word for word in sentence if word not in stop_words_tr]
    for sentence in word_tokens
]

# Stemming ve Lemmatization

import snowballstemmer


stemmer = snowballstemmer.stemmer('turkish')

stems = [
    [stemmer.stemWord(word) for word in sentence]
    for sentence in comments_without_stop_words
]

def clean_text(word):
    manual_lemma = {
        "ür": "ürün",
        "tavsi": "tavsiye",
        "ia": "iade",
        "harik": "harika",
        "iste": "istemek",
        "paketlenmiş": "paket",
        "kesinlikl": "kesinlikle",
        "alışveriş": "alışveriş",
        "yapma": "yapmak",
        "teşekkür": "teşekkür"
    }
    return manual_lemma.get(word, word)  # kelime sözlükte varsa karşılığını, yoksa kendisini döndür


stemmed_fixed = [
    [clean_text(word) for word in sentence]
    for sentence in stems
]




