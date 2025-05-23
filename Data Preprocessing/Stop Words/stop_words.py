import nltk
from nltk.corpus import stopwords

nltk.download("stopwords") # farklı dillerde en cok kullanılan stop words iceren veri seti.

# İngilizce stop words analizi (nltk)

stop_words_eng = set(stopwords.words("english"))

text = "There are some examples of handling stop words from some texts."

text_list = text.split()
filtered_words = [word for word in text_list if word.lower() not in stop_words_eng]

print(f"Filtered words : {filtered_words}")

# Türkçe stop words analizi (nltk)

stop_words_tr = set(stopwords.words("turkish"))

metin = "merhaba arkadaşlar çok güzel bir ders oluyor ve bu ders faydalı mı ?"
metin_list = metin.split()

filtered_words_tr = [word for word in metin_list if word.lower() not in stop_words_tr]
print(f"Filrelenmiş türkçe metin : {filtered_words_tr}")

# Kütüphanesiz stop words cıkarımı

#stop words listeni oluştur.

tr_stop_words = ["için" , "bu", "ile" ,"mu", "mi", "özel"]

metin_2 = "Bu bir denemedir. Amacımız bu metinde bulunan özel karakterleri elemek mi acaba?"
metin_list_2 = metin_2.split()

filtered_words_tr_2 = [word for word in metin_list_2 if word.lower() not in tr_stop_words]

print(f"Özel filtreleme ile filtrelenmiş metin : {filtered_words_tr_2}")