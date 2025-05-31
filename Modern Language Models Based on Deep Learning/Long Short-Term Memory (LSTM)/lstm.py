"""
metin üretimi

lstm train with text data

text data = gpt ile oluşturucaz.
"""


# import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# eğitim verisi oluştur with chatgpt
texts = [
    "bugün hava çok güzel, dışarıda yürüyüş yapmayı düşünüyorum.",
    "kitap okumak beni gerçekten mutlu ediyor.",
    "yeni bir film izledim, çok eğlenceliydi.",
    "çalışma odamda huzur içinde çalışıyorum.",
    "güneş batarken sahilde yürümek çok keyifli.",
    "yaz tatili için plan yapmaya başladım.",
    "arkadaşımın doğum günü için hediye aldım.",
    "bu sabah kahvemi içip dergilere göz attım.",
    "spor salonuna yeni üyelik aldım, düzenli gitmeye karar verdim.",
    "akşam yemeği için makarna yapmayı planlıyorum.",
    "bugün çok yoğun bir iş günüydü, akşam dinlenmek istiyorum.",
    "güzel bir kahvaltı yaptıktan sonra parka gittim.",
    "hafta sonu için bir kaç gün tatil planım var.",
    "yeni telefon almak için araştırmalar yapıyorum.",
    "dün akşam arkadaşlarla sinemaya gittik, çok eğlendik.",
    "haftanın her günü spor yapmaya karar verdim.",
    "yaz tatili için yurtdışına seyahat etmeyi planlıyorum.",
    "bugün bol bol kitap okumayı düşünüyorum.",
    "dışarıda yağmur yağıyor, içeriye çekilip film izleyebilirim.",
    "gün sonunda güzel bir yürüyüş yapmayı seviyorum.",
    "bugün iş yerinde harika bir sunum yaptım.",
    "arkadaşım yeni bir iş buldu, onunla kutlama yapmalıyız.",
    "gün boyu dışarıda olduğum için çok yoruldum.",
    "akşam yemeğini evde yapmak istiyorum, sağlıklı bir tarif arıyorum.",
    "bugün öğle arasında bir kafede vakit geçirdim.",
    "yeni bir dil öğrenmeye karar verdim.",
    "kışın kar tatili yapmak harika bir fikir.",
    "dün akşam misafirimiz vardı, birlikte çok keyifli bir akşam yemeği yedik.",
    "bugün sabah erken kalkıp yürüyüş yaptım, çok dinç hissediyorum.",
    "yeni bir kitap alıp okumaya başladım.",
    "gece geç saatlere kadar çalıştım, şimdi dinlenmeye ihtiyacım var.",
    "yazın sahil kenarında vakit geçirmek harika olur.",
    "sabahları mutlaka kahvemi içmeden güne başlamam.",
    "bugün kendime vakit ayırıp birkaç saat meditasyon yaptım.",
    "yaz tatili için gezilecek yerler araştırıyorum.",
    "bugün kendime küçük bir tatil günü yaptım.",
    "yeni projeler üzerinde çalışmaya başladım.",
    "bugün öğleden sonra bir kafede arkadaşlarımla buluşmayı planlıyorum.",
    "spor salonuna katıldım, düzenli olarak gitmeye karar verdim.",
    "bugün uzun bir yürüyüş yaptım, doğa harikaydı.",
    "ailemle birlikte kahvaltı yapmak çok güzel bir deneyimdi.",
    "öğle arası dışarıda biraz vakit geçirmeyi düşünüyorum.",
    "gün sonu yürüyüş yapmayı çok seviyorum.",
    "bugün yoga dersine katıldım, çok rahatladım.",
    "arkadaşımın yeni evine taşındım, çok heyecanlıyım.",
    "bugün çok güzel bir gün batımı izledim.",
    "dışarıda çok güzel bir manzara var, fotoğraf çekmeyi düşünüyorum.",
    "şu sıralar dışarıda yürüyüş yapmayı çok seviyorum.",
    "bugün dışarıda vakit geçirmenin keyfini çıkaracağım.",
    "yeni bir restoranda akşam yemeği yemeyi planlıyorum.",
    "yaz tatili için yazlık ev arayışım başladı.",
    "bugün çok keyifli bir kahve içtim.",
    "iş sonrası arkadaşlarımla dışarı çıkmayı düşünüyorum.",
    "sabah erken uyanıp deniz kenarında yürüyüş yapacağım.",
    "hafta sonu yeni bir aktivite yapmak istiyorum.",
    "akşam yemeğini dışarıda yemeyi düşünüyorum.",
    "gün boyu dışarıda olmak çok keyifliydi.",
    "yeni bir hobi edinmek istiyorum.",
    "hafta sonu için arkadaşlarımla tatil planı yapıyorum.",
    "bugün çok yoğun bir gündü, artık dinlenmek istiyorum.",
    "öğle yemeğimi dışarıda yemeyi düşünüyorum.",
    "yeni bir dil öğrenme kararı aldım, çalışmalara başladım.",
    "sabahları kahve içmeden güne başlamam.",
    "akşam yürüyüşüne çıkmayı düşünüyorum.",
    "bugün çok keyifli bir alışveriş yaptım.",
    "yaz tatilinde gezilecek yerler hakkında araştırmalar yapıyorum.",
    "bugün çok verimli bir gün geçirdim.",
    "yaz tatili için uçak biletimi aldım, heyecanlıyım.",
    "bugün iş dışında kendime vakit ayırdım.",
    "dışarıda çok güzel bir hava var, bir süre dışarıda vakit geçireceğim.",
    "gece geç saatlerde çalışmak yerine daha erken yatmayı planlıyorum.",
    "yeni bir hobi edinmek istiyorum.",
    "hafta sonu tatili için plan yapmaya başladım.",
    "bu akşam sinemaya gitmeyi planlıyorum.",
    "bugün iş yerinde çok yoğun bir gün geçirdim.",
    "bugün sabah koşuya çıktım, çok iyi hissettirdi.",
    "akşam yemeği için bir arkadaşımın evine gideceğim.",
    "spor salonunda çok verimli bir antrenman yaptım.",
    "bugün dışarıda arkadaşlarımla vakit geçirdim.",
    "bu hafta sonu dinlenmeye karar verdim.",
    "bugün çok güzel bir kitap okudum.",
    "bugün erken uyudum, dinç kalktım.",
    "yaz tatili için hazırlıklara başladım.",
    "bugün dışarıda çok keyifli bir yürüyüş yaptım.",
    "gün sonunda mutlaka bir yürüyüş yapmayı severim.",
    "bugün çok güzel bir sabah kahvaltısı yaptım.",
    "bugün iş yerinde çok verimli çalıştım.",
    "bugün kendime vakit ayırıp sevdiğim şeyleri yaptım.",
    "spor salonunda çok iyi bir antrenman yapmayı düşünüyorum.",
    "yaz tatili için bir kaç farklı seçenek araştırıyorum.",
    "bugün öğleden sonra bir kafede arkadaşlarımla buluşmayı planlıyorum.",
    "akşam yemeği için dışarıda yemek yiyeceğim.",
    "bugün güzel bir kitap aldım, okumaya başladım.",
    "dışarıda çok güzel bir hava var, bir süre parkta vakit geçirmeyi düşünüyorum.",
    "bugün işyerinde çok yoğun bir gündü.",
    "bu akşam sinemaya gitmeyi düşünüyorum.",
    "yaz tatilinde gitmek istediğim yerleri araştırmaya başladım.",
    "bugün dışarıda bir süre vakit geçireceğim.",
    "bugün bol bol kitap okumayı düşünüyorum.",
    "bugün öğle yemeğimi dışarıda yemeyi planlıyorum.",
    "akşam yemeği için sevdiğim bir restorana gitmeyi düşünüyorum.",
    "spor salonuna üyelik aldım, düzenli gitmeye karar verdim.",
    "bugün öğle arasında dışarıda vakit geçireceğim.",
    "gün boyunca çok verimli çalıştım, şimdi dinlenmeye çekileceğim.",
    "akşam yemeği için bir arkadaşımın evine gideceğim.",
    "dışarıda güzel bir hava var, yürüyüş yapmayı düşünüyorum.",
    "bugün iş yerinde yoğun bir gündü.",
    "yaz tatili için yeni yerler araştırıyorum.",
    "dışarıda çok güzel bir gün, parkta vakit geçirmeyi planlıyorum."
]


## metin temizleme ve preprocessing : tokenization, padding, label encoding

# tokenization

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts) # metinler üzerindeki kelime frekansını öğren
total_words = len(tokenizer.word_index) + 1 # toplam kelime sayısı

# n-gram dizileri oluştur ve padding uygula
input_sequences = []
for text in texts:
    # metinleri kelime indekslerine çevir
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # her bir metin için n-gram dizisi oluşturalım
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
        
# en uzun diziyi bulup sonrasında tüm dizileri aynı uzunluga getirelim
max_sequence_length = max(len(x) for x in input_sequences)

# dizilere padding işlemi uygulayalım böylelikle aynı uzunlukta olmasını saglarız
input_sequences = pad_sequences(input_sequences, maxlen = max_sequence_length, padding = "pre")

# X(girdi) ve y(hedef değişken)
X, y = input_sequences[:,:-1] , input_sequences[:,-1]

y = tf.keras.utils.to_categorical(y, num_classes = total_words) # one hot encoding
    
## lstm modeli oluştur: compile, train ve evaluate

model = Sequential()

# embedding
model.add(Embedding(total_words, 50, input_length = X.shape[1]))

# LSTM
model.add(LSTM(100, return_sequences = False))

# output
model.add(Dense(total_words, activation = "softmax"))

# model compile
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

#model training
model.fit(X,y, epochs = 100, verbose = 1)

## model prediction 
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        # girdi metnini sayısal verilere dönüştür
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # padding
        token_list = pad_sequences([token_list], maxlen = max_sequence_length -1, padding = "pre")
        
        # prediction
        predicted_probabilities = model.predict(token_list, verbose = 0)
        
        # en yüksek olasılıga sahip kelimenin indexini bul
        predicted_word_index = np.argmax(predicted_probabilities, axis = -1)
        
        # tokenizer ile kelime index'inden asıl kelimeyi bul
        predicted_word = tokenizer.index_word[predicted_word_index[0]]
        
        # tahmin edilen kelimeyi seed text'e ekleyelim
        seed_text = seed_text + " " + predicted_word
        
    return seed_text

seed_text = "Kitap yazmak"
print(generate_text(seed_text,1))

# @TODO veri setini çoğalt.  Cümle bitip yeni cümle başlatan bir algoritma yaz.
