"""
Solve Classification problem (sentiment analysis in NLP) with RNN (Deep Learning based Language Model)

sentiment analysis -> bir cümlenin etiketlenmesi (positive and negative)

restauran yorumları değerlendirme 
"""

# import libraries
import numpy as np
import pandas as pd

from gensim.models import Word2Vec # metin temsili

from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# @TODO data boyutunu artır.
# create dataset
data = {
    "text": [
        "Yemekler harikaydı", "Servis çok yavaştı", "Tatlılar mükemmeldi", "Çorba soğuktu",
        "Garsonlar çok kibardı", "Garson bizimle hiç ilgilenmedi", "Fiyat performans oranı çok iyiydi",
        "Porsiyonlar çok küçüktü", "Atmosfer çok keyifliydi", "Müzik çok gürültülüydü",
        "Sunum çok şıktı", "Yemeklerde tuz oranı berbattı", "Tatlılar taptazeydi",
        "Salata bayattı", "İkramlar çok hoştu", "Çatal bıçaklar kirliydi", 
        "Rezervasyonumuzu unuttular", "Masamıza hemen oturduk", "Kahve nefisti", 
        "Sandalyeler çok rahatsızdı", "Personel çok güler yüzlüydü", "Yemek beklediğimden kötüydü",
        "Tatlılar beklentimin üzerindeydi", "Hizmet kalitesi çok düşüktü", "Mekanın dekorasyonu çok zevkliydi",
        "Yemek buz gibi geldi", "Garson siparişi yanlış getirdi", "Servis oldukça hızlıydı",
        "İçecekler sıcak geldi", "Garson siparişi hızlıca aldı", "Limonata ferahlatıcıydı",
        "Hesap çok pahalıydı", "Bize yanlış yemek getirildi", "Menüdeki çeşitlilik çok iyiydi",
        "Garsonun tavsiyesi mükemmeldi", "Çorba yağlıydı", "Tatlılar bayattı",
        "Masamızı çok geç temizlediler", "Çalışanlar çok ilgiliydi", "Masa düzeni çok iyiydi",
        "İçecekler çok geç geldi", "İç mekan çok ferah", "Dışarıda sigara içilmesi rahatsız edici",
        "Garsonlar çok samimiydi", "Yemekler midemi bozdu", "Menüde vejetaryen seçenek çoktu",
        "Servis mükemmeldi", "Tavuk çiğdi", "Pizzaları çok başarılı", "Hamburgerin eti çiğ kalmıştı"
    ],
    "label": [
        "positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive", "negative", "negative", "positive", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive", "negative", "negative", "positive", "positive", "positive",
        "negative", "negative", "positive", "positive", "negative", "negative", "negative", "positive", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative"
    ]
}


df = pd.DataFrame(data)

## metin temizleme ve preprocessing : tokenization , padding, label encoding, train test split

# tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
word_index = tokenizer.word_index

# padding process
maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen = maxlen)
print(X.shape)

# label encoding

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])

# train ve test split

X_train ,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

## metin temsili: word embedding -> Word2Vec

sentences = [text.split() for text in df["text"]]
word2vec_model = Word2Vec(sentences, vector_size = 50, window = 5, min_count = 1) 

embedding_dim = 50
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

## modelling : build , train and test -> rnn model

#build model
model = Sequential()

# embedding
model.add(Embedding(input_dim = len(word_index)+ 1, output_dim = embedding_dim, weights = [embedding_matrix], input_length = maxlen, trainable = False))

# RNN layer
model.add(SimpleRNN(50, return_sequences = False))

# output layer
model.add(Dense(1, activation = "sigmoid"))

# compile model

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# train model

model.fit(X_train, y_train, epochs = 10, batch_size = 2, validation_data = (X_test,y_test))

# evaluate RNN model

test_loss, test_accuracy = model.evaluate(X_test,y_test)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

# cumle sınıflandırma çalışması

def classify_sentence(sentence):
    
    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq, maxlen = maxlen)
    
    prediction = model.predict(padded_seq)
    
    predicted_class = (prediction > 0.5).astype(int)
    label = "positive" if predicted_class[0][0] == 1 else "negative"
    
    return label

sentence = "Restaurant berbattı"

result = classify_sentence(sentence)
print(f"Result: {result}")