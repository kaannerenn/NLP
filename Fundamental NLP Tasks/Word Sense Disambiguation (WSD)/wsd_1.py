import nltk
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

# Yolu manuel olarak tanımla
nltk.data.path.clear()
nltk.data.path.append("C:/Users/kaann/AppData/Roaming/nltk_data")

# gerekli nltk paketlerini indirelim.
nltk.download("wordnet")
nltk.download("own-1.4")
nltk.download("punkt")

# first sentence
sentence_1 = " I go to the bank to deposit money"
word_1 = "bank"

sense1 = lesk(nltk.word_tokenize(sentence_1),word_1)

print(f"İlk cumle: {sentence_1}")
print(f"Kelime: {word_1}")
print(f"Anlam: {sense1.definition()}")

sentence_2 = "The river bank is flooded after the heavy rain"
word_2 = "bank"
sense2 = lesk(nltk.word_tokenize(sentence_2), word_2)

print(f"İkinci cumle: {sentence_2}")
print(f"Kelime: {word_2}")
print(f"Anlam: {sense2.definition()}")