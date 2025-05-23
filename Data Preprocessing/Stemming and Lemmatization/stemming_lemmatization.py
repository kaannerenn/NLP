import nltk

nltk.download("wordnet") # wordnet: lemmatization islemi icin gerekli veri tabani.

from nltk.stem import PorterStemmer # stemming icin fonksiyon

# porter stemmer nesnesini oluşturalım

stemmer = PorterStemmer()

words = ["running", "runner", "ran", "runs", "better","go","went"]

# Kelimelerin stemlerini (köklerini) buluyoruz.
# Bunu yaparken de porter stemmerin stem() fonksiyonunu kullanıyoruz.
stems = [stemmer.stem(w) for  w in words]
print(f"Stem: {stems}")

# Lemmatization

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

lemmas = [lemmatizer.lemmatize(w, pos = "v") for w in words] # pos = "v" verblerin anlamlı köklerini bulmak için.

print(f"Lemmas: {lemmas}")
