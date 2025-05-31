# count vectorizer içeriye aktar
from sklearn.feature_extraction.text import CountVectorizer

# veri seti olustur
documents = [
    "kedi evde",
    "kedi bahçede"]
# vectorizer tanımla 
vectorizer = CountVectorizer()


# metni sayısal vektörlere çevir
X = vectorizer.fit_transform(documents)

# kelime kümesi oluşturma
feature_names = vectorizer.get_feature_names_out() # kelime kümesini oluşturma
print(f"Kelime kümesi = {feature_names}")

# vektör temsili
vector_represantation = X.toarray()

print(f"Vektör temsili = {vector_represantation}")

# Output
"""
Kelime kümesi = ['bahçede' 'evde' 'kedi']
Vektör temsili = [[0 1 1]
 [1 0 1]]
"""