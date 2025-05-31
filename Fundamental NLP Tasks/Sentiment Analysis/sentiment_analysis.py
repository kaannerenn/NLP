"""
definition of the problem and dataset:
    https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv
    
    amazon veri seti içerisinde bulunan yorumların pozitif mi yoksa negatif mi olduğunu sınıflandırmak
    binary classification problem
    """
# import libraries
import pandas as pd
import nltk 

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize # nltk kütüphanemdeki sorunu halledemediğim icin gecici basic bir tokenize işlemi yapıcam.
from nltk.stem import WordNetLemmatizer
import re # basit tokenizer için

nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

# upload dataset
df = pd.read_csv("amazon.csv")

# text cleaning and preprocessing
# Basit tokenizer fonksiyonu
def simple_tokenize(text):
    # Küçük harfe çevir ve sadece kelimeleri (harf ve rakam) seç
    return re.findall(r'\b\w+\b', text.lower())

lemmatizer = WordNetLemmatizer()

def clean_preprocess_data(text):
    
    # tokenize
    tokens = simple_tokenize(text)
    
    # stopwords
    filtered_tokens = [token for token in tokens if token not in stopwords.words("english")]
    
    # lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # join words
    processed_text = " ".join(lemmatized_tokens)
    
    return processed_text

df["reviewText2"] = df["reviewText"].apply(clean_preprocess_data)

# sentiment analysis (nltk)
analyzer = SentimentIntensityAnalyzer()

def get_sentiments(text):
    
    score = analyzer.polarity_scores(text)
    
    sentiment = 1 if score["pos"] > 0 else 0
    
    return sentiment

df["sentiment"] = df["reviewText2"].apply(get_sentiments)

# evaluation - test
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(df["Positive"], df["sentiment"])
print(f"Confusion matrix: {cm}")
"""
Confusion matrix: [[ 1063  3704]
                   [  536 14697]]
"""

cr = classification_report(df["Positive"], df["sentiment"])
print(f"Classification report: \n{cr}")

"""
Classification report: 
              precision    recall  f1-score   support

           0       0.66      0.22      0.33      4767
           1       0.80      0.96      0.87     15233

    accuracy                           0.79     20000
   macro avg       0.73      0.59      0.60     20000
weighted avg       0.77      0.79      0.75     20000

"""

# Göründüğü üzere gerçekte 0 olan yorumları yani negatif olanları tahmin etme score'umuz kötü.
# Fakat gerçekte 1 olan yorumları yani pozitif olanları tahmin etme score'umuz iyi.