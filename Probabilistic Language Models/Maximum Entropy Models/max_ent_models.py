"""
classification problem : duygu analizi (sentiment analysis) -> olumlu veya olumsuz olarak sınıflandırma
"""
# import libraries
from nltk.classify import MaxentClassifier

# veri seti tanımlama
train_data = [
    ({"love":True,"amazing":True,"happy":True, "terrible":False},"positive"),
    ({"hate":True, "terrible":True},"negative"),
    ({"joy":True,"happy":True,"hate":False},"positive"),
    ({"sad":True,"depressed" :True, "love":False},"negative")
    ]

# train maximum entropy classifier
classifier = MaxentClassifier.train(train_data, max_iter = 10)

# yeni cümle ile test
test_sentence = "I hate this movie and it was terrible"
features = {word: (word in test_sentence.lower().split()) for word in ["love","amazing","terrible","joy","happy","sad","depressed","hate"]}

label = classifier.classify(features)
print(f"Result: {label}")