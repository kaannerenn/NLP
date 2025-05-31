"""
Part of Speech: kelimelerin uygun sözcük türünü bulma çalışması
HMM

I(Pronoun) am a teacher(Noun)

"""

# import libraries
import nltk
from nltk.tag import hmm


# ornek training data tanımla
train_data = [
    [("I","PRP"),("am", "VBP"),("a", "DT"),("teacher","NN")],
    [("You", "PRP"),("are", "VBP"),("a", "DT"),("student", "NN"),],
    ]

# train HMM
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)



# yeni bir cümle oluştur ve cümlenin içerisinde bulunan her bir sözcüğün türünü etiketle

test_sentence = "I am a student".split()

tags = hmm_tagger.tag(test_sentence)

print(f"Yeni cümle: {tags}")

"""
Output: Yeni cümle: [('I', 'PRP'), ('am', 'VBP'), ('a', 'DT'), ('student', 'NN')]
"""

test_sentence2 = "He is a driver".split()

tags2 = hmm_tagger.tag(test_sentence2)

print(f"Yeni cümle 2: {tags2}")
"""
Output2: Yeni cümle 2: [('He', 'PRP'), ('is', 'PRP'), ('a', 'PRP'), ('driver', 'PRP')]
Veri eksikliğinden yanlış sonuçlar aldık.
"""