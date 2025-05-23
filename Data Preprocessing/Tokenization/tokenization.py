import nltk # natural language toolkit

nltk.download("punkt") # metni kelime ve cümle bazında

text = "Hello, World! How are you? Hello, hi..."

# kelime tokenizatonu: word_tokenize: metni kelimelere ayırır. 
# Noktalama işaretleri ve boşluklar ayrı birer token olarak elde edilecektir.
word_tokens = nltk.word_tokenize(text)
print(word_tokens)

# cumle tokenizasyonu: sent_tokenize: metni cümlelere ayırır. her cümle birer token olur.
sentence_tokens = nltk.sent_tokenize(text)
print(sentence_tokens)


