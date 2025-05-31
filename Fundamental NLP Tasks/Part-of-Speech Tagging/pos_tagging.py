import spacy

nlp = spacy.load("en_core_web_sm")

sentence1 = "What is the weather like today or tomorrow"
doc1 = nlp(sentence1)

sentence2 = "What are you gonna do right now"
doc2 = nlp(sentence2)

for token in doc1:
    print(token.text, token.pos_)
    
"""
OUTPUT: 
    What PRON 
    is AUX
    the DET
    weather NOUN
    like ADP
    today NOUN
    or CCONJ
    tomorrow
"""
print()
for token in doc2:
    print(token.text, token.pos_)
"""
OUTPUT 2:
    What PRON
    are AUX
    you PRON
    gon VERB
    na PART
    do VERB
    right ADV
    now ADV
"""