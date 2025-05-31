import spacy

nlp = spacy.load("en_core_web_sm")

# incelenecek olan kelimeler

word = "I go to school 123"

# kelimeyi nlp isleminden geçir
doc = nlp(word)

for token in doc:
    
    print(f"Text: {token.text}") # kelimenin kendisi
    print(f"Lemma: {token.lemma_}") # kelimenin kök hali
    print(f"POS: {token.pos_}") # kelimenin dilbilgisel özelliği (verb, noun....)
    print(f"Tag: {token.tag_}") # kelimenin detaylı dilbilgisel özelliği (şimdiki zaman, geçmis zaman...)
    print(f"Dependency: {token.dep_}") # kelimenin rolü
    print(f"Shape: {token.shape_}") # kelimenin karakter yapısı
    print(f"Is alpha: {token.is_alpha}") # kelimenin sadece alfabetik karakterlerden oluşup oluşmadığını kontrol eder.
    print(f"Is stop: {token.is_stop}") # kelimenin stop words olup olmadığı
    print(f"Morphology: {token.morph}") # kelimenin morfolojik özelliklerini verir
    print(f"Is plural: {'Number = Plur' in token.morph}") # kelimenin cogul olup olmadıgı
    
    print()