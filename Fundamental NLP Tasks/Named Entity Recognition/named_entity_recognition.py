"""
varlık ismi tanıma: metin (cümle) -> metin icinde bulunan varlık isimlerini tanımla.
"""

# import libraries
import pandas as pd
import spacy


# spacy modeli ile varlık ismi tanımla
nlp = spacy.load("en_core_web_sm") # spacy kütüphanesi ingilizce dil modeli

content = "Alice works at Amazon and lives in London. She visited British Museum last weekend."

doc = nlp(content) # bu işlem metindeki varlıkları (entities) analiz eder.


for ent in doc.ents:
    # ent.text: entity name (Alice, Amazon)
    # ent.start_char, ent.end_char: entity'nin text'teki başlangıç ve bitiş karakterleri
    # ent.label_: entity type
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
    
    
# ent.lemma_ : entity'nin kök hali
entities = [(ent.text, ent.label_, ent.lemma_) for ent in doc.ents]

# varlık listesini pandas dataframe'e çevir daha rahat bir görüntü için, daha rahat incelemek icin

df = pd.DataFrame(entities, columns = ["text", "type", "lemma"])

"""
| Etiket            | Açıklama                                                              |
| ----------------- | --------------------------------------------------------------------- |
| **PERSON**        | Kişi isimleri (örneğin: *Barack Obama*)                               |
| **NORP**          | Milliyet, din veya politik gruplar (*American*, *Muslim*, *Democrat*) |
| **FAC**           | Fiziksel yapılar (bina, köprü, havaalanı)                             |
| **ORG**           | Organizasyonlar, şirketler, kurumlar (*Google*, *UN*, *NASA*)         |
| **GPE**           | Ülkeler, şehirler, eyaletler (*Turkey*, *Istanbul*, *California*)     |
| **LOC**           | Coğrafi yerler (dağlar, nehirler - *Mount Everest*, *Amazon*)         |
| **PRODUCT**       | Ürünler (araçlar, cihazlar, yazılımlar)                               |
| **EVENT**         | Tarihi olaylar, festivaller, savaşlar (*World War II*, *Olympics*)    |
| **WORK\_OF\_ART** | Sanat eserleri (*Mona Lisa*, *Game of Thrones*)                       |
| **LAW**           | Yasal belgeler (*Constitution*, *First Amendment*)                    |
| **LANGUAGE**      | Diller (*English*, *Turkish*)                                         |
| **DATE**          | Tarihler (*June 5th, 2025*)                                           |
| **TIME**          | Saatler (*10:30am*, *midnight*)                                       |
| **PERCENT**       | Yüzde değerleri (*45%*)                                               |
| **MONEY**         | Para miktarları (*\$20 million*)                                      |
| **QUANTITY**      | Miktarlar (*200 kilograms*)                                           |
| **ORDINAL**       | Sıra sayıları (*first*, *second*)                                     |
| **CARDINAL**      | Sayılar (*one*, *two thousand*)                                       |

"""