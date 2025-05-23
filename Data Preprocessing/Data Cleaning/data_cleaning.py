# Metinlerdeki fazla boşlukları ortadan kaldırma
text = "Hello,     World!          2025"
"""
text.split()
Out[2]: ['Hello,', 'World!', '2025']
"""

cleaned_text1 = " ".join(text.split()) 

print(f"Original text : {text} \nCleaned text : {cleaned_text1}")

# " ".join Kısmını boşlukla oluşturmalıyız ki düzgün bir cleaned text elde edelim.

# Büyük - küçük harf çevirimi
text = "Hello, World! 2025"
cleaned_text2 = text.lower()
print(f"Original text : {text} \nCleaned text 2 : {cleaned_text2}")

# Noktalama işaretlerini kaldırma
import string

text = "Hello, World! 2025"

cleaned_text3 = text.translate(str.maketrans("","",string.punctuation))
print(f"Original text : {text} \nCleaned text 3 : {cleaned_text3}")
# İlk 2 argümanımız boş, buda hiçbir karakteri bir şeyle değiştirmeden noktalamaları kaldır.
# İlk 2 argümana bir değer girersek örneğin H girip yerine K girersek o harfler değişir.

# Özel karakter kaldırma
import re

text = "Hello, World! 2035%  @"

cleaned_text4 = re.sub(r"[^A-Za-z0-9\s]","1",text)
# Buraada A-Z ve a-z ve 0-9 hariç özel karakterleri kaldırıp yerine 1 koyduk.
# Kullanım anlaşılsın diye 1 koydum normalde genelde boşluk koyuluyor.
print(f"Original text : {text} \nCleaned text 4 : {cleaned_text4}")

# Yazım hatalarını düzeltme 
from textblob import TextBlob # text analysis'de kullanılan bir kütüphane

text = "Hellio, Wirld! 2025"
cleaned_text5 = TextBlob(text).correct() #correct() : yazım hatalarını düzeltir.

print(f"Original text : {text} \nCleaned text 5 : {cleaned_text5}")

# html ya da url etiketlerini kaldırma
from bs4 import BeautifulSoup

html_text = "<div>Hello, World! 2025</div>" # html etiketi var

# beautiful soup ile htlm yapısını parse et, get_text ile text kısmını çek.

cleaned_text6 = BeautifulSoup(html_text, "html.parser").get_text()
# html_parser kullanımı daha hızlı fakat kütüphane yüklemek lazım bunun yerine html.parser kullandık.


print(f"Original text : {html_text} \nCleaned text 6 : {cleaned_text6}")