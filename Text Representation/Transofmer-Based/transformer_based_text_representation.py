# import libraries
from transformers import AutoTokenizer, AutoModel
import torch

# model ve tokenizer yükle
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# input text tanimla
text = "Transformers can be used for natural language processing."

# metni tokenlara cevirmek
inputs = tokenizer(text, return_tensors = "pt") # cıktı pytorch tensoru olarak return edilir.

# modeli kullanarak metin temsili oluştur
with torch.no_grad(): # gradyanların hesaplanması durdurulur, böylece bellegi daha verimli kullanırız.
    outputs = model(**inputs)

# modelin cıkısından son gizli durumu alalım
last_hidden_state = outputs.last_hidden_state # tüm token cıktılarını almak ıcın

# ilk tokenin embeddingini alalım ve print ettirelim
first_token_embedding = last_hidden_state[0,0,:].numpy()

print(f"Metin temsili: {first_token_embedding}")


