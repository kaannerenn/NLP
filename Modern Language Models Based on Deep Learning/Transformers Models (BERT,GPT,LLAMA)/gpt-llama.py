"""
metin Ã¼retimi

gpt-2 kullanarak metin Ã¼retimi Ã§alÄ±ÅmasÄ±
llama

"""
#### UYARI !!!! llama kütüphaneleri yaklaşık 13 gb 
# import libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM ## llama libraries

# modelin tanimlanmasÄ±
model_name = "gpt2"
model_name_2 = "huggyllama/llama-7b"

# tokenizer tanÄ±mlama ve model oluÅturma
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2) ##llama

model = GPT2LMHeadModel.from_pretrained(model_name)
model_2 = AutoModelForCausalLM.from_pretrained(model_name_2) ## llama

# metin Ã¼retimi icin gerekli olan baÅlangÄ±Ã§ texti
text = "I go to school for"

# tokenization 
inputs = tokenizer.encode(text, return_tensors = "pt")
inputs_2 = tokenizer_2(text, return_tensors = "pt") ## llama

# metin Ã¼retimi gerÃ§ekleÅtirelim
outputs = model.generate(inputs, max_length = 15) # inputs = modelin baÅlangÄ±Ã§ noktasÄ±, max_length -> maksimum token sayÄ±sÄ±.
outputs_2 = model_2.generate(inputs_2.inputs_ids, max_length = 15) ## llama

# modelin Ã¼rettiÄi tokenlarÄ± okunabilir hale getirmeliyiz
generated_text = tokenizer.decode(outputs[0], skip_special_tokens = True) # Ã¶zel tokenlarÄ± (Ã¶rn: cÃ¼mle baslangÄ±c, bitis tokenlarÄ±) metinden cÄ±kar
generated_text_2 = tokenizer.decode(outputs[0], skip_special_tokens = True) ## llama

# Ã¼retilen metni print ettirelim
print(generated_text)
print(generated_text_2)