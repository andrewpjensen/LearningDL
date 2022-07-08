from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizertwo = AutoTokenizer.from_pretrained("bert-base-cased")

print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
print(tokenizertwo.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))

print(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?"))

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")

# tokenizer = AutoTokenizer.from_pretrained("t5-small")
# tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")

