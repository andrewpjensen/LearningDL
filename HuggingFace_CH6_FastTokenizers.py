from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
print(type(encoding))

print(tokenizer.is_fast)
print(encoding.is_fast)

print(encoding.tokens())
print(encoding.word_ids())

tokenizer2 = AutoTokenizer.from_pretrained("bert-base-cased")
example2 = '81s'
encoder2 = tokenizer2(example2)
print(encoder2.tokens())
print(encoder2.word_ids())

tokenizer3 = AutoTokenizer.from_pretrained("roberta-base")
example3 = '81s'
encoder3 = tokenizer3(example3)
print(encoder3.tokens())
print(encoder3.word_ids())

start, end = encoding.word_to_chars(3)
print(example[start:end])

tokenizer4 = AutoTokenizer.from_pretrained("bert-base-cased")
example4 = ['Hello This is a test', 'This is another test']
encoder4 = tokenizer4(example4)
print(encoder4.tokens())
print(encoder4.word_ids())

from transformers import pipeline

token_classifier = pipeline("token-classification")
print(token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn."))


from transformers import pipeline

token_classifier = pipeline("token-classification", aggregation_strategy="simple")
print(token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn."))

