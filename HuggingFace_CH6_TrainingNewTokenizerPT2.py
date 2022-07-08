from transformers import AutoTokenizer
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

#Testing the original tokenizer
# example = '''def add_numbers(a, b):
#     """Add the two numbers `a` and `b`."""
#     return a + b'''

# tokens = old_tokenizer.tokenize(example)
# print(tokens)

#Training a new tokenizer
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)