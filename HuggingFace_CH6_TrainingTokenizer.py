from datasets import load_dataset
from pprint import pprint
# This can take a few minutes to load, so grab a coffee or tea while you wait!
raw_datasets = load_dataset("code_search_net", "python")

pprint(raw_datasets["train"])

# This returns an error, because you need to make an iterator
# print(raw_datasets["train"][123456]["whole_func_string"])

# Don't uncomment the following line unless your dataset is small!
# training_corpus = [raw_datasets["train"][i: i + 1000]["whole_func_string"] for i in range(0, len(raw_datasets["train"]), 1000)]

# def get_training_corpus():
#     return (
#         raw_datasets["train"][i : i + 1000]["whole_func_string"]
#         for i in range(0, len(raw_datasets["train"]), 1000)
#     )

#(OR)

def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]

training_corpus = get_training_corpus()

from transformers import AutoTokenizer
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

#Testing the original tokenizer
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

# tokens = old_tokenizer.tokenize(example)
# print(tokens)

#Training a new tokenizer
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
print(tokenizer)

tokens = tokenizer.tokenize(example)
print(tokens)

# tokenizer.save_pretrained("code-search-net-tokenizer")