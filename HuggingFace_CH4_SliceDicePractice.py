from transformers import AutoTokenizer
from datasets import load_dataset

slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)


def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)

data_files = {"train": "data\drugsComTrain_raw.tsv", "test": "data\drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
# tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)
# print('done')

