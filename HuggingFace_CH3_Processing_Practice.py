from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

#Load up datasets glue and mrpc
raw_datasets = load_dataset("sst2", "mrpc")

#Grab the train dataset from the whole dateset
raw_train_dataset = raw_datasets['train']

# print(raw_train_dataset.features)
# #{'idx': Value(dtype='int32', id=None), 'sentence': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=2, names=['negative', 'positive'], id=None)}

#Set checkpoint as bert-base-uncased
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#Create function that returns the tokenizer for the dataset
def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)

#Use the map function to map the tokenized dataset to the orginal dataset
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

#added a feature that removes the sentence from the tokenized data set for padding
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

print(tokenized_datasets['train'][:5])
#Create a data collator to start the padding process
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#Sample to demonstrate the different lengths of the dataset
samples = tokenized_datasets["train"][:5]

samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence"]}
print({len(x) for x in samples["input_ids"]})

batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})
# #{'input_ids': torch.Size([8, 67]), 'token_type_ids': torch.Size([8, 67]), 'attention_mask': torch.Size([8, 67]), 'labels': torch.Size([8])}
