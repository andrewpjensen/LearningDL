from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

#Load up datasets glue and mrpc
raw_datasets = load_dataset("glue", "mrpc")

#Grab the train dataset from the whole dateset
raw_train_dataset = raw_datasets['train']

#Use features parameter to see into the datset
datafeatures = raw_train_dataset.features
 
#Test to see what the 15th element of the train dataset is
test1 = raw_train_dataset[14]
print(test1)

#Test to see what the 87th element of the validation dataset is
raw_validate_dataset = raw_datasets['validation']
test2 = raw_train_dataset[86]
print(test2)

#Set checkpoint as bert-base-uncased
checkpoint = "bert-base-uncased"

#Tokenizer checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Non function way to tokenize the dataset
# tokenized_dataset = tokenizer(
#     raw_datasets["train"]["sentence1"],
#     raw_datasets["train"]["sentence2"],
#     padding=True,
#     truncation=True,
# )

#Create function that returns the tokenizer for the dataset
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

#Use the map function to map the tokenized dataset to the orginal dataset
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

#Create a data collator to start the padding process
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#Sample to demonstrate the different lengths of the dataset
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print([len(x) for x in samples["input_ids"]])
#[50, 59, 47, 67, 59, 50, 62, 32]

#batch the samples together
batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})
#{'input_ids': torch.Size([8, 67]), 'token_type_ids': torch.Size([8, 67]), 'attention_mask': torch.Size([8, 67]), 'labels': torch.Size([8])}
