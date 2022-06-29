import torch
from transformers import BertConfig, BertModel, BertTokenizer
from pprint import pprint
# Building the config
config = BertConfig()

# Building the tokenizer from the config
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Building the model from the config
model = BertModel.from_pretrained("bert-base-cased")

#initialize Sequences
sequences = ["Hello!", "Cool.", "Nice!"]

#Initialize tokenizer
inputs = tokenizer(sequences, return_tensors="pt")

#Encoded Sequences are builts from the input_ids - this is a tensor already
encoded_sequences = inputs.input_ids

#Torch clone the tensor, rather than .tensor() because it's already structured
model_inputs = torch.clone(encoded_sequences)

#Prints the tensor, under the new variable
print(model_inputs)

