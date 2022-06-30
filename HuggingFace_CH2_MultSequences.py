import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#Checkpoint, set tokenizer and model, tokenize a sequence, and convert to IDs
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequence = "I've been waiting for a HuggingFace course my whole life."
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

#Create a tensor with the IDs wrapped in an array
input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

#Move IDs into the model 
output = model(input_ids)
print("Logits:", output.logits)

#Batch the ids together in one array
batched_ids = [ids, ids]

#Create a new tensor
newtensor = torch.tensor(batched_ids)
print(newtensor)

# RUn the new tensor through the model (two sets of logits)
output = model(newtensor)
print(output)

#Padding
padding_id = 200

#This batch cannot be run, because it is the wrong shape
batched_ids_wrong = [
    [200,200,200],
    [200,200]
]

#This is the right format  
batched_ids_right = [
    [200,200,200],
    [200,200, padding_id]
]

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1,1,1],
    [1,1,0],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
output = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(output.logits)