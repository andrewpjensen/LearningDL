#Using a BERTTokenizer specific to the BERT base cased model
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
print(tokenizer("Using a Transformer network is simple"))

#Using an AutoTokenizer that can be used with any checkpoint or model
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print(tokenizer("Using a Transformer network is simple"))

#tokenizer.save_pretrained("tokenizer_test1")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)


#TEST CASE; understanding more about tokenizer
#Import the AutoTokenizer
from transformers import AutoTokenizer

#Use the bert-base-cased checkpoint with the autotokenizer library
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

#Create sequence
sequence = "I\'ve been waiting for a HuggingFace course my whole life."

#break up the sequence using the tokenize() method
tokens = tokenizer.tokenize(sequence)
print(tokens)

#Turn the create tokens into numerical ids 
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

#decode the ids back into a comprehendible sentence
decoded_string = tokenizer.decode(ids)
print(decoded_string)
