from transformers import pipeline
from pprint import pprint

#GET INTO MODEL
# ner = pipeline("ner", grouped_entities=True, model = "flair/pos-english")
# pprint(ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))

newlist = []
finallist = []
with open('testtextsum.txt') as file:
    mystr = file.read()

print(mystr)

summarizer = pipeline("summarization", truncation = True)
pprint(summarizer(
    mystr
))