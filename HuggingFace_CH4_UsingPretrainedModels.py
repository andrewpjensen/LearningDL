from transformers import pipeline
from pprint import pprint
camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")

from datasets import load_dataset

uci_database = load_dataset('json', data_files='data\wikivital_mathematics.json') #C:\Users\andre\projects\LearningDL\data\wikivital_mathematics.json
print(uci_database['train'][10])