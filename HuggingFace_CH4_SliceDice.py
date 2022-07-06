from datasets import load_dataset
from pprint import pprint
import html

data_files = {"train": "data\drugsComTrain_raw.tsv", "test": "data\drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
drug_sample[:3]

for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))

drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
# print(drug_dataset)

num = drug_dataset['train'].unique('drugName')
num2 = drug_dataset['train'].unique('condition')
num3 = drug_dataset['test'].unique('drugName')
num4 = drug_dataset['test'].unique('condition')
# print(len(num))
# print(len(num2))
# print(len(num3))
# print(len(num4))

def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

drug_dataset = drug_dataset.map(lowercase_condition)
# Check that lowercasing worked
print(drug_dataset["train"]["condition"][:3])

#adding new columns
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}

drug_dataset = drug_dataset.map(compute_review_length)
# Inspect the first training example
print(drug_dataset["train"][0])

print(drug_dataset["train"].sort("review_length")[:3])

#strip reviews with less than 30 reviews
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)

drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})

new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
)

