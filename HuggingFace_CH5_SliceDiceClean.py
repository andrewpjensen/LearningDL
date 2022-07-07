from datasets import load_dataset
from pprint import pprint
import html
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

#Identify datasets for use
data_files = {"train": "data\drugsComTrain_raw.tsv", "test": "data\drugsComTest_raw.tsv"}
#Load the dataset \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
#Random sample and select
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
pprint(drug_sample[:3])
#Problems include 1) the 0 column looks like patient ID, the condition column has upper and lowercase labels \
# and 3) the reviews are of varying length and contain a mix of Python line seperators and HTML character codes

#Check if column length matches the length of the split of the dataset
for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))

#Change column name to patient_id
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
print(drug_dataset, '\n')
# pprint(drug_dataset['train'][:3])

#Practice - use dataset.unique() function to find the unique number of drugs and conditions in the training and test sets
length1a = len(drug_dataset['train'].unique('drugName'))
length1b = len(drug_dataset['train'].unique('condition'))
length2a = len(drug_dataset['test'].unique('drugName'))
length2b = len(drug_dataset['test'].unique('condition'))
print(f'The training dataset number of drugs is {length1a}, and the number of conditions is {length1b} \n')
print(f'The testing dataset number of drugs is {length2a}, and the number of conditions is {length2b} \n')

#Next, we need to deal with with normalizing the condition labels

#First, filter out none conditions 
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

#Next, define a function that can be applied to each row of the drug_dataset that will lower the text of each condition
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

#Next, define a function that can be applied to each row of the drug_dataset that will lower the text of each review
def lowercase_review(example):
    return {"review": example["review"].lower()}

#Then, use the map function to apply the function to every row of the dataset
drug_dataset = drug_dataset.map(lowercase_condition)
drug_dataset = drug_dataset.map(lowercase_review)

#Test to see if that worked
print(drug_dataset["train"]["condition"][:3])

#Create New Columns - checking the number of words in each review

#Make a function that computes the length of the review
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}

#Map the new function compute_review_length to the drug_dataset OR you can use the Dataset.add_column() function
drug_dataset = drug_dataset.map(compute_review_length)
# Inspect the first training example
pprint(drug_dataset["train"][0])

#See lower bounds
print(drug_dataset["train"].sort("review_length")[:3])

print(drug_dataset["train"].sort("review_length", reverse = True)[:3])

#filter reviews less than 30 words in length
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)

#Remove the HTML characters from the review
drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})
print('complete1')
pprint(drug_dataset["train"][0])

#Increase speed with comprehensions 
new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
)
print('complete2')

def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)

#Speeding up even more iwth multiprocessing
tokenized_dataset = drug_dataset.map(tokenize_function, batched=True)
slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)
def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)

tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)

def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extract mapping between new and old indices
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result

tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
print(tokenized_dataset)

#MAKE A VALIDATE DATASET
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]
drug_dataset_clean

#drug_dataset.set_format("pandas")
#drug_dataset_clean.save_to_disk("drug-reviews")

