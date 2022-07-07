import GetOldTweets3 as got
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

#Getting classifier from transformers pipeline
classifier = pipeline("zero-shot-classification")

