from transformers import pipeline
from pprint import pprint

classifier = pipeline("sentiment-analysis")
pprint(classifier("I've been waiting for a HuggingFace course my whole life."))

classifier2 = pipeline("zero-shot-classification")
pprint(classifier2(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
))

generator = pipeline("text-generation", model = "distilgpt2")
#NUM RETURN SEQUENCES - how many sequences, and max length of each output
pprint(generator("In this course, we will teach you how to", num_return_sequences = 2, max_length = 30))

unmasker = pipeline("fill-mask")
pprint(unmasker("This course will teach you all about <mask> models.", top_k=2))

ner = pipeline("ner", grouped_entities=True)
pprint(ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))

question_answerer = pipeline("question-answering")
pprint(question_answerer(
    question="What city do I work in?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
))

summarizer = pipeline("summarization")
pprint(summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
))

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
pprint(translator("Ce cours est produit par Hugging Face."))