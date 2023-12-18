# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import transformers
import torch
from transformers import pipeline
import string
from transformers import BertTokenizer, BertForMaskedLM

from valuemap.models import Model, MultiModel
from valuemap.values import ValueMap, ValueSearch


# -

def load_dataset():
    # Load the dataset into separate DataFrames for each split
    df_training = pd.read_csv('arguments-training.tsv', delimiter='\t')
    df_validation = pd.read_csv('arguments-validation.tsv', delimiter='\t')
    df_test = pd.read_csv('arguments-test.tsv', delimiter='\t')

    # Concatenate all the dataframes
    df = pd.concat([df_training, df_validation, df_test])

    # Extract the argument text from each DataFrame
    arguments = df['Premise'].tolist()
    stances = df['Stance'].tolist()
    conclusions = df['Conclusion'].tolist()

    return arguments, stances, conclusions


def process_sentences(sentences):
    processed = []
    for sentence in sentences:
        # Remove punctuation
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        # Convert to lowercase
        sentence = sentence.lower()
        processed.append(sentence)
    return processed


# model inizialization
model_bert = pipeline('fill-mask', model='bert-base-uncased') # Bert


# +
# model inizialization for adjectiveList-based approach
def load_model() :
    # Load the model and tokenizer
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

model, tokenizer = load_model()[0], load_model()[1]

# +
# Load dataset
arguments, stances, conclusions = load_dataset()

# Preprocess premise and conclusion
arguments = process_sentences(arguments)
conclusions = process_sentences(conclusions)
# -

# Map each argument to the corresponding stance and conclusion
if len(arguments) != len(stances) or len(arguments) != len(conclusions):
    print("Error: incompatible data")
else:
    input = {}
    for i in range(len(arguments)):
        input[arguments[i]] = (stances[i], conclusions[i])

# +
import random

# Random argument selection
input_argument = random.choice(list(input.keys()))

argument = input_argument
stance = input[input_argument][0]
conclusion = input[input_argument][1]


# -

def generate_word(model):
    # Initialize an empty list to store the results
    results = []

    # Create prompt 
    prompt = f"i am {stance} the fact that {conclusion} because i think that {argument}. i am a {model.tokenizer.mask_token}."
    
    # Generate the filling
    output = model(prompt)
   
    # Sort output by 'score'
    output.sort(key=lambda x: x['score'], reverse=True)
    
    # Create a dictionary of other predicted words with their scores
    predictions = {result['token_str']: result['score'] for result in output}

    # Extract 'sequence' with higher 'score'
    description = output[0]['sequence']
    
    return predictions, description


print(f"{generate_word(model_bert)[0]}\n\n{generate_word(model_bert)[1]}")

adjective_list = [
    "conservative",
    "liberal",
    "republican",
    "libertarian",
    "democrat",
    "progressive",
    "socialist",
    "communist",
    "anarchist",
    "centrist",
    "capitalist",
]


def generate_word_adj(model, tokenizer, adjective_list):
    # Create prompt 
    prompt = f"i am {stance} the fact that {conclusion} because i think that {argument}. i am a {tokenizer.mask_token}."
    # Initialize a dictionary to store the probabilities
    probabilities = {}
    # For each word in the list, generate a score
    for word in adjective_list:
        # Replace the mask token with the word
        new_prompt = prompt.replace(tokenizer.mask_token, word)
        # Encode the new prompt
        inputs = tokenizer.encode_plus(new_prompt, return_tensors='pt')
        # Generate the filling
        outputs = model(**inputs)
        logits = outputs.logits
        # Calculate the softmax probabilities from logits
        softmax_probs = torch.nn.functional.softmax(logits, dim=-1)
        # Get the probability of the word
        word_id = tokenizer.encode(word, add_special_tokens=False)[0]
        word_prob = softmax_probs[0, -1, word_id].item()

        # Store the probability of the word
        probabilities[word] = word_prob

    # Choose the word with the highest probability
    top_word = max(probabilities, key=probabilities.get)

    # Replace the mask token in the original prompt with the top word
    description = prompt.replace(tokenizer.mask_token, top_word)

    return probabilities, description


print(f"{generate_word_adj(model, tokenizer, adjective_list)[0]}\n\n{generate_word_adj(model, tokenizer, adjective_list)[1]}")






