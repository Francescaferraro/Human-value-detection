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
import matplotlib.pyplot as plt
import numpy as np
import transformers
import torch
from transformers import pipeline
import string
from transformers import BertTokenizer, BertForMaskedLM
import statistics

from valuemap.models import Model, MultiModel
from valuemap.values import ValueMap, ValueSearch
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors


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


def radial_plot(results):
    # Create a list of adjectives and their probabilities
    adjectives = list(results.keys())
    probabilities = list(results.values())

    # Compute angle for each adjective
    angles = np.linspace(0, 2 * np.pi, len(adjectives), endpoint=False).tolist()

    # The figure is plotted in a polar projection
    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
    
    # Plot each line separately
    for i in range(len(adjectives)):
        ax.plot([angles[i], angles[i]], [0, probabilities[i]], color='blue')
    
    # Fill the area under the curve
    ax.fill(angles, probabilities, 'blue', alpha=0.1)
    
    # Set the yticks to be empty and xticks to be the adjectives
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(adjectives)
    
    # Display the plot
    plt.show()


# +
import random

# Random argument selection
input_argument = random.choice(list(input.keys()))

argument = input_argument
stance = input[input_argument][0]
conclusion = input[input_argument][1]


# -

def generate_prompt(stance, conclusion, argument):
    return f"i am {stance} the fact that {conclusion} because i think that {argument}. i am a {tokenizer.mask_token}."


def generate_word(model):
    # Initialize an empty list to store the results
    results = []

    # Create prompt 
    prompt = generate_prompt(stance, conclusion, argument)
    
    # Generate the filling
    output = model(prompt)
   
    # Sort output by 'score'
    output.sort(key=lambda x: x['score'], reverse=True)
    
    # Create a dictionary of other predicted words with their scores
    predictions = {result['token_str']: result['score'] for result in output}

    # Extract 'sequence' with higher 'score'
    description = output[0]['sequence']
    
    return predictions, description


generate_word(model_bert)

radial_plot(generate_word(model_bert)[0])

adjective_list = [ 
    "conservative","republican","capitalist",
    "libertarian","centrist","democrat",
    "liberal","progressive","socialist",
    "communist","anarchist",
]


def sort_elements(element) :
    sorted_element = {k: v for k, v in sorted(element.items(), key=lambda item: item[1], reverse=True)}
    return sorted_element


def generate_word_adj(model, tokenizer, adjective_list):
    # Create prompt 
    prompt = generate_prompt(stance, conclusion, argument)
    
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

    # Sort the probabilities dictionary by value in descending order
    sorted_probabilities = sort_elements(probabilities)

    # Choose the word with the highest probability
    top_word = next(iter(sorted_probabilities))

    # Replace the mask token in the original prompt with the top word
    description = prompt.replace(tokenizer.mask_token, top_word)

    return probabilities, sorted_probabilities, description


desc_order_probabilities = generate_word_adj(model, tokenizer, adjective_list)[1]
political_order_probabilities = generate_word_adj(model, tokenizer, adjective_list)[0]

generate_word_adj(model, tokenizer, adjective_list)[1:]

radial_plot(political_order_probabilities)

en_path = 'C:/Users/ffraa/Desktop/Università/Tesi/vast-value-map-main/valuemap/Data/wiki.en.vec'
n_max = 100000
eng = Model(en_path, n_max=n_max)
oracle = MultiModel(eng=eng)
vocabulary_path = 'C:/Users/ffraa/Desktop/Università/Tesi/vast-value-map-main/valuemap/Refined_dictionary.txt'
value_map = ValueMap.from_vocabulary(vocabulary_path) 
value_oracle = ValueSearch(valuemap=value_map, oracle=oracle, k=10, depth=2)

# Load pretrained word2vec model
path = 'C:/Users/ffraa/Desktop/Università/Tesi/vast-value-map-main//valuemap/GoogleNews-vectors-negative300.bin'
similarity_model = KeyedVectors.load_word2vec_format(path, binary=True)


def find_closest_word(word, dictionary, model):
    # Get the embedding for the predicted word
    word_embedding = model[word]
    max_similarity = -1
    closest_word = None
    # Iterate over all words in the dictionary
    for dict_word in dictionary.keys():
        try:
            # Get the embedding for the word from the dictionary
            dict_word_embedding = model[dict_word]

            # Calculate the cosine similarity between the two embeddings
            similarity = cosine_similarity([word_embedding], [dict_word_embedding])

            # If the similarity is higher than the current maximum, update the maximum
            if similarity > max_similarity:
                max_similarity = similarity
                closest_word = dict_word
        except KeyError:
            continue  # Continue if the word isn't into the model
    return closest_word


def calculate_oracle_answer(word):
    answers = value_oracle.search(word, lang='eng')
    aggregated = value_oracle.aggregated_search(word, lang='eng')
    
    if answers is not None:
        # Convert the aggregated values to a series and normalize them
        aggregated_series = pd.Series(aggregated).sort_values(ascending=False) / sum(aggregated.values())
        
        return answers, aggregated_series

    return None, None


def human_value_detection(word, dictionary, model, aggregated_values):
    
    try:
        # Calculate answers
        print(f"\n{word}")
        answers, aggregated = calculate_oracle_answer(word)
        
        if not answers :
            print(f"No values available for the word. Finding the closest word in the dictionary...")
            closest_word = find_closest_word(word, dictionary, model)
            print(f"The closest word in the dictionary for {word} is {closest_word}.")        
            # Closest word's answers
            answers, aggregated = calculate_oracle_answer(closest_word)
        # Print answers 
        sorted_answers = sorted(answers, key=lambda x: x['similarity'], reverse=True)
        for answer in sorted_answers:
            print(f"{answer}")

        # Add aggregate values
        for value, probability in aggregated.items():
            if value in aggregated_values:
                aggregated_values[value] += probability
            else:
                aggregated_values[value] = probability
    except KeyError as e:
        print(e)
    # Normalize aggregated values
    total_probability = sum(aggregated_values.values())
    normalized_values = {value: probability / total_probability for value, probability in aggregated_values.items()}
    
    return normalized_values


aggregated_values = {}
for word in generate_word(model_bert)[0].keys():
    aggregated_values = human_value_detection(word, value_map, similarity_model, aggregated_values)

#Print radial plot containing all the occurencies for the values in sorted order
sorted_aggregated_values = sort_elements(aggregated_values)
radial_plot(sorted_aggregated_values)
print(sorted_aggregated_values)

aggregated_values_adj = {}
for word in generate_word_adj(model, tokenizer, adjective_list)[1].keys() :
    aggregated_values_adj = human_value_detection(word, value_map, similarity_model, aggregated_values_adj)

#Print radial plot containing all the occurencies for the values in sorted order
sorted_aggregated_values_adj = sort_elements(aggregated_values_adj)
radial_plot(sorted_aggregated_values_adj)
print(sorted_aggregated_values_adj)

# Import the human values from values.py labels
from valuemap.values import VALUE_LABELS

# +
# For the guided approach, it's used an a-priori association between adjectives and human values
mapped_adjective_values = {
    "conservative": ['1', '3', '2'],
    "republican": ['1', '9', '10'],
    "capitalist": ['6', '9', '10'],
    "libertarian": ['6', '7', '8'],
    "centrist": ['4', '5', '2'],
    "democrat": ['5', '4', '6'],
    "liberal": ['5', '7', '6'],
    "progressive": ['5', '7', '4'],
    "socialist": ['5', '4', '1'],
    "communist": ['5', '1', '4'],
    "anarchist": ['6', '7', '8'],
}

for adjective, values in mapped_adjective_values.items():
    print(f"{adjective}: {[VALUE_LABELS[value] for value in values]}")

# +
desc_order = {}
value_probs = {}

for adj, prob in desc_order_probabilities.items():
    human_value = mapped_adjective_values.get(adj)
    if human_value is not None:
        human_value = [VALUE_LABELS.get(k) for k in human_value]
        if None not in human_value:
            desc_order[tuple(human_value)] = prob
            
for values, prob in desc_order.items():
    for value in values:
        if value not in value_probs:
            value_probs[value] = 0
        value_probs[value] += prob

total_prob = sum(value_probs.values())

# Normalize
normalized_probs = {k: v / total_prob for k, v in value_probs.items()}

radial_plot(normalized_probs)
normalized_probs
# -
