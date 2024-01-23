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
from sklearn.metrics import f1_score, accuracy_score


# -

def load_dataset():
    # Load the arguments and labels into separate DataFrames
    df_arguments = pd.read_csv('arguments-validation.tsv', delimiter='\t')
    df_labels = pd.read_csv('labels-validation.tsv', delimiter='\t')

    # Merge the two DataFrames on the 'Argument ID' column
    df = pd.merge(df_arguments, df_labels, on='Argument ID')

    # Extract the argument text from each DataFrame
    id = df_arguments['Argument ID'].tolist()
    arguments = df_arguments['Premise'].tolist()
    stances = df_arguments['Stance'].tolist()
    conclusions = df_arguments['Conclusion'].tolist()

    return arguments, stances, conclusions, id, df


def get_all_human_values(df):
    # Initialize an empty dictionary to store the human values for each argument ID
    human_values_dict = {}

    # Iterate over all rows in the DataFrame
    for _, row in df.iterrows():
        # Get the argument ID
        argument_id = row['Argument ID']

        # Extract the human values
        values_dict = row.iloc[4:].to_dict()  # Adjust the column index as needed

        # Get a list of human values that have value 1
        human_values = [key.split(':')[0] for key, value in values_dict.items() if value == 1]

        # Convert the list to a set to remove duplicates, then convert it back to a list
        human_values = list(set(human_values))

        # Store the human values in the dictionary
        human_values_dict[argument_id] = human_values

    return human_values_dict


semeval_validation_set_results = get_all_human_values(load_dataset()[4])

print(semeval_validation_set_results)


def process_sentences(sentences):
    processed = []
    for sentence in sentences:
        # Remove punctuation
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        # Convert to lowercase
        sentence = sentence.lower()
        processed.append(sentence)
    return processed


# +
# Load dataset
arguments, stances, conclusions, id, df = load_dataset()

# Preprocess premise and conclusion
arguments = process_sentences(arguments)
conclusions = process_sentences(conclusions)
# -

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


# -

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


def generate_prompt(stance, conclusion, argument):
    return f"i am {stance} the fact that {conclusion} because i think that {argument}. i am a {tokenizer.mask_token}."


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


def sort_elements(element) :
    sorted_element = {k: v for k, v in sorted(element.items(), key=lambda item: item[1], reverse=True)}
    return sorted_element


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


# Import the human values from values.py labels
from valuemap.values import VALUE_LABELS


def generate_word(stance, conclusion, argument, model):
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


adjective_list = [ 
    "conservative","republican","capitalist",
    "libertarian","centrist","democrat",
    "liberal","progressive","socialist",
    "communist","anarchist",
]


def generate_word_adj(stance, conclusion, argument, model, tokenizer, adjective_list):
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
# -



# +
# Initialize an empty dictionary to store the human values for each argument ID
human_values_dict = {}

# Iterate over all arguments, stances, and conclusions
for argument, stance, conclusion, argument_id in zip(arguments, stances, conclusions, id):
    # Generate words using model
    generated_words = generate_word(stance, conclusion, argument, model_bert)

    # Plot the generated words
    radial_plot(generated_words[0])

    # Initialize an empty dictionary for aggregated values
    aggregated_values = {}

    # Detect human values for each generated word
    for word in generated_words[0].keys():
        aggregated_values = human_value_detection(word, value_map, similarity_model, aggregated_values)

    # Sort the aggregated values
    sorted_aggregated_values = sort_elements(aggregated_values)

    # Plot the sorted aggregated values
    radial_plot(sorted_aggregated_values)

    # Print the sorted aggregated values
    print(sorted_aggregated_values)

    # Store the sorted aggregated values in the dictionary
    human_values_dict[argument_id] = sorted_aggregated_values
# -

transformed_dict = {k: list(v.keys()) for k, v in human_values_dict.items()}
print(transformed_dict)



# +
# Initialize an empty dictionary to store the human values for each argument ID
human_values_dict_guided = {}

# Iterate over all arguments, stances, and conclusions
for argument, stance, conclusion, argument_id in zip(arguments, stances, conclusions, id):
    # Generate words using model
    generated_words = generate_word_adj(stance, conclusion, argument, model, tokenizer, adjective_list)[1]
    
    # Plot the generated words
    radial_plot(generate_word_adj(stance, conclusion, argument, model, tokenizer, adjective_list)[0])
    
    # Initialize an empty dictionary for aggregated values
    aggregated_values_adj = {}

    # Detect human values for each generated word
    for word in generated_words.keys():
        aggregated_values_adj = human_value_detection(word, value_map, similarity_model, aggregated_values_adj)

    # Sort the aggregated values
    sorted_aggregated_values = sort_elements(aggregated_values_adj)
    
    # Plot the sorted aggregated values
    radial_plot(sorted_aggregated_values)

    # Print the sorted aggregated values
    print(sorted_aggregated_values)

    # Store the sorted aggregated values in the dictionary
    human_values_dict_guided[argument_id] = sorted_aggregated_values
# -

transformed_dict_guided = {k: list(v.keys()) for k, v in human_values_dict_guided.items()}
print(transformed_dict_guided)



# +
human_values_dict_guidedValues = {}

for argument, stance, conclusion, argument_id in zip(arguments, stances, conclusions, id):
    desc_order = {}
    value_probs = {}
    
    for adj, prob in generate_word_adj(stance, conclusion, argument, model, tokenizer, adjective_list)[1].items():
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
    print(normalized_probs)
    human_values_dict_guidedValues[argument_id] = normalized_probs
# -



transformed_dict_guidedValues = {k: list(v.keys())[:3] for k, v in human_values_dict_guidedValues.items()}
print(transformed_dict_guidedValues)



def calculate_average_metrics(predicted_dict, ground_truth_dict):
    total_f1 = 0
    total_accuracy = 0
    count = 0

    for key in ground_truth_dict.keys():
        if key in predicted_dict:
            # Converting human values to binary format for calculation
            human_values = list(set(ground_truth_dict[key] + predicted_dict[key]))
            y_true_binary = [1 if value in ground_truth_dict[key] else 0 for value in human_values]
            y_pred_binary = [1 if value in predicted_dict[key] else 0 for value in human_values]

            # Calculating F1 Score
            f1 = f1_score(y_true_binary, y_pred_binary, average='binary')

            # Calculating Accuracy
            accuracy = accuracy_score(y_true_binary, y_pred_binary)

            total_f1 += f1
            total_accuracy += accuracy
            count += 1

    # Calculating average F1 and Accuracy
    avg_f1 = total_f1 / count if count > 0 else 0
    avg_accuracy = total_accuracy / count if count > 0 else 0

    return avg_f1, avg_accuracy


f1_free, accuracy_free = calculate_average_metrics(transformed_dict, semeval_validation_set_results)
print(f"FREE APPROACH = F1 Score: {f1}, Accuracy: {accuracy}")

f1_guided, accuracy_guided = calculate_average_metrics(transformed_dict_guided, semeval_validation_set_results)
print(f"GUIDED APPROACH - FREE MAPPING = F1 Score: {f1}, Accuracy: {accuracy}")

f1_guidedMapping, accuracy_guidedMapping = calculate_average_metrics(transformed_dict_guidedValues, semeval_validation_set_results)
print(f"GUIDED APPROACH - GUIDED MAPPING = F1 Score: {f1}, Accuracy: {accuracy}")

# +
import matplotlib.pyplot as plt

models = ['Free approach', 'Guided approach', 'Guided Mapping approach']
f1_scores = [f1_free, f1_guided, f1_guidedMapping]
accuracies = [accuracy_free, accuracy_guided, accuracy_guidedMapping]

fig, ax = plt.subplots(figsize=(6, 4))

bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(models, f1_scores, bar_width,
alpha=opacity,
color='b',
label='F1 Score')

rects2 = plt.bar(models, accuracies, bar_width,
alpha=opacity,
color='g',
label='Accuracy')

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Scores by model')
plt.legend()

plt.tight_layout()
plt.show()

# -
