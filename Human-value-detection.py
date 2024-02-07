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
import keras
import re

from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer

from valuemap.models import Model, MultiModel
from valuemap.values import ValueMap, ValueSearch
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
# Import the human values from values.py labels
from valuemap.values import VALUE_LABELS

from sklearn.feature_extraction.text import CountVectorizer
from keras.utils import to_categorical
from bornrule import BornClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bornrule import BornClassifier


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

semeval_validation_set_arguments = load_dataset()[0]


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



# +
# For this experiment, only 10 human values are taken into consideration
allowed_values = set(['Security', 'Conformity', 'Tradition', 'Benevolence', 'Universalism', 
                      'Self-direction', 'Stimulation', 'Hedonism', 'Achievement', 'Power'])

# Filtering the SemEval dict for avoiding the other values
filtered_dict = {key: [value for value in values if value in allowed_values] for key, values in semeval_validation_set_results.items()}
# -



def calculate_average_metrics(predicted_dict, ground_truth_dict):
    total_f1 = 0
    total_accuracy = 0
    count = 0

    for key in ground_truth_dict.keys():
        if key in predicted_dict:
            # Converting human values to binary format for calculation, in a case insensitive way
            human_values = list(set([v.lower() for v in ground_truth_dict[key] + predicted_dict[key]]))
            y_true_binary = [1 if value in [v.lower() for v in ground_truth_dict[key]] else 0 for value in human_values]
            y_pred_binary = [1 if value in [v.lower() for v in predicted_dict[key]] else 0 for value in human_values]

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


f1_free, accuracy_free = calculate_average_metrics(transformed_dict, filtered_dict)
print(f"FREE APPROACH = F1 Score: {f1_free}, Accuracy: {accuracy_free}")

f1_guided, accuracy_guided = calculate_average_metrics(transformed_dict_guided, filtered_dict)
print(f"GUIDED APPROACH - FREE MAPPING = F1 Score: {f1_guided}, Accuracy: {accuracy_guided}")

f1_guidedMapping, accuracy_guidedMapping = calculate_average_metrics(transformed_dict_guidedValues, filtered_dict)
print(f"GUIDED APPROACH - GUIDED MAPPING = F1 Score: {f1_guidedMapping}, Accuracy: {accuracy_guidedMapping}")

# +
models = ['Free approach', 'Guided approach', 'Guided Mapping approach']
f1_scores = [f1_free, f1_guided, f1_guidedMapping]
accuracies = [accuracy_free, accuracy_guided, accuracy_guidedMapping]

# Getting the x locations for the groups
ind = np.arange(len(models)) 

fig, ax = plt.subplots(figsize=(6, 4))
bar_width = 0.35
opacity = 0.8

# Create the bars for F1 scores
rects1 = ax.bar(ind - bar_width/2, f1_scores, bar_width, alpha=opacity, color='b', label='F1 Score')

# Create the bars for accuracies
rects2 = ax.bar(ind + bar_width/2, accuracies, bar_width, alpha=opacity, color='g', label='Accuracy')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Scores by model')
ax.set_xticks(ind)
ax.set_xticklabels(models)
ax.legend()

plt.tight_layout()
plt.show()
# -





# +
# Calculating number of occurrencies for each value in the dictionaries, in a case insensitive way
def count_predictions(dictionary):
    all_values = [value.lower() for values in dictionary.values() for value in values]
    return Counter(all_values)

counts_semeval = count_predictions(filtered_dict)
counts_transformed = count_predictions(transformed_dict)
counts_guided = count_predictions(transformed_dict_guided)
counts_guidedValues = count_predictions(transformed_dict_guidedValues)

# Creating a human values list
human_values = list(set(list(counts_semeval.keys()) + list(counts_transformed.keys()) + list(counts_guided.keys()) + list(counts_guidedValues.keys())))

# Counting list
counts = [counts_semeval, counts_transformed, counts_guided, counts_guidedValues]
data = [[count[value] for value in human_values] for count in counts]
# Normalization
total_predictions = sum([sum(count.values()) for count in counts])
data = [[count / total_predictions for count in dataset] for dataset in data]

# Radial plot
theta = np.linspace(0.0, 2 * np.pi, len(human_values), endpoint=False)
colors = ['b', 'r', 'y', '#fa7516']
labels = ['SemEval validation', 'Free approach', 'Guided approach', 'Guided Mapping approach']

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 4))

for i in range(len(data)):
    lines = ax.plot(theta, data[i], color=colors[i], label=labels[i])

ax.set_xticks(theta)
ax.set_xticklabels(human_values)
ax.yaxis.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(2, 1.1), fontsize='small')

plt.show()


# -

def calculate_metrics_per_value(predicted_dict, ground_truth_dict):
    # Initialize a dictionary to store the metrics for each human value
    metrics = defaultdict(lambda: {'total_precision': 0, 'total_recall': 0, 'total_f1': 0, 'count': 0})
    # Iterate over each key in the ground truth dictionary
    for key in ground_truth_dict.keys():
        # Check if the key is also present in the predicted dictionary
        if key in predicted_dict:
            # Convert human values to binary format for calculation
            # All values are converted to lowercase to make the function case insensitive
            human_values = list(set([v.lower() for v in ground_truth_dict[key] + predicted_dict[key]]))
            y_true_binary = [1 if value in [v.lower() for v in ground_truth_dict[key]] else 0 for value in human_values]
            y_pred_binary = [1 if value in [v.lower() for v in predicted_dict[key]] else 0 for value in human_values]
            
            # Calculate Precision, Recall, and F1 Score for each human value
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary)

            # Update the metrics for each human value
            for i, value in enumerate(human_values):
                if y_true_binary[i] == 1 or y_pred_binary[i] == 1:
                    metrics[value]['total_precision'] += precision
                    metrics[value]['total_recall'] += recall
                    metrics[value]['total_f1'] += f1
                    metrics[value]['count'] += 1
    # Calculate average Precision, Recall, and F1 Score for each human value
    for value, data in metrics.items():
        avg_precision = data['total_precision'] / data['count'] if data['count'] > 0 else 0
        avg_recall = data['total_recall'] / data['count'] if data['count'] > 0 else 0
        avg_f1 = data['total_f1'] / data['count'] if data['count'] > 0 else 0
        metrics[value] = {'Average Precision': avg_precision, 'Average Recall': avg_recall, 'Average F1': avg_f1}
    # Sort the metrics by F1 score in descending order
    sorted_metrics = dict(sorted(metrics.items(), key=lambda item: item[1]['Average F1'], reverse=True))
    for value, metric in sorted_metrics.items():
      print(f"Human Value: {value}\n Average F1 Score: {metric['Average F1']}\n Average Precision: {metric['Average Precision']}\n Average Recall: {metric['Average Recall']}\n")
    
    return sorted_metrics


metrics_free = calculate_metrics_per_value(transformed_dict, filtered_dict)

metrics_guided = calculate_metrics_per_value(transformed_dict_guided, filtered_dict)

metrics_guidedValues = calculate_metrics_per_value(transformed_dict_guidedValues, filtered_dict)


def plot_results_byValues(metrics) :
    labels = list(metrics.keys())
    precision_vals = [metrics[l]['Average Precision'] for l in labels]
    recall_vals = [metrics[l]['Average Recall'] for l in labels]
    f1_vals = [metrics[l]['Average F1'] for l in labels]
    
    x = np.arange(len(labels))  
    width = 0.3  
    
    # Dictionary for mapping human values into number ids (from 1 to 10)
    human_values_dict = {human_value: i+1 for i, human_value in enumerate(metrics.keys())}
    # Dictionary for legend explaination
    legend_dict = {i: human_value for human_value, i in VALUE_LABELS.items()}
    # Substituting labels with numbers
    labels = [int(key) for key in VALUE_LABELS.keys()]
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, precision_vals, width, label='Precision')
    rects2 = ax.bar(x, recall_vals, width, label='Recall')
    rects3 = ax.bar(x + width, f1_vals, width, label='F1 Score')
    ax.set_ylabel('Scores')
    ax.set_title('Scores by Human Value')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend() 
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    fig.tight_layout()
    plt.show()
    print("Legend:")
    for num, human_value in legend_dict.items():
        print(f"{num}: {human_value}")


plot_results_byValues(metrics_free)

plot_results_byValues(metrics_guided)

plot_results_byValues(metrics_guidedValues)





def load_traintest_dataset(group):
    # Load the arguments and labels into separate DataFrames
    df_arguments = pd.read_csv('arguments-'+group+'.tsv', delimiter='\t')
    df_labels = pd.read_csv('labels-'+group+'.tsv', delimiter='\t')

    # Merge the two DataFrames on the 'Argument ID' column
    df = pd.merge(df_arguments, df_labels, on='Argument ID')

    # Extract the argument text from each DataFrame
    id = df_arguments['Argument ID'].tolist()
    arguments = df_arguments['Premise'].tolist()
    stances = df_arguments['Stance'].tolist()
    conclusions = df_arguments['Conclusion'].tolist()

    return arguments, stances, conclusions, id, df, df_labels


labels_training = load_traintest_dataset("training")[5]
labels_test = load_traintest_dataset("test")[5]

# +
# Load training dataset
arguments_training, stances_training, conclusions_training, id_training, df_training, df_labels = load_traintest_dataset("training")

# Load test dataset
arguments_test, stances_test, conclusions_test, id_test, df_test, df_labels = load_traintest_dataset("test")


# -

# Preprocessing function
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)
    return text


def classification(test, training, columns) :
    test_df = pd.read_csv(test, sep="\t")
    training_df = pd.read_csv(training, sep="\t")    
    # Concatenate the specified columns
    test_df['combined'] = test_df[columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    training_df['combined'] = training_df[columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    expanded_text_test = test_df['combined'].tolist()
    expanded_text_training = training_df['combined'].tolist()
    # Preprocess the text
    expanded_text_test_preprocessed = [preprocess_text(text) for text in expanded_text_test]
    expanded_text_training_preprocessed = [preprocess_text(text) for text in expanded_text_training]
    # Create the CountVectorizer
    vectorizer = CountVectorizer()
    # Fit the vectorizer to the data and transform the sentences
    X_train = vectorizer.fit_transform(expanded_text_training_preprocessed)
    X_test = vectorizer.transform(expanded_text_test_preprocessed)
    labels_training_drop = labels_training.drop('Argument ID', axis=1)
    labels_training_array = labels_training_drop.values
    labels_test_drop = labels_test.drop('Argument ID', axis=1)
    labels_test_array = labels_test_drop.values
    
    classifier = BornClassifier()
    classifier.fit(X=X_train, y=labels_training_array)
    labels_pred = classifier.predict(X_test)
    # One-hot encode labels_pred, before calculating evaluation metrics
    labels_pred_encode = keras.utils.to_categorical(labels_pred)
    
    f1 = f1_score(labels_test_array, labels_pred_encode, average='samples')
    print(f"F1 Score: {f1}")

    precision = precision_score(labels_test_array, labels_pred_encode, average='samples')
    print(f"Precision: {precision}")
    
    recall = recall_score(labels_test_array, labels_pred_encode, average='samples')
    print(f"Recall: {recall}")

    return classifier, vectorizer, f1, precision, recall, expanded_text_test_preprocessed


def important_words(classifier, vectorizer, expanded_text):
    global_weights = classifier.explain()
    
    feature_names = vectorizer.get_feature_names_out()
    
    # Dictionary with weights and feature names
    feature_weights = dict(zip(feature_names, global_weights))
    
    # Average absolute value for each matrix
    feature_weights_abs_avg = {word: np.abs(matrix).mean() for word, matrix in feature_weights.items()}
    
    # Sorted dictionary
    sorted_features = sorted(feature_weights_abs_avg.items(), key=lambda x: x[1], reverse=True)
    
    # Most important words
    for word, weight in sorted_features[:100]:
        print(f"Word: {word}, Weight: {weight}")

    added_words_weights = {}
    if len(expanded_text) != 0 :
        added_words = [text.split()[-1] for text in expanded_text]
        for word in added_words:
            if word in feature_weights_abs_avg and word not in added_words_weights:
                added_words_weights[word] = feature_weights_abs_avg[word]
        sorted_added_words = sorted(added_words_weights.items(), key=lambda x: x[1], reverse=True)

        return sorted_added_words


def print_and_plot_word_weights(word_weights):
    words, weights = zip(*word_weights)
    for word, weight in word_weights:
        print(f"Added word: {word}, Weight: {weight}")
    
    plt.barh(words, weights, color='skyblue')
    plt.xlabel('Importance')
    plt.title('Added Word Weights')
    plt.gca().invert_yaxis()
    plt.show()


classifier_original = classification("arguments-test.tsv", "arguments-training.tsv", ["Conclusion", "Stance", "Premise"])

f1_original = classifier_original[2]
precision_original = classifier_original[3]
recall_original = classifier_original[4]

important_words(classifier_original[0], classifier_original[1], [])

classifier_expanded = classification("HVD-expanded_dataset_test.tsv", "HVD-expanded_dataset_training.tsv", ["EXPANDED TEXT"])

f1_expanded = classifier_expanded[2]
precision_expanded = classifier_expanded[3]
recall_expanded = classifier_expanded[4]

# Get the word weights
word_weights = important_words(classifier_expanded[0], classifier_expanded[1], classifier_expanded[5])

# Print and plot the word weights
print_and_plot_word_weights(word_weights)

classifier_expanded_adj = classification("HVD-expanded_dataset_test.tsv", "HVD-expanded_dataset_training.tsv", ["EXPANDED TEXT ADJECTIVES"])

f1_expanded_adj = classifier_expanded_adj[2]
precision_expanded_adj = classifier_expanded_adj[3]
recall_expanded_adj = classifier_expanded_adj[4]

word_weights_adj = important_words(classifier_expanded_adj[0], classifier_expanded_adj[1], classifier_expanded_adj[5])

# Print and plot the word weights
print_and_plot_word_weights(word_weights_adj)

# +
models = ['Original dataset', 'Free approach', 'Guided approach']
precision_vals = [precision_original, precision_expanded, precision_expanded_adj]
recall_vals = [recall_original, recall_expanded, recall_expanded_adj]
f1_vals = [f1_original, f1_expanded, f1_expanded_adj]

# Getting the x locations for the groups
x = np.arange(len(models)) 
width = 0.3

fig, ax = plt.subplots(figsize=(6, 4))
bar_width = 0.35
opacity = 0.8

rects1 = ax.bar(x - width, precision_vals, width, label='Precision')
rects2 = ax.bar(x, recall_vals, width, label='Recall')
rects3 = ax.bar(x + width, f1_vals, width, label='F1 Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Scores by model')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.tight_layout()
plt.show()
# -
