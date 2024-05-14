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
