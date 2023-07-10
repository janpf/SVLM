import random
from random import shuffle

import numpy as np
from datasets import load_dataset
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

max_seq_len = 30

dataset = load_dataset("amazon_reviews_multi", "de", split="train[:5%]")


# Tokenize the corpus characterwise by their ASCII code points
tokens = [[ord(char) for char in sentence] for sentence in dataset["review_body"]]
tokens = [[t for t in sample if t < 256] for sample in tokens]

# Create input subsequences and corresponding next character for every single subsequence of tokens
contexts, targets = [], []
for sequence in tokens:
    for i in range(1, len(sequence)):
        # Consider only the last 20 characters
        contexts.append(sequence[max(i - 20, 0) : i])
        targets.append(sequence[i])

# Transform contexts to a matrix representation
X = np.zeros((len(contexts), 256))
# with weights, so that the last character has the highest weight
weights = [1 / (i + 1) for i in range(max_seq_len)]
for i, context in enumerate(contexts):
    for j, ascii_val in enumerate(reversed(context)):
        X[i, ascii_val] += weights[j]

# Split into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.2, random_state=42)
random.seed(42)
tmp = list(zip(X, targets))
shuffle(tmp)
X, targets = zip(*tmp)

# Create and train the multiclass SVM
# model = svm.SVC(decision_function_shape="ovo")
model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)

model.fit(X, targets)


def decode(model, initial_context, length):
    text = initial_context
    for _ in range(length):
        # Transform the current context into the matrix representation
        X = np.zeros((1, 256))
        context = [ord(char) for char in text[-20:]]
        for i, ascii_val in enumerate(reversed(context)):
            X[0, ascii_val] += weights[i]
        # Predict the next character
        prediction = model.predict(X)
        # Decode the prediction
        next_char = chr(int(prediction))
        # Add the predicted character to the text
        text += next_char
    return text


# Use the decoder to generate text
generated_text = decode(model, "A", max_seq_len)
print(generated_text)
